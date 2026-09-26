#!/usr/bin/env python3
"""
Download a Hugging Face model repository with resumable, verified,
parallel transfers and a live progress display.

Standard library only: no huggingface_hub, no tqdm, no requests.

Usage:
    python3 scripts/download_model.py openai/gpt-oss-20b
    python3 scripts/download_model.py openai/gpt-oss-20b --dest models/gpt-oss-20b \
        --exclude 'metal/*' 'original/*' --workers 4

Features:
  * Lists files through the HF tree API (sizes + LFS sha256 oids).
  * Resumes partial downloads with HTTP Range requests (.part files).
  * Verifies sha256 of every LFS file against the repository's oid.
  * Skips files that are already complete and verified.
  * Parallel workers, retry with exponential backoff.
  * Single-line live progress on a TTY, periodic log lines otherwise.
  * Reads HF_TOKEN from the environment for gated repos / higher rate limits.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Iterable

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
CHUNK = 4 << 20  # 4 MiB read size
USER_AGENT = "coral-download/1.0 (python-urllib)"


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class RemoteFile:
    path: str
    size: int
    sha256: str | None  # only LFS files carry an oid


@dataclass
class FileState:
    remote: RemoteFile
    done_bytes: int = 0
    status: str = "pending"  # pending | downloading | verifying | done | skipped | failed
    error: str | None = None


@dataclass
class Progress:
    files: list[FileState]
    lock: threading.Lock = field(default_factory=threading.Lock)
    started: float = field(default_factory=time.monotonic)
    stop: threading.Event = field(default_factory=threading.Event)
    # sliding window for speed estimate: (t, bytes)
    samples: list[tuple[float, int]] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(f.remote.size for f in self.files)

    def done_bytes(self) -> int:
        return sum(f.done_bytes for f in self.files)

    def add(self, fs: FileState, n: int) -> None:
        with self.lock:
            fs.done_bytes += n

    def snapshot(self):
        with self.lock:
            done = self.done_bytes()
            now = time.monotonic()
            self.samples.append((now, done))
            # keep ~10s window
            while len(self.samples) > 2 and now - self.samples[0][0] > 10:
                self.samples.pop(0)
            if len(self.samples) >= 2:
                (t0, b0), (t1, b1) = self.samples[0], self.samples[-1]
                speed = (b1 - b0) / (t1 - t0) if t1 > t0 else 0.0
            else:
                speed = 0.0
            active = [f for f in self.files if f.status in ("downloading", "verifying")]
            counts = {}
            for f in self.files:
                counts[f.status] = counts.get(f.status, 0) + 1
            return done, speed, active, counts


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:6.1f} {unit}" if unit != "B" else f"{int(n):6d} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def fmt_eta(seconds: float) -> str:
    if seconds <= 0 or seconds != seconds or seconds == float("inf"):
        return "--:--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def auth_headers() -> dict[str, str]:
    h = {"User-Agent": USER_AGENT}
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        tok_file = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(tok_file):
            with open(tok_file) as f:
                tok = f.read().strip() or None
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def http_json(url: str, retries: int = 5):
    """GET a JSON document with retries. Returns (json, headers)."""
    delay = 1.0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=auth_headers())
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r), r.headers
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise SystemExit(
                    f"HTTP {e.code} for {url}\n"
                    "This repository may be gated. Set HF_TOKEN and accept the license on huggingface.co."
                )
            if e.code == 404:
                raise SystemExit(f"Not found: {url}")
            if attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries - 1:
                raise
        time.sleep(delay)
        delay = min(delay * 2, 30)
    raise RuntimeError("unreachable")


def list_repo_files(repo: str, revision: str) -> list[RemoteFile]:
    """Recursively list files with sizes and LFS oids via the tree API (paginated)."""
    files: list[RemoteFile] = []
    url = (
        f"{HF_ENDPOINT}/api/models/{repo}/tree/{urllib.parse.quote(revision, safe='')}"
        "?recursive=true&expand=false"
    )
    while url:
        entries, headers = http_json(url)
        for e in entries:
            if e.get("type") != "file":
                continue
            lfs = e.get("lfs") or {}
            files.append(
                RemoteFile(
                    path=e["path"],
                    size=int(lfs.get("size") or e.get("size") or 0),
                    sha256=lfs.get("oid"),
                )
            )
        # Pagination via Link: <url>; rel="next"
        url = None
        link = headers.get("Link")
        if link:
            for part in link.split(","):
                if 'rel="next"' in part:
                    url = part.split(";")[0].strip().strip("<>")
    files.sort(key=lambda f: f.path)
    return files


def matches_any(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path, p) for p in patterns)


def sha256_of(path: str, fs: FileState, prog: Progress) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Per-file download
# --------------------------------------------------------------------------- #
def download_one(repo: str, revision: str, dest_root: str, fs: FileState, prog: Progress,
                 retries: int) -> None:
    rf = fs.remote
    final = os.path.join(dest_root, rf.path)
    part = final + ".part"
    os.makedirs(os.path.dirname(final), exist_ok=True)

    # Already complete?
    if os.path.exists(final) and os.path.getsize(final) == rf.size:
        if rf.sha256:
            fs.status = "verifying"
            if sha256_of(final, fs, prog) == rf.sha256:
                with prog.lock:
                    fs.done_bytes = rf.size
                fs.status = "skipped"
                return
            os.remove(final)  # corrupt; redownload
        else:
            with prog.lock:
                fs.done_bytes = rf.size
            fs.status = "skipped"
            return

    url = f"{HF_ENDPOINT}/{repo}/resolve/{urllib.parse.quote(revision, safe='')}/{urllib.parse.quote(rf.path)}"
    hasher = hashlib.sha256() if rf.sha256 else None

    # Resume: hash existing partial content first so the final digest is valid.
    existing = os.path.getsize(part) if os.path.exists(part) else 0
    if existing > rf.size:
        os.remove(part)
        existing = 0
    if existing and hasher is not None:
        fs.status = "verifying"
        with open(part, "rb") as f:
            while True:
                b = f.read(CHUNK)
                if not b:
                    break
                hasher.update(b)
    with prog.lock:
        fs.done_bytes = existing

    fs.status = "downloading"
    delay = 1.0
    for attempt in range(retries):
        if prog.stop.is_set():
            fs.status = "failed"
            fs.error = "interrupted"
            return
        try:
            headers = auth_headers()
            if existing:
                headers["Range"] = f"bytes={existing}-"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                if existing and r.status != 206:
                    # Server ignored Range: start over.
                    existing = 0
                    hasher = hashlib.sha256() if rf.sha256 else None
                    with prog.lock:
                        fs.done_bytes = 0
                    mode = "wb"
                else:
                    mode = "ab" if existing else "wb"
                with open(part, mode) as out:
                    while True:
                        if prog.stop.is_set():
                            fs.status = "failed"
                            fs.error = "interrupted"
                            return
                        b = r.read(CHUNK)
                        if not b:
                            break
                        out.write(b)
                        if hasher is not None:
                            hasher.update(b)
                        existing += len(b)
                        prog.add(fs, len(b))
            break  # success
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403, 404):
                fs.status = "failed"
                fs.error = f"HTTP {e.code}"
                return
            if attempt == retries - 1:
                fs.status = "failed"
                fs.error = f"{type(e).__name__}: {e}"
                return
            existing = os.path.getsize(part) if os.path.exists(part) else 0
            time.sleep(delay)
            delay = min(delay * 2, 60)

    got = os.path.getsize(part)
    if got != rf.size:
        fs.status = "failed"
        fs.error = f"size mismatch: got {got}, expected {rf.size}"
        return
    if hasher is not None:
        fs.status = "verifying"
        digest = hasher.hexdigest()
        if digest != rf.sha256:
            fs.status = "failed"
            fs.error = f"sha256 mismatch: {digest[:12]}… != {rf.sha256[:12]}…"
            os.remove(part)
            with prog.lock:
                fs.done_bytes = 0
            return
    os.replace(part, final)
    fs.status = "done"


# --------------------------------------------------------------------------- #
# Progress reporter
# --------------------------------------------------------------------------- #
def reporter(prog: Progress, is_tty: bool, interval: float) -> None:
    total = prog.total_bytes
    last_line_len = 0
    while not prog.stop.is_set():
        done, speed, active, counts = prog.snapshot()
        pct = 100.0 * done / total if total else 100.0
        remaining = total - done
        eta = remaining / speed if speed > 0 else float("inf")
        elapsed = time.monotonic() - prog.started
        finished = counts.get("done", 0) + counts.get("skipped", 0)
        failed = counts.get("failed", 0)

        if is_tty:
            cols = shutil.get_terminal_size((100, 20)).columns
            bar_w = max(10, min(40, cols - 70))
            filled = int(bar_w * pct / 100)
            bar = "█" * filled + "░" * (bar_w - filled)
            head = (
                f"\r[{bar}] {pct:5.1f}%  {human(done)}/{human(total)}  "
                f"{human(speed)}/s  ETA {fmt_eta(eta)}  files {finished}/{len(prog.files)}"
            )
            if failed:
                head += f"  FAILED {failed}"
            # Show the current file(s) on the same line if there is room.
            names = ", ".join(
                f"{os.path.basename(a.remote.path)} {100.0 * a.done_bytes / max(a.remote.size, 1):.0f}%"
                for a in active[:2]
            )
            line = head
            if names and len(head) + len(names) + 4 < cols:
                line = head + "  | " + names
            line = line[:cols]
            sys.stdout.write(line + " " * max(0, last_line_len - len(line)))
            sys.stdout.flush()
            last_line_len = len(line)
        else:
            sys.stdout.write(
                f"[{fmt_eta(elapsed)}] {pct:5.1f}%  {human(done)}/{human(total)}  "
                f"{human(speed)}/s  ETA {fmt_eta(eta)}  files {finished}/{len(prog.files)}"
                + (f"  FAILED {failed}" if failed else "")
                + "\n"
            )
            sys.stdout.flush()
        prog.stop.wait(interval if not is_tty else 0.25)
    if is_tty:
        sys.stdout.write("\n")
        sys.stdout.flush()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", help="Hugging Face repo id, e.g. openai/gpt-oss-20b")
    ap.add_argument("--dest", help="Destination directory (default: models/<repo name>)")
    ap.add_argument("--revision", default="main", help="Branch, tag or commit (default: main)")
    ap.add_argument("--include", nargs="*", default=[], help="Only download paths matching these globs")
    ap.add_argument("--exclude", nargs="*", default=[], help="Skip paths matching these globs")
    ap.add_argument("--workers", type=int, default=4, help="Parallel downloads (default: 4)")
    ap.add_argument("--retries", type=int, default=8, help="Retries per file (default: 8)")
    ap.add_argument("--log-interval", type=float, default=5.0, help="Seconds between log lines when not a TTY")
    ap.add_argument("--dry-run", action="store_true", help="List what would be downloaded and exit")
    args = ap.parse_args()

    dest = args.dest or os.path.join("models", args.repo.split("/")[-1])
    dest = os.path.abspath(dest)

    print(f"Listing {args.repo}@{args.revision} …", flush=True)
    all_files = list_repo_files(args.repo, args.revision)
    selected = [
        f for f in all_files
        if (not args.include or matches_any(f.path, args.include))
        and not matches_any(f.path, args.exclude)
    ]
    if not selected:
        print("Nothing matched the include/exclude filters.")
        return 1

    total = sum(f.size for f in selected)
    print(f"{len(selected)} files, {human(total).strip()} -> {dest}")
    for f in selected:
        print(f"  {human(f.size)}  {f.path}" + ("" if f.sha256 else "  (no checksum)"))
    if args.dry_run:
        return 0

    os.makedirs(dest, exist_ok=True)
    free = shutil.disk_usage(dest).free
    if free < total * 1.05:
        print(f"WARNING: only {human(free).strip()} free on disk; need about {human(total).strip()}.")

    prog = Progress(files=[FileState(remote=f) for f in selected])

    def on_sigint(_sig, _frm):
        prog.stop.set()
        sys.stdout.write("\nInterrupted: partial files are kept as .part and will resume next run.\n")

    signal.signal(signal.SIGINT, on_sigint)
    signal.signal(signal.SIGTERM, on_sigint)

    rep = threading.Thread(target=reporter, args=(prog, sys.stdout.isatty(), args.log_interval), daemon=True)
    rep.start()

    # Large files first so the tail of the run is short small files, not one big one.
    order = sorted(prog.files, key=lambda fs: -fs.remote.size)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(download_one, args.repo, args.revision, dest, fs, prog, args.retries) for fs in order]
        for fut in as_completed(futs):
            exc = fut.exception()
            if exc is not None:
                # Should not happen (download_one records its own failures), but never lose an error.
                sys.stderr.write(f"\nworker crashed: {exc!r}\n")

    prog.stop.set()
    rep.join(timeout=2)

    failed = [f for f in prog.files if f.status == "failed"]
    done = [f for f in prog.files if f.status in ("done", "skipped")]
    elapsed = time.monotonic() - prog.started
    print(f"Finished {len(done)}/{len(prog.files)} files in {fmt_eta(elapsed)} "
          f"({human(prog.done_bytes()).strip()}).")
    if failed:
        print("Failed:")
        for f in failed:
            print(f"  {f.remote.path}: {f.error}")
        print("Re-run the same command to resume.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
