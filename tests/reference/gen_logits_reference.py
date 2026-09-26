#!/usr/bin/env python3
"""
Golden logits for end-to-end verification of coral's forward pass.

Runs gpt-oss-20b with mlx-lm (Apple's reference-quality MXFP4 implementation)
on a few short token sequences and writes, for each, the fp32 logits of the
LAST position plus the greedy continuation, to tests/data/logits_reference.json.

This is offline dev tooling only. The C++ engine and its tests never run
Python; tests/test_forward.cpp reads the JSON.

Usage (from the repo root; needs ~14 GB free RAM, ~1 min):
    uv run --with mlx-lm python3 tests/reference/gen_logits_reference.py \
        --model models/gpt-oss-20b --out tests/data/logits_reference.json

Notes:
  * mlx-lm loads the HF safetensors directly (MXFP4 experts). By default the
    bf16 attention/embedding weights and all activations are promoted to fp32
    so the reference is not itself noisy (bf16 vs fp32 MLX differ at ~0.99
    correlation on the top-64 logits).
  * We store only the top-64 logits (id, value) of the last position and the
    argmax, plus a 16-token greedy continuation, to keep the file small.
  * Sequences are given as token ids so the check does not depend on the
    tokenizer (which has its own exact tests).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/gpt-oss-20b")
    ap.add_argument("--out", default="tests/data/logits_reference.json")
    ap.add_argument("--continuation", type=int, default=16)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                    help="dtype for the non-quantized (attention/embedding) weights and activations; "
                         "float32 gives a low-noise reference (MXFP4 experts stay quantized)")
    args = ap.parse_args()

    import mlx.core as mx  # noqa: E402
    from mlx_lm import load  # noqa: E402

    t0 = time.time()
    model, tokenizer = load(args.model)
    if args.dtype == "float32":
        model.set_dtype(mx.float32)
    print(f"loaded in {time.time() - t0:.1f}s ({args.dtype})", file=sys.stderr)

    prompts = [
        # Plain text
        "The capital of France is",
        "def fibonacci(n):\n    ",
        # ~40 tokens of prose: exercises more positions and a mid-sentence continuation
        "In 1969, the Apollo 11 mission landed the first humans on the Moon. Neil Armstrong "
        "and Buzz Aldrin spent about two and a half hours outside the lunar module, collecting",
        # Non-English + digits
        "La photosynthèse transforme le CO2 et l'eau en glucose grâce à l'énergie",
        # Harmony-formatted turn (the format the engine will serve). Rendered
        # with the model's own chat template so it is exactly right.
        None,
    ]
    cases = []
    for p in prompts:
        if p is None:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": "Say hello in one word."}],
                add_generation_prompt=True,
            )
            if isinstance(ids, str):
                ids = tokenizer.encode(ids)
            name = "harmony_hello"
        else:
            ids = tokenizer.encode(p)
            name = p[:24]
        ids = [int(t) for t in ids]

        x = mx.array([ids])
        logits = model(x)  # [1, T, V]
        last = logits[0, -1].astype(mx.float32)
        mx.eval(last)
        vals = last.tolist()
        order = sorted(range(len(vals)), key=lambda i: -vals[i])[: args.topk]
        top = [[int(i), float(vals[i])] for i in order]
        argmax = int(order[0])

        # Greedy continuation, token by token (uses the model's own cache path).
        from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

        cache = make_prompt_cache(model)
        out = model(mx.array([ids]), cache=cache)
        cont = []
        nxt = int(mx.argmax(out[0, -1]).item())
        for _ in range(args.continuation):
            cont.append(nxt)
            out = model(mx.array([[nxt]]), cache=cache)
            nxt = int(mx.argmax(out[0, -1]).item())

        cases.append(
            {
                "name": name,
                "tokens": ids,
                "argmax": argmax,
                "top_logits": top,
                "greedy_continuation": cont,
                "logit_mean": float(mx.mean(last).item()),
                "logit_std": float(mx.sqrt(mx.var(last)).item()),
            }
        )
        print(f"{name!r}: {len(ids)} tokens, argmax {argmax}, cont {cont[:8]}...", file=sys.stderr)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"model": os.path.basename(args.model.rstrip('/')), "generator": "mlx-lm", "dtype": args.dtype, "cases": cases}, f, indent=1)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
