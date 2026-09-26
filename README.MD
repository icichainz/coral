# coral

A high-performance LLM inference engine for Apple Silicon GPUs, built to serve
**gpt-oss-20b** (and later gpt-oss-120b) as fast as the memory bus allows.

Written in standard C++20 and Metal Shading Language. One Objective-C++ file
owns every Metal call. No Python, no MLX, no MPS, no CoreML in the engine.

## Language policy

| Layer | Language | Where |
|---|---|---|
| Kernels | Metal Shading Language (MSL 3.2) | `src/kernels/*.metal` |
| GPU access | Objective-C++, Metal 4 API | `src/gpu/metal_backend.mm` (the only `.mm`) |
| Everything else | C++20 | `src/**`, `include/coral/**` |
| Dev tooling only | Python 3 | `scripts/`, `tests/reference/` |

The engine never executes Python. `scripts/download_model.py` fetches weights;
`tests/reference/` will hold numpy references for kernel validation.

## Why this shape

* **Metal 4** (macOS 26, Apple7+ GPUs): command allocators, argument tables
  (bindless), residency sets and no per-dispatch hazard tracking. A decode
  step is a few hundred dispatches; this keeps the host out of the way.
* **Zero-copy weights**: every safetensors shard is `mmap`'d page-aligned and
  wrapped with `newBufferWithBytesNoCopy`. The 12.8 GB checkpoint is never
  duplicated in memory.
* **Runtime shader compilation**: kernels are embedded as strings at build time
  and compiled by the Metal runtime on first launch (about 60 ms for the
  current set). No Metal Toolchain download is required to build or run.
* **MXFP4 stays packed**: expert weights are decoded in registers inside the
  matmul kernels, never materialized. Attention weights and embeddings are bf16.

## Build

Requirements: macOS 26+, Apple silicon, Xcode 26 command-line tools.

```sh
make            # build/coral and build/coral_tests
make test       # run unit tests (weight tests skip if the model is absent)
./build/coral info
./build/coral selftest
```

Fetch the model (12.8 GB, resumable, sha256-verified):

```sh
python3 scripts/download_model.py openai/gpt-oss-20b --exclude 'metal/*' 'original/*'
./build/coral inspect models/gpt-oss-20b
```

## CLI

```sh
# Chat (Harmony format). Reasoning ("analysis" channel) is printed dimmed to
# stderr, the final answer to stdout. Stops on <|return|> / <|call|>.
./build/coral run models/gpt-oss-20b -p "Why is the sky blue?" --reasoning low
echo "Name three primes." | ./build/coral run models/gpt-oss-20b --temp 0.7 --top-p 0.9 --seed 7

# Raw continuation: the prompt is tokenized as-is (special tokens allowed).
./build/coral run models/gpt-oss-20b --raw -p "The capital of France is" -n 32

# Greedy decode benchmark: prefill time, ms/token, tok/s, achieved GB/s
# (from Model::decode_bytes_per_token), host encode time and dispatches/token.
./build/coral bench models/gpt-oss-20b -n 2048 --report-every 512
./build/coral bench models/gpt-oss-20b -n 128 --ctx 1000     # 1000-token prompt first
```

`run` options: `-p/--prompt` (default: stdin), `-n/--max-tokens`, `--temp`
(0 = greedy, the default), `--top-p`, `--top-k`, `--rep-penalty`, `--seed`,
`--ctx` (KV cache positions; default max(8192, prompt + n)), `--raw`,
`--reasoning low|medium|high`, `--system`.

Measured on an M2 Max (400 GB/s), gpt-oss-20b, greedy (`coral bench`; GB/s
from `Model::decode_bytes_per_token`, 3.14 GB/token with the int8 lm_head):

| | ms/token | tok/s | GB/s | host encode |
|---|---|---|---|---|
| decode @ ctx ~500 | 9.9-10.1 | 99-101 | 313-318 | ~250 us, hidden (pipelined) |
| decode @ ctx ~2250 | 10.5 | 95 | 304 | ~240 us, hidden |
| decode @ ctx ~500, `CORAL_LM_HEAD=bf16` | 11.4 | 88 | 324 | hidden |
| prefill (token-by-token, 16/submit), 1000 tokens | 8.5 | 117 | | |

Before step 7 tuning: 13.4 ms/token (74.5 tok/s, 282 GB/s of 3.72 GB).

Where a decode step goes (per layer, rotating over all 24 layers, M2 Max):

| stage | bytes | time | GB/s |
|---|---|---|---|
| fused rmsnorm + QKV + RoPE + KV write | 29.5 MB | 94 us | 312 |
| attention core (split-K + in-kernel merge), ctx 300 | ~0.6 MB | 21 us | |
| o_proj + residual | 23.6 MB | 75 us | 314 |
| router + top-k | 0.2 MB | 10-12 us | |
| gate_up + SwiGLU (4 experts, MXFP4) | 35.3 MB | 106-110 us | 322-333 |
| down + weighted residual add | 17.6 MB | 61 us | 288 |
| lm_head (int8, final norm folded), once | 580 MB | 1.61 ms | 360 |

For reference, a pure streaming read of a 23.6 MB tensor per dispatch
(rotating over the 24 real `wq` tensors, no cache reuse) reaches 332 GB/s,
and an empty dispatch + barrier costs 2-4 us (`CORAL_MOE_LAB=1 coral_tests
moe_lab_roofline`); the bf16/MXFP4 GEMVs are within ~5% of that.

Knobs (environment, override `ModelOptions`): `CORAL_LM_HEAD=bf16|int8`
(unembedding precision; int8 = per-row absmax scales, quantized on the GPU at
load, +580 MB; on the reference prompts identical argmax to bf16, corr(top-64)
>= 0.99983), `CORAL_PIPELINE=0|1` (pipelined greedy decode: step t+1 is
encoded and submitted, its token read on the GPU from step t's argmax, while
step t runs; one wasted step when generation stops).

Stage benchmarks: `CORAL_MOE_BENCH=1 coral_tests moe_bench`,
`CORAL_ATTN_BENCH=1 [CORAL_ATTN_POS=2000] coral_tests attn_bench`,
`CORAL_MOE_LAB=1 coral_tests moe_lab` (kernel-shape sweep + roofline).

## Serve

```sh
./build/coral serve models/gpt-oss-20b --port 8080 --reasoning low
curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"gpt-oss-20b","messages":[{"role":"user","content":"Hi"}],"stream":true}'
```

The API is OpenAI-compatible: analysis-channel text arrives as
`reasoning_content`, the final channel as `content`, and Harmony tool calls as
`tool_calls`. One request generates at a time; others queue in order.

## Layout

```
include/coral/     public C++ interfaces (one header per module)
  gpu.h            Device / Buffer / Kernel / Args / CommandStream
  kernels.h        embedded MSL access + compile
  safetensors.h    zero-copy mmap loader, bf16 helpers
  config.h         ModelConfig from config.json
  json.h           minimal JSON DOM
  tokenizer.h      BPE tokenizer interface, Utf8Streamer       [interface]
  harmony.h        Harmony chat format render/parse
  model.h          gpt-oss forward pass (decode/prefill), KVCache
  engine.h         generation loop, sampler
  server.h         kqueue HTTP/1.1 + OpenAI-compatible API
src/
  gpu/metal_backend.mm   Metal 4 implementation of gpu.h
  gpu/kernels.cpp        includes the generated shaders.inc
  kernels/               MSL: gemv_bf16, gemv_int8, gemv_mxfp4, moe, attention, rope,
                         norm, embed, embed_f32, mxfp4, smoke, zz_lab (dev roofline)
                         (+ common.h)
  model/attention_ops.*  attention half of a layer (3 dispatches)
  model/gemv_i8.*        int8 lm_head: load-time quantization + GEMV
  model/moe_ops.*        MoE half of a layer (3 dispatches)
  model/weights.*        zero-copy tensor binding
  loader/safetensors.cpp
  model/config.cpp, model/gptoss.cpp
  tokenizer/tokenizer.cpp, tokenizer/harmony.cpp
  engine/engine.cpp
  server/http.cpp, server/openai_api.cpp
  main.cpp               CLI: info | selftest | inspect | run | bench | serve
tools/embed_shaders.cpp  build step: .metal -> shaders.inc
tests/                   unit tests (make test)
scripts/                 Python dev tooling
```

## gpt-oss-20b facts the kernels are built around

* 24 layers, hidden 2880, 64 query heads / 8 KV heads, head_dim 64.
* Alternating **sliding-window (128)** and full attention; learned per-head
  **attention sinks**; RoPE theta 150000 with **YaRN** ×32 (4096 -> 131072).
* MoE: 32 experts, top-4, SwiGLU with clamping (`swiglu_limit` 7, alpha 1.702),
  expert weights **MXFP4** (E2M1 nibbles, E8M0 block-32 scales), biases bf16.
* Vocab 201088 (o200k_harmony); `<|return|>`=200002 and `<|call|>`=200012 stop.
* Bytes read per decoded token, roughly: attention 1.3 GB + active experts
  1.3 GB + unembedding 1.2 GB (bf16). On an M2 Max (about 400 GB/s) that caps
  single-stream decode near 100 tok/s. The int8 unembedding (default)
  removes ~0.58 GB/token.

## Roadmap

1. **Scaffold** — done: build system, Metal 4 backend, runtime kernel
   compile, zero-copy safetensors, config, JSON, first kernels + tests.
2. **Weight binding** — done: `Model::load` maps every tensor to (buffer,
   offset), zero-copy, checkpoint layout (no repack).
3. **Tokenizer + Harmony** — done: o200k_harmony BPE, id-for-id identical to
   tiktoken; render/parse of channels and tool calls identical to the chat
   template.
4. **Kernels, decode path (M=1)** — done: fused rmsnorm+QKV+RoPE+KV-write,
   split-K attention with sinks + sliding ring buffer, fused router, MXFP4 GEMV
   with in-register decode and fused top-k/SwiGLU, fused residual adds, fused
   final-norm + lm_head GEMV, GPU argmax.
5. **Engine** — done: full decode forward pass (170 dispatches, 172 with
   argmax, one command buffer per token), token-by-token prefill (16 tokens
   per command buffer), `Engine::generate` (GPU argmax for greedy, CPU
   sampler with temperature / top-k / top-p / repetition penalty otherwise),
   `coral run` and `coral bench`. Verified against fp32 mlx-lm golden logits
   (`tests/test_forward.cpp`, 5 prompts): same argmax on every case; 4 of 5
   correlate > 0.995 on the top-64 logits with 16/16 greedy tokens identical.
   The fifth ("The capital of France is") has a router near-tie at layer 0
   that a bf16 KV cache flips, so its continuation diverges after one token.
6. **Server** — done: kqueue HTTP/1.1 (one IO thread, one generation worker,
   chunked SSE, keep-alive, backpressure, client-disconnect cancellation),
   `/v1/chat/completions` (streaming and not, `reasoning_content`, tools and
   `tool_calls`, `reasoning_effort`), `/v1/completions`, `/v1/models`,
   `/health`. Continuous batching across requests is future work.
7. **Tuning** — decode path done (13.4 -> ~10.0 ms/token): MXFP4 GEMVs with
   a 4-lanes-per-block mapping (MoE 235 -> 173 us/layer), int8 lm_head
   (3.2 -> 1.6 ms), attention split merge fused into the split kernel and
   top-k fused into the router (172 -> 148 dispatches), pipelined greedy
   decode hiding host encode + submit latency. Next: prefill GEMMs with
   `simdgroup_matrix` for long prompts (prefill currently replays the decode
   path per token), speculative decoding, Metal 4 binary archives for instant
   startup, gpt-oss-120b.
