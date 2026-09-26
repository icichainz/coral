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

Measured on an M2 Max (400 GB/s), gpt-oss-20b, greedy:

| | ms/token | tok/s | GB/s | host encode |
|---|---|---|---|---|
| decode @ ctx ~100-500 | 13.4 | 74.6 | 282 | ~230 us |
| decode @ ctx ~2000 | 13.8 | 72.4 | 276 | ~240 us |
| prefill (token-by-token, 16/submit) | 10.2 | 98 | | |

The final RMSNorm + lm_head GEMV alone is 3.2 ms/token (1.16 GB at 357 GB/s),
so the 8-bit unembedding in step 7 is the largest single win left.

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
  server.h         HTTP + OpenAI-compatible API                [interface]
src/
  gpu/metal_backend.mm   Metal 4 implementation of gpu.h
  gpu/kernels.cpp        includes the generated shaders.inc
  kernels/               MSL: gemv_bf16, gemv_mxfp4, moe, attention, rope, norm,
                         embed, embed_f32, mxfp4, smoke (+ common.h)
  model/attention_ops.*  attention half of a layer (4 dispatches)
  model/moe_ops.*        MoE half of a layer (3 dispatches)
  model/weights.*        zero-copy tensor binding
  loader/safetensors.cpp
  model/config.cpp, model/gptoss.cpp
  tokenizer/tokenizer.cpp, tokenizer/harmony.cpp
  engine/engine.cpp
  server/server.cpp
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
  single-stream decode near 100 tok/s. Quantizing the unembedding to 8-bit
  removes ~0.6 GB/token.

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
   `coral run` and `coral bench`. Verified against mlx-lm golden logits
   (`tests/test_forward.cpp`): same argmax and 16/16 greedy continuation on
   every reference case.
6. **Server**: kqueue HTTP/1.1, SSE streaming, `/v1/chat/completions`,
   `/v1/completions`, `/v1/models`; continuous batching across requests.
7. **Tuning**: prefill GEMMs with `simdgroup_matrix` for long prompts (prefill
   currently replays the decode path per token), 8-bit unembedding,
   overlapping host encode of token t+1 with GPU work of token t, speculative
   decoding, Metal 4 binary archives for instant startup, gpt-oss-120b.
