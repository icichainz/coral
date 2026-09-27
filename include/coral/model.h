// gpt-oss transformer: weights bound zero-copy from safetensors, KV cache,
// and the forward pass expressed as kernel dispatches on a CommandStream.
//
// Per-layer weight layout (HF checkpoint names in parentheses):
//   attn_norm   bf16 [H]                          (input_layernorm.weight)
//   wq/bq       bf16 [Q,H] / [Q]                  (self_attn.q_proj)   Q = heads*head_dim
//   wk/bk       bf16 [KV,H] / [KV]                (self_attn.k_proj)   KV = kv_heads*head_dim
//   wv/bv       bf16 [KV,H] / [KV]                (self_attn.v_proj)
//   wo/bo       bf16 [H,Q] / [H]                  (self_attn.o_proj)
//   sinks       bf16 [heads]                      (self_attn.sinks)  learned attention sink logits
//   mlp_norm    bf16 [H]                          (post_attention_layernorm.weight)
//   router w/b  bf16 [E,H] / [E]                  (mlp.router)
//   gate_up     mxfp4 blocks u8 [E,2I,H/32,16] + scales u8 [E,2I,H/32], bias bf16 [E,2I]
//               rows interleaved: even = gate, odd = up
//   down        mxfp4 blocks u8 [E,H,I/32,16]  + scales u8 [E,H,I/32],  bias bf16 [E,H]
//
// Expert MLP: h = swiglu(x Wgu + b) with gate clamped to (-inf, limit],
// up clamped to [-limit, limit], gate*sigmoid(alpha*gate) * (up+1), alpha=1.702.
// Attention: GQA, RoPE(YaRN), softmax over [sink, scores] per head, sliding
// window of 128 on even layers, full causal on odd layers.
//
// Decode (M = 1) is one command buffer per token: embed (1 dispatch) +
// 24 x [attention (3) + MoE (3)] + fused final-norm/lm_head GEMV (1; int8
// weights by default, see ModelOptions)
// [+ GPU argmax (2)] = 146 (148 with argmax) dispatches.
// Prefill of two or more tokens is batched (src/model/prefill_ops.h): the
// prompt is processed in chunks of up to 1024 positions, one command buffer
// per chunk, with simdgroup-matrix GEMMs for the projections, grouped MXFP4
// GEMMs for the experts and a flash-attention kernel; only the last position
// gets the unembedding. CORAL_PREFILL=token selects the token-by-token decode
// path (16 tokens per command buffer) instead.
// Several sequences decode together through encode_decode_batch (one command
// buffer per step for all of them, 176 dispatches; see src/model/batch_ops.h)
// — the engine's continuous batching (engine.h).
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "coral/config.h"
#include "coral/gpu.h"
#include "coral/safetensors.h"

namespace coral {

// One sequence's KV cache. Full-attention layers grow with the context;
// sliding-window layers are 128-entry ring buffers (24 KB/token total for 20b).
struct KVCache {
    uint32_t capacity = 0;     // max positions for full-attention layers
    uint32_t length = 0;       // positions written so far
    gpu::Buffer k_full, v_full;      // bf16 [layers_full][capacity][kv_heads][head_dim]
    gpu::Buffer k_slide, v_slide;    // bf16 [layers_slide][window][kv_heads][head_dim]
};

// Accumulated (+=) by the synchronous entry points below.
struct ForwardStats {
    double gpu_seconds = 0;      // submit -> completion wall time, summed over submits
    size_t dispatches = 0;
    double encode_seconds = 0;   // host time recording commands (begin -> submit)
    uint32_t submits = 0;
    uint32_t tokens = 0;         // positions processed
};

// Load-time options.
struct ModelOptions {
    // Unembedding precision. Int8 (default): at load, lm_head is quantized on
    // the GPU to int8 with one fp32 scale per row (symmetric absmax/127) into
    // a private buffer (~580 MB for 20b; the bf16 original stays mmap'd but is
    // not read by decode). Halves the bytes of the largest single GEMV;
    // measured on the mlx reference cases: identical argmax, corr(top-64 vs
    // bf16) > 0.9995 (tests/test_forward.cpp). Bf16 = exact checkpoint weights.
    // The environment variable CORAL_LM_HEAD=bf16|int8 overrides this field.
    enum class LmHead { Int8, Bf16 };
    LmHead lm_head = LmHead::Int8;

    // Pipelined greedy decode: decode_argmax() submits step p+1 (token taken
    // on the GPU from step p's argmax) before waiting for step p, hiding host
    // encode and submit latency. Costs one wasted speculative step when the
    // caller stops or continues differently. CORAL_PIPELINE=0|1 overrides.
    bool pipeline_greedy = true;
};

class Model {
public:
    // Load config + weights from `model_dir`; binds all shards to the GPU with no copy.
    static std::unique_ptr<Model> load(gpu::Device& dev, const std::string& model_dir);  // throws
    static std::unique_ptr<Model> load(gpu::Device& dev, const std::string& model_dir, const ModelOptions& opt);
    virtual ~Model() = default;

    virtual const ModelConfig& config() const = 0;

    virtual KVCache new_cache(uint32_t capacity) const = 0;

    // ---- encode-only -------------------------------------------------------
    // Record one decode step for `token` at position cache.length into `cs`
    // (which must be recording: between begin() and submit()) and bump
    // cache.length. Nothing is submitted.
    //   logits_out  fp32 [vocab]; allocated if invalid. nullptr = skip the
    //               unembedding entirely (non-final prefill positions) unless
    //               `argmax_out` is given, which then uses an internal buffer.
    //   argmax_out  int32 greedy token id (lowest index on ties); allocated if
    //               invalid. nullptr = no argmax.
    // Throws std::out_of_range if the cache is full.
    virtual void encode_decode(gpu::CommandStream& cs, KVCache& cache, int32_t token,
                               gpu::Buffer* logits_out, gpu::Buffer* argmax_out = nullptr) = 0;

    // ---- synchronous (begin / encode / submit / wait) ------------------------
    // `cs` must not be recording. Stats, when given, are accumulated.

    // Prefill: run `tokens` at positions [cache.length, cache.length+n) and
    // return logits for the last position (fp32 [vocab]) in `logits_out`.
    virtual void prefill(gpu::CommandStream& cs, KVCache& cache, std::span<const int32_t> tokens,
                         gpu::Buffer& logits_out, ForwardStats* stats = nullptr) = 0;
    // Same, and returns the GPU argmax of the last position's logits.
    virtual int32_t prefill_argmax(gpu::CommandStream& cs, KVCache& cache, std::span<const int32_t> tokens,
                                   gpu::Buffer& logits_out, ForwardStats* stats = nullptr) = 0;

    // Decode: one token at position cache.length, logits for it.
    virtual void decode(gpu::CommandStream& cs, KVCache& cache, int32_t token,
                        gpu::Buffer& logits_out, ForwardStats* stats = nullptr) = 0;
    // Same, plus the GPU argmax (only 4 bytes are read back for greedy decoding).
    // With ModelOptions::pipeline_greedy (default) the step after this one is
    // already submitted when this returns (see gptoss.cpp), `cs` is unused and
    // `logits_out` is re-pointed at an internal buffer holding this step's
    // logits (valid until the next call into the model).
    virtual int32_t decode_argmax(gpu::CommandStream& cs, KVCache& cache, int32_t token,
                                  gpu::Buffer& logits_out, ForwardStats* stats = nullptr) = 0;

    // Bytes the GPU must read for one decode step (weights + KV) — used to
    // report achieved bandwidth in `coral bench`.
    virtual uint64_t decode_bytes_per_token(uint32_t context_len) const = 0;

    // ---- multi-sequence decode (continuous batching) --------------------------
    // One decode step for B = caches.size() independent sequences, 1 <= B <=
    // kMaxDecodeBatch: row i consumes tokens[i] at position caches[i]->length
    // of its own cache and bumps that length. Caches must be distinct.
    //   logits_out  fp32 [B][vocab] (row stride vocab); allocated / grown if
    //               too small. nullptr = internal buffer (needed for argmax).
    //   argmax_out  int32 [B] greedy ids; allocated if invalid. nullptr = none.
    // For B >= 2 the kernels are batched twins of the single-sequence ones
    // with the same per-row arithmetic (src/model/batch_ops.h): each row's
    // hidden states and KV-cache rows are bitwise those of decoding the
    // sequence alone, and so are its logits for B = 2 (for B >= 3 the lm_head
    // runs on simdgroup MMA: ~1e-6 relative). Dense weights are read once per
    // step for all rows, each selected expert's gate_up once per tile of <= 2 rows.
    // B == 1 is the single-sequence path.
    // Throws std::out_of_range if a cache is full (nothing is recorded then).
    static constexpr uint32_t kMaxDecodeBatch = 8;
    virtual void encode_decode_batch(gpu::CommandStream& cs, std::span<KVCache* const> caches,
                                     std::span<const int32_t> tokens, gpu::Buffer* logits_out,
                                     gpu::Buffer* argmax_out);
    // Synchronous form (begin / encode / submit / wait); `cs` must not be recording.
    virtual void decode_batch(gpu::CommandStream& cs, std::span<KVCache* const> caches,
                              std::span<const int32_t> tokens, gpu::Buffer* logits_out,
                              gpu::Buffer* argmax_out, ForwardStats* stats = nullptr);
};

} // namespace coral
