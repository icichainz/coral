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
// 24 x [attention (4) + MoE (3)] + fused final-norm/lm_head GEMV (1)
// [+ GPU argmax (2)] = 170 (172 with argmax) dispatches.
// Prefill currently replays the decode path token by token, 16 tokens per
// command buffer; a batched GEMM prefill is roadmap step 7.
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

class Model {
public:
    // Load config + weights from `model_dir`; binds all shards to the GPU with no copy.
    static std::unique_ptr<Model> load(gpu::Device& dev, const std::string& model_dir);  // throws
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
    virtual int32_t decode_argmax(gpu::CommandStream& cs, KVCache& cache, int32_t token,
                                  gpu::Buffer& logits_out, ForwardStats* stats = nullptr) = 0;

    // Bytes the GPU must read for one decode step (weights + KV) — used to
    // report achieved bandwidth in `coral bench`.
    virtual uint64_t decode_bytes_per_token(uint32_t context_len) const = 0;
};

} // namespace coral
