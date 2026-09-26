// Internal: decode-path (M=1) attention and bf16 GEMV encoders for gpt-oss.
//
// Every function here only ENCODES dispatches onto a CommandStream; the caller
// submits. Activations are fp32, attention weights bf16 (zero-copy TensorRefs),
// the KV cache bf16 laid out as in include/coral/model.h.
//
// Kernels (src/kernels/gemv_bf16.metal, rope.metal, attention.metal):
//   gemv_bf16_r{1,2,4,8}, gemv_bf16_norm_r{1,2,4,8}   bf16 GEMV (+bias, +accumulate, +folded RMSNorm)
//   rmsnorm_f32                                      fp32 RMSNorm
//   argmax_f32_partial, argmax_f32_final             GPU greedy argmax
//   rope_yarn_f32                                    standalone RoPE (tests / prefill)
//   attn_qkv_rope                                    rmsnorm + QKV + bias + RoPE + KV-cache write
//   attn_decode_partial, attn_decode_combine         split-K attention with sinks
//
// Attention decode per layer = 4 dispatches:
//   1. attn_qkv_rope        norm(residual) -> q (fp32, roped), k/v (bf16) into the cache slot
//   2. attn_decode_partial  (kv_heads x splits) threadgroups
//   3. attn_decode_combine  merge splits + sink -> attn_out fp32 [Q]
//   4. gemv_bf16_r*         residual += Wo . attn_out + bo
//
// Alignment relied on (checked at encode time, std::invalid_argument on violation):
//   * bf16 weight / norm tensors: GPU address multiple of 8 bytes (the gpt-oss
//     shards put every tensor at 8 mod 16), row length K multiple of 8.
//   * fp32 activation buffers (x, y, q, residual): address multiple of 16.
//   * KV cache rows: multiple of 16 (always true for Model::new_cache buffers).
#pragma once

#include <cstdint>
#include <vector>

#include "coral/config.h"
#include "coral/gpu.h"
#include "coral/model.h"
#include "weights.h"

namespace coral {

// ---- RoPE / YaRN (CPU, exact port of HF _compute_yarn_parameters, truncate=false) ----
struct YarnParams {
    std::vector<float> inv_freq;   // [head_dim/2], fp32 like HF
    float attention_factor = 1.0f; // 0.1*ln(factor)+1, applied to cos and sin
};
YarnParams yarn_parameters(const ModelConfig& cfg);

// float2(cos, sin) * attention_factor, [max_positions][head_dim/2], fp32
// (freq = fp32(inv_freq[j]) * fp32(pos), as HF does in fp32).
gpu::Buffer make_rope_table(gpu::Device& dev, const ModelConfig& cfg, uint32_t max_positions);

// ---- scratch ----
struct AttnScratch {
    gpu::Buffer q;          // fp32 [Q]      roped queries
    gpu::Buffer attn_out;   // fp32 [Q]      attention output (pre o_proj)
    gpu::Buffer partials;   // fp32 [heads][max_splits][head_dim+2]
    gpu::Buffer rope;       // see make_rope_table
    uint32_t max_positions = 0;
    uint32_t max_splits = 0;
};
AttnScratch make_attn_scratch(gpu::Device& dev, const ModelConfig& cfg, uint32_t max_positions);

// Index of layer `layer` among layers of its own kind (sliding / full) — its
// slot in KVCache::k_slide/v_slide or k_full/v_full.
uint32_t kv_slot_for_layer(const ModelConfig& cfg, uint32_t layer);

// Positions per split and number of splits used by attention for `n` attended slots.
struct AttnSplit { uint32_t chunk, splits; };
AttnSplit attn_split(uint32_t n);

// Full attention block for one token at absolute position `pos`:
//   residual += o_proj(attention(rope(qkv(rmsnorm(residual)))))
// Writes K/V for `pos` into the cache (ring slot pos % window for sliding
// layers). Does not touch cache.length. Requires pos < cache.capacity (full
// layers) and pos < scratch.max_positions. 4 dispatches.
void encode_attention_decode(gpu::Device& dev, gpu::CommandStream& cs, const ModelConfig& cfg,
                             const LayerWeights& L, uint32_t layer_index, bool sliding, uint32_t kv_slot,
                             const KVCache& cache, uint32_t pos,
                             const gpu::Buffer& residual, size_t residual_offset, AttnScratch& s);

// The pieces of encode_attention_decode, for timing and finer-grained use:
//   encode_attn_qkv  : dispatch 1 (writes s.q and the cache row for `pos`)
//   encode_attn_core : dispatches 2-3 (reads s.q + cache, writes s.attn_out)
void encode_attn_qkv(gpu::Device& dev, gpu::CommandStream& cs, const ModelConfig& cfg,
                     const LayerWeights& L, uint32_t layer_index, bool sliding, uint32_t kv_slot,
                     const KVCache& cache, uint32_t pos,
                     const gpu::Buffer& residual, size_t residual_offset, AttnScratch& s);
void encode_attn_core(gpu::Device& dev, gpu::CommandStream& cs, const ModelConfig& cfg,
                      const LayerWeights& L, uint32_t layer_index, bool sliding, uint32_t kv_slot,
                      const KVCache& cache, uint32_t pos, AttnScratch& s);

// ---- generic bf16 GEMV ----
//   y[r] = (accumulate ? y[r] : 0) + W[r,:] . xn + bias[r],  r < rows = W.dim(0), K = W.dim(1)
//   xn = norm ? rmsnorm(x) * norm_w : x   (norm folded in: no separate dispatch)
struct GemvOptions {
    const TensorRef* bias = nullptr;    // bf16 [rows]
    bool accumulate = false;            // add into existing y (fused residual add)
    const TensorRef* norm = nullptr;    // bf16 [K] RMSNorm weight
    float eps = 1e-5f;
    uint32_t rows_per_simdgroup = 0;    // 0 = default (tuned); else 1, 2, 4 or 8
};
void encode_gemv_bf16(gpu::Device& dev, gpu::CommandStream& cs, const TensorRef& W,
                      const gpu::Buffer& x, size_t x_offset, const gpu::Buffer& y, size_t y_offset,
                      const GemvOptions& opt = {});

// y[row] = rmsnorm(x[row]) * w, fp32 in/out, `rows` rows of n = w.dim(0).
void encode_rmsnorm_f32(gpu::Device& dev, gpu::CommandStream& cs, const TensorRef& w, float eps,
                        const gpu::Buffer& x, size_t x_offset, const gpu::Buffer& y, size_t y_offset,
                        uint32_t rows = 1);

// out (int32) = argmax(x[0..n)); lowest index on ties. 2 dispatches.
// `scratch` must hold argmax_scratch_bytes() bytes.
size_t argmax_scratch_bytes();
void encode_argmax_f32(gpu::Device& dev, gpu::CommandStream& cs, const gpu::Buffer& x, size_t x_offset,
                       uint32_t n, const gpu::Buffer& out, size_t out_offset, const gpu::Buffer& scratch);

// Standalone RoPE on fp32 [n_heads][64] in place, for position `pos` (tests / prefill).
void encode_rope_f32(gpu::Device& dev, gpu::CommandStream& cs, const AttnScratch& s, uint32_t pos,
                     const gpu::Buffer& x, size_t x_offset, uint32_t n_heads);

} // namespace coral
