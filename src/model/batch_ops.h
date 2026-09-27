// Internal: multi-sequence decode ("continuous batching") encoders for gpt-oss.
//
// One step decodes B independent sequences (2 <= B <= kMaxRows), row b at its
// own position in its own KVCache. Kernels: src/kernels/seq_batch.metal.
//
// Per step (174 dispatches, 176 with argmax; the M = 1 path has 148):
//   pf_embed                         x[b] = embed[ids[b]]
//   24 x attention (encode_batch_attention), 4:
//     sb_norm_prep       inv[b], x[b] * attn_norm (the values the fused M = 1 kernel computes)
//     sb_qkv_rope_b*     QKV (+bias) + RoPE, K/V into row b's cache slot
//     sb_attn            split-K attention with sinks, grid (kv_heads, splits, B)
//     sb_gemv_bf16_r*_b* x[b] += Wo . attn[b] + bo
//   24 x MoE (encode_batch_moe), 3:
//     sb_router          rmsnorm + router + top-k per row, grid (E, B)
//     sb_gate_up_*       grid (tiles, blocks): tiles of <= 2 entries of one expert,
//                        its weights decoded once for the tile's rows
//     sb_down_r*         grid (B, blocks), row fastest: x[b] += sum_k p_k (W_d h_k + b_d)
//   sb_norm_prep + sb_lm_mma (B >= 3) / sb_gemv_i8_norm (B = 2)  final norm + int8 lm_head
//   sb_argmax_partial/final (optional)
//
// Dense weights are read once per step for all rows, each selected expert's
// gate_up once per tile, down shared through the cache. Everything but the
// B >= 3 lm_head keeps the M = 1
// per-row arithmetic: rows are bitwise the single-sequence decode (hidden
// states, KV cache; logits too for B = 2), see tests/test_batch.cpp.
#pragma once

#include <cstdint>
#include <span>

#include "attention_ops.h"
#include "coral/config.h"
#include "coral/gpu.h"
#include "coral/model.h"
#include "gemv_i8.h"
#include "weights.h"

namespace coral {

// Row table entry (matches SbRow in seq_batch.metal).
struct BatchRow {
    uint64_t k_full = 0, v_full = 0, k_slide = 0, v_slide = 0;   // GPU addresses of the cache buffers
    uint32_t pos = 0;                                            // position decoded this step
    uint32_t capacity = 0;                                       // full-layer slots
    uint32_t pad0 = 0, pad1 = 0;
};
static_assert(sizeof(BatchRow) == 48);

struct BatchScratch {
    static constexpr uint32_t kMaxRows = 8;   // SB_MAX_ROWS
    uint32_t rows = 0;          // capacity (kMaxRows)
    gpu::Buffer x;              // fp32 [B][H]        residual streams
    gpu::Buffer xg;             // fp32 [B][H]        x * norm weight (sb_norm_prep)
    gpu::Buffer inv;            // fp32 [B]           rsqrt(mean(x^2) + eps)
    gpu::Buffer q;              // fp32 [B][Q]
    gpu::Buffer attn_out;       // fp32 [B][Q]
    gpu::Buffer partials;       // fp32 [B][heads][max_splits][66]
    gpu::Buffer attn_counters;  // uint [B][kv_heads] (zero between dispatches)
    gpu::Buffer normed;         // fp32 [B][H]
    gpu::Buffer router_logits;  // fp32 [B][E]
    gpu::Buffer expert_ids;     // int32 [B][K]
    gpu::Buffer probs;          // fp32 [B][K]
    gpu::Buffer router_counters;// uint [B]
    gpu::Buffer h;              // fp32 [B][K][I]
    gpu::Buffer argmax_part;    // fp32 [B][256] values, then uint [B][256] indices
    uint32_t max_splits = 0;
};
BatchScratch make_batch_scratch(gpu::Device& dev, const ModelConfig& cfg);

// Kernel shape knobs (tests / tuning); 0 / -1 = the tuned default (batch_ops.cpp).
struct BatchKernelConfig {
    uint32_t qkv_chunk = 0;     // batch rows per threadgroup in the QKV GEMV (1..8)
    uint32_t gemv_rows = 0;     // o_proj rows per simdgroup (2 or 4)
    uint32_t gemv_chunk = 0;    // batch rows per threadgroup in o_proj
    int32_t lm_mma = -1;        // lm_head: 1 MMA, 0 GEMV (bitwise), -1 by B
    uint32_t lm_rows = 0;       // lm_head GEMV rows per simdgroup (2 or 4)
    uint32_t lm_chunk = 0;      // batch rows per threadgroup in the lm_head GEMV
    uint32_t moe_cap = 0;       // entries per gate_up expert tile (1 or 2)
    uint32_t gu_rows = 0, gu_sg = 0;   // gate_up rows per simdgroup (4 or 8), simdgroups per threadgroup
    uint32_t dn_rows = 0, dn_sg = 0;   // down rows per simdgroup (2 or 4), simdgroups per threadgroup
    // Profiling only: layer stages to record (bit 0 norm + qkv, 1 attention,
    // 2 o_proj, 3 router, 4 gate_up, 5 down). Anything but ~0 gives
    // wrong results.
    uint32_t stage_mask = ~0u;
};
BatchKernelConfig& batch_kernel_config();

// x[b] = embed[tokens[b]] for b < B (ids staged as inline bytes: no host
// buffer to race with a step still in flight).
void encode_batch_embed(gpu::Device& dev, gpu::CommandStream& cs, const TensorRef& embed,
                        std::span<const int32_t> tokens, BatchScratch& s);

// Attention half of one layer for B rows: x[b] += o_proj(attn(qkv(rmsnorm(x[b])))),
// K/V of row b written at rows[b].pos into its cache. `rope` provides the
// rope table (must cover every row's position). 4 dispatches.
void encode_batch_attention(gpu::Device& dev, gpu::CommandStream& cs, const ModelConfig& cfg,
                            const LayerWeights& L, uint32_t kv_slot, std::span<const BatchRow> rows,
                            const AttnScratch& rope, BatchScratch& s);

// MoE half of one layer for B rows: x[b] += moe(rmsnorm(x[b])). 3 dispatches.
void encode_batch_moe(gpu::Device& dev, gpu::CommandStream& cs, const ModelConfig& cfg,
                      const LayerWeights& L, uint32_t B, BatchScratch& s);

// logits[b] (fp32, row stride `ld` floats) = lm_head_i8 . rmsnorm(x[b]) * final_norm. 2 dispatches.
void encode_batch_unembed_i8(gpu::Device& dev, gpu::CommandStream& cs, const QuantI8& W, const TensorRef& norm,
                             float eps, uint32_t B, BatchScratch& s, const gpu::Buffer& logits, uint32_t ld);

// out[b] = argmax(x[b][0..n)) (int32; lowest index on ties). 2 dispatches.
void encode_batch_argmax(gpu::Device& dev, gpu::CommandStream& cs, const gpu::Buffer& x, uint32_t n, uint32_t ld,
                         uint32_t B, BatchScratch& s, const gpu::Buffer& out);

} // namespace coral
