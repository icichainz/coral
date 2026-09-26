// Internal: batched prefill (M > 1 positions per pass) encoders for gpt-oss.
//
// Everything here only ENCODES dispatches; the caller submits. Activations are
// fp32 row-major [M][dim], weights bound zero-copy as in attention_ops.h /
// moe_ops.h. Kernels: src/kernels/prefill_gemm.metal, prefill_attn.metal,
// prefill_moe.metal.
//
// One chunk of M positions [pos0, pos0+M) through one decoder layer:
//
//  attention half (encode_prefill_attention), 5 dispatches (+1 on sliding layers):
//   1. rmsnorm_f32         normed = rmsnorm(x) * attn_norm                       [M][H]
//   2. pf_gemm_bf16_m*     qkv = normed · [Wq;Wk;Wv]^T + [bq;bk;bv] (3 segments) [M][Q+2KV]
//   3. pf_rope_kv          RoPE(q) in place; RoPE(k), v -> bf16 staging [M][KV]
//                          (+ the full-layer cache rows pos0..pos0+M-1)
//   4. pf_attention        causal / sliding flash attention with sinks, GQA,
//                          simdgroup MMA; keys < pos0 from the cache, keys >= pos0
//                          from the staging rows -> attn [M][Q]
//   5. pf_ring_write       (sliding layers only) staging rows of the last
//                          min(M, window) positions -> ring slots pos % window.
//                          Runs AFTER attention: a chunk longer than the window
//                          overwrites ring slots that earlier queries of the same
//                          chunk still need, so attention never reads the ring
//                          for positions >= pos0.
//   6. pf_gemm_bf16_m*     x += attn · Wo^T + bo
//
//  MoE half (encode_prefill_moe), 6 dispatches:
//   1. rmsnorm_f32         normed = rmsnorm(x) * mlp_norm
//   2. pf_router           logits = normed · Wr^T + br; top-k + softmax per row
//   3. pf_moe_sort         counting sort of the M*k (row, slot) pairs by expert
//                          -> list, tile table [(expert, list start, rows)]
//   4. pf_moe_gate_up_m*   grouped MXFP4 GEMM over the tile table, fused
//                          bias + clamp + SwiGLU -> h [M*k][I] (list order)
//   5. pf_moe_down_m*      grouped MXFP4 GEMM, + bias, * routing prob
//                          -> y [M][k][H] (one slot per selected expert)
//   6. pf_moe_reduce       x[m] += y[m][0] + ... + y[m][k-1]  (fixed order: deterministic)
//
// _m16/_m32/_m64: GEMM row-tile height (dense: 16 for M <= 16, else 32; MoE:
// by expected rows per expert, see moe_row_tile). Everything is fp32 math on
// simdgroup_matrix<float> with bf16/MXFP4 weights staged exactly, so a row's
// result does not depend on which other rows share its chunk or tile: the
// batched prefill is bitwise independent of the chunking.
#pragma once

#include <cstddef>
#include <cstdint>

#include "attention_ops.h"
#include "coral/config.h"
#include "coral/gpu.h"
#include "coral/model.h"
#include "weights.h"

namespace coral {

struct PrefillScratch {
    uint32_t max_rows = 0;     // chunk capacity M
    gpu::Buffer x;             // fp32 [M][H]      residual stream of the chunk
    gpu::Buffer normed;        // fp32 [M][H]
    gpu::Buffer qkv;           // fp32 [M][Q+2KV]  (q roped in place)
    gpu::Buffer attn;          // fp32 [M][Q]
    gpu::Buffer kst, vst;      // bf16 [M][KV]     this chunk's K (roped) / V
    gpu::Buffer ids;           // int32 [M]        token ids (host-written)
    gpu::Buffer top_ids;       // int32 [M][k]     selected experts
    gpu::Buffer top_probs;     // fp32 [M][k]
    gpu::Buffer list;          // uint [M*k]       (row << 8 | slot), grouped by expert
    gpu::Buffer tiles;         // uint4 [1 + max_tiles]: [0].x = tile count, then (expert, list start, rows, 0)
    gpu::Buffer h;             // fp32 [M*k][I]    SwiGLU activations in list order
    gpu::Buffer y;             // fp32 [M][k][H]   weighted expert outputs per slot
    uint32_t max_tiles = 0;
};
PrefillScratch make_prefill_scratch(gpu::Device& dev, const ModelConfig& cfg, uint32_t max_rows);

// ---- dense bf16 GEMM ----------------------------------------------------------
//   Y[m][n] = (accumulate ? Y[m][n] : 0) + sum_k X[m][k] * W[n][k] + bias[n]
// W given as 1-3 segments stacked along N (e.g. Wq, Wk, Wv), each [rows_i][K];
// every segment's rows must be a multiple of 64, K a multiple of 32. X/Y fp32
// with row strides ldx/ldy (floats, multiples of 4, 16-byte aligned).
struct GemmSegment { const TensorRef* W = nullptr; const TensorRef* bias = nullptr; };
void encode_gemm_bf16(gpu::Device& dev, gpu::CommandStream& cs, const GemmSegment* segs, uint32_t n_segs,
                      const gpu::Buffer& X, size_t x_offset, uint32_t ldx,
                      const gpu::Buffer& Y, size_t y_offset, uint32_t ldy, uint32_t M, bool accumulate);

// ---- attention half -------------------------------------------------------------
// x (s.x) rows [0, M) hold positions [pos0, pos0+M). Requires pos0 + M <=
// cache.capacity and <= rope.max_positions; cache rows < pos0 must be valid.
// Does not touch cache.length.
void encode_prefill_attention(gpu::Device& dev, gpu::CommandStream& cs, const ModelConfig& cfg,
                              const LayerWeights& L, uint32_t layer_index, uint32_t kv_slot,
                              const KVCache& cache, uint32_t pos0, uint32_t M,
                              const AttnScratch& rope, PrefillScratch& s);

// Pieces (tests): dispatches 3, 4 and 5 above. qkv must already hold the
// projections (un-roped).
void encode_prefill_rope_kv(gpu::Device& dev, gpu::CommandStream& cs, const ModelConfig& cfg, bool sliding,
                            uint32_t kv_slot, const KVCache& cache, uint32_t pos0, uint32_t M,
                            const AttnScratch& rope, PrefillScratch& s);
void encode_prefill_attn_core(gpu::Device& dev, gpu::CommandStream& cs, const ModelConfig& cfg,
                              const LayerWeights& L, bool sliding, uint32_t kv_slot, const KVCache& cache,
                              uint32_t pos0, uint32_t M, PrefillScratch& s);
void encode_prefill_ring_write(gpu::Device& dev, gpu::CommandStream& cs, const ModelConfig& cfg,
                               uint32_t kv_slot, const KVCache& cache, uint32_t pos0, uint32_t M,
                               PrefillScratch& s);

// ---- MoE half --------------------------------------------------------------------
// s.x[m] += moe(rmsnorm(s.x[m])) for m < M.
void encode_prefill_moe(gpu::Device& dev, gpu::CommandStream& cs, const ModelConfig& cfg,
                        const LayerWeights& L, uint32_t M, PrefillScratch& s);
// Its two halves: dispatches 1-3 (norm, router, sort) and 4-6 (experts, reduce).
void encode_prefill_moe_route(gpu::Device& dev, gpu::CommandStream& cs, const ModelConfig& cfg,
                              const LayerWeights& L, uint32_t M, PrefillScratch& s);
void encode_prefill_moe_experts(gpu::Device& dev, gpu::CommandStream& cs, const ModelConfig& cfg,
                                const LayerWeights& L, uint32_t M, PrefillScratch& s);

// MoE row tile (16, 32 or 64) used for a chunk of M rows (CORAL_PF_MOE_BM overrides).
uint32_t moe_row_tile(const ModelConfig& cfg, uint32_t M);

// ---- misc ------------------------------------------------------------------------
// s.x[m] = embed[s.ids[m]] (fp32) for m < M.
void encode_prefill_embed(gpu::Device& dev, gpu::CommandStream& cs, const TensorRef& embed, uint32_t M,
                          PrefillScratch& s);

} // namespace coral
