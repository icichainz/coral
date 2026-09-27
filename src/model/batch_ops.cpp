#include "batch_ops.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace coral {

namespace {

// Must match src/kernels (seq_batch.metal and the M = 1 kernels it mirrors).
constexpr uint32_t kHeadDim = 64, kGroup = 8;
constexpr uint32_t kGemvSimdgroups = 8;   // GEMV_SG
constexpr uint32_t kQkvPairs = 2;         // ATTN_QKV_P
constexpr uint32_t kSplitCap = 16;        // attention_ops.cpp kMaxSplits (attn_split)
constexpr uint32_t kArgmaxPartials = 256;
constexpr uint32_t kRouterThreads = 256;
constexpr float kSwigluAlpha = 1.702f;

// Defaults, tuned on M2 Max with tests/test_batch.cpp batch_zz_profile (real
// sequences at ctx ~500-700; ms per step for all 24 layers):
//   QKV chunk 4 at B = 8: 2.8 vs 4.3 (one chunk of 8); B <= 4: one chunk.
//   o_proj rows 4, chunk 4 at B = 8: 2.2 vs 2.8 (rows 2, one chunk).
//   lm_head: GEMV 1.8 / 2.6 / 5.0 at B = 2 / 4 / 8; MMA 1.8 at every B.
//   experts: see sb_gate_up / sb_down in seq_batch.metal.
constexpr uint32_t kDenseChunk = 4;
constexpr uint32_t kGemvRows = 4;
constexpr uint32_t kLmMmaMinRows = 3;
constexpr uint32_t kLmMmaTiles = 4, kLmMmaSimdgroups = 2;
constexpr uint32_t kMoeTileCap = 2;
constexpr uint32_t kGateUpRows = 8, kGateUpSimdgroups = 2;
constexpr uint32_t kDownRows = 4, kDownSimdgroups = 2;   // as the M = 1 down (moe_kernel_config)

void require(bool ok, const std::string& what) {
    if (!ok) throw std::invalid_argument("batch decode: " + what);
}

struct PrepParams { uint32_t K; float eps; };
struct QkvParams { uint32_t H, Q, KV; float eps; uint32_t nb, kv_slot, sliding, window; };
struct AttnParams { uint32_t kv_heads, max_splits, kv_slot, sliding, window, split_cap; float scale; uint32_t pad; };
struct GemvParams { uint32_t rows, K, flags, nb, ldx, ldy; };
struct RouterParams { uint32_t H, E; float eps; uint32_t do_norm, K; };
struct GateUpParams { uint32_t N, Kd, lda, K, n, I; float limit, alpha; };
struct DownParams { uint32_t H, I, K; };
struct I8Params { uint32_t rows, K; float eps; uint32_t nb, ldy, pad0, pad1, pad2; };
struct LmParams { uint32_t rows, K, nb, ldy; };
struct ArgmaxParams { uint32_t n, ldx, groups, pad; };
struct EmbedParams { uint32_t H; };

void check_rows(uint32_t B) {
    static_assert(BatchScratch::kMaxRows * 4 <= 32, "expert tiles are derived by one simdgroup (rows * top-k <= 32)");
    require(B >= 2 && B <= BatchScratch::kMaxRows,
            "batch of " + std::to_string(B) + " rows (supported: 2.." + std::to_string(BatchScratch::kMaxRows) + ")");
}

uint32_t pick(uint32_t knob, uint32_t def) { return knob ? knob : def; }
// Batch rows per threadgroup of a dense GEMV: all of them up to kDenseChunk rows.
uint32_t chunk_of(uint32_t B, uint32_t knob) { return std::min(B, knob ? knob : (B <= kDenseChunk ? B : kDenseChunk)); }

void encode_prep(gpu::Device& dev, gpu::CommandStream& cs, const TensorRef& w, float eps, uint32_t B, BatchScratch& s) {
    cs.dispatch(dev.kernel("sb_norm_prep"),
                gpu::Args().buffer(0, s.x).buffer(1, w.buf, w.offset).buffer(2, s.xg).buffer(3, s.inv)
                           .value(4, PrepParams{uint32_t(w.dim(0)), eps}),
                {B}, {256});
}

} // namespace

BatchScratch make_batch_scratch(gpu::Device& dev, const ModelConfig& c) {
    BatchScratch s;
    const size_t B = BatchScratch::kMaxRows;
    const size_t H = c.hidden_size, Q = c.q_dim(), I = c.intermediate_size, K = c.experts_per_token, E = c.num_experts;
    require(c.head_dim == kHeadDim && c.num_kv_heads * kGroup == c.num_heads, "unsupported head layout");
    require(K * B <= 32 && E <= 256, "unsupported expert count / top-k");
    s.rows = uint32_t(B);
    s.max_splits = kSplitCap;
    s.x = dev.alloc(B * H * 4, true);
    s.xg = dev.alloc(B * H * 4, true);
    s.inv = dev.alloc(B * 4 + 16, true);
    s.q = dev.alloc(B * Q * 4, true);
    s.attn_out = dev.alloc(B * Q * 4, true);
    s.partials = dev.alloc(B * c.num_heads * kSplitCap * (kHeadDim + 2) * 4, true);
    s.attn_counters = dev.alloc(B * c.num_kv_heads * 4, true);
    s.normed = dev.alloc(B * H * 4, true);
    s.router_logits = dev.alloc(B * E * 4, true);
    s.expert_ids = dev.alloc(B * K * 4, true);
    s.probs = dev.alloc(B * K * 4, true);
    s.router_counters = dev.alloc(B * 4 + 16, true);
    s.h = dev.alloc(B * K * I * 4, true);
    s.argmax_part = dev.alloc(B * kArgmaxPartials * 8, true);
    return s;
}

BatchKernelConfig& batch_kernel_config() {
    static BatchKernelConfig k;
    return k;
}

void encode_batch_embed(gpu::Device& dev, gpu::CommandStream& cs, const TensorRef& embed,
                        std::span<const int32_t> tokens, BatchScratch& s) {
    const uint32_t B = uint32_t(tokens.size());
    check_rows(B);
    const uint32_t H = uint32_t(embed.dim(1));
    cs.dispatch(dev.kernel("pf_embed"),
                gpu::Args().buffer(0, embed.buf, embed.offset).bytes(1, tokens.data(), tokens.size_bytes())
                           .buffer(2, s.x).value(3, EmbedParams{H}),
                {(H / 4 + 255) / 256, B}, {256});
}

void encode_batch_attention(gpu::Device& dev, gpu::CommandStream& cs, const ModelConfig& c,
                            const LayerWeights& L, uint32_t kv_slot, std::span<const BatchRow> rows,
                            const AttnScratch& rope, BatchScratch& s) {
    const uint32_t B = uint32_t(rows.size());
    check_rows(B);
    require(c.hidden_size % 8 == 0 && c.q_dim() % 64 == 0 && c.kv_dim() % 64 == 0, "bad dims");
    const uint32_t H = c.hidden_size, Q = c.q_dim(), KV = c.kv_dim(), W = c.sliding_window;
    uint32_t max_splits = 1;
    for (const BatchRow& r : rows) {
        require(r.pos < rope.max_positions, "position beyond the rope table");
        require(r.pos < r.capacity, "position beyond KV cache capacity");
        require(r.k_full && r.v_full && r.k_slide && r.v_slide, "KV cache not allocated");
        const uint32_t n = L.sliding ? std::min(r.pos + 1, W) : r.pos + 1;
        max_splits = std::max(max_splits, attn_split(n).splits);
    }
    require(max_splits <= s.max_splits, "attention split count exceeds the scratch");
    const size_t table = rows.size_bytes();
    const BatchKernelConfig& k = batch_kernel_config();
    const uint32_t mask = k.stage_mask;

    // 1. RMSNorm (sb_norm_prep) + QKV + RoPE + K/V into each row's cache slot.
    if (mask & 1u) {
        encode_prep(dev, cs, L.attn_norm, c.rms_norm_eps, B, s);
        const uint32_t qc = chunk_of(B, k.qkv_chunk);
        const uint32_t pairs = (Q + 2 * KV) / 2, per_tg = kQkvPairs * kGemvSimdgroups;
        cs.dispatch(dev.kernel("sb_qkv_rope_b" + std::to_string(qc)),
                    gpu::Args().buffer(0, s.xg).buffer(1, s.inv)
                               .buffer(2, L.wq.buf, L.wq.offset).buffer(3, L.bq.buf, L.bq.offset)
                               .buffer(4, L.wk.buf, L.wk.offset).buffer(5, L.bk.buf, L.bk.offset)
                               .buffer(6, L.wv.buf, L.wv.offset).buffer(7, L.bv.buf, L.bv.offset)
                               .buffer(8, s.q).bytes(9, rows.data(), table).buffer(10, rope.rope)
                               .value(11, QkvParams{H, Q, KV, c.rms_norm_eps, B, kv_slot, L.sliding ? 1u : 0u, W}),
                    {(B + qc - 1) / qc, (pairs + per_tg - 1) / per_tg}, {kGemvSimdgroups * 32});
    }
    // 2. attention per (kv head, split, row).
    if (mask & 2u) {
        const AttnParams ap{c.num_kv_heads, s.max_splits, kv_slot, L.sliding ? 1u : 0u, W, kSplitCap,
                            1.0f / std::sqrt(float(kHeadDim)), 0};
        cs.dispatch(dev.kernel("sb_attn"),
                    gpu::Args().buffer(0, s.q).bytes(1, rows.data(), table).buffer(2, s.partials)
                               .buffer(3, L.sinks.buf, L.sinks.offset).buffer(4, s.attn_out)
                               .buffer(5, s.attn_counters).value(6, ap),
                    {c.num_kv_heads, max_splits, B}, {kGroup * 32});
    }
    // 3. x[b] += Wo . attn[b] + bo.
    if (mask & 4u) {
        const uint32_t R = pick(k.gemv_rows, kGemvRows), oc = chunk_of(B, k.gemv_chunk);
        const uint32_t rows_o = uint32_t(L.wo.dim(0)), per = R * kGemvSimdgroups;
        cs.dispatch(dev.kernel("sb_gemv_bf16_r" + std::to_string(R) + "_b" + std::to_string(oc)),
                    gpu::Args().buffer(0, L.wo.buf, L.wo.offset).buffer(1, s.attn_out).buffer(2, s.x)
                               .buffer(3, L.bo.buf, L.bo.offset)
                               .value(4, GemvParams{rows_o, uint32_t(L.wo.dim(1)), 3u, B, Q, H}),
                    {(B + oc - 1) / oc, (rows_o + per - 1) / per}, {kGemvSimdgroups * 32});
    }
}

void encode_batch_moe(gpu::Device& dev, gpu::CommandStream& cs, const ModelConfig& c,
                      const LayerWeights& L, uint32_t B, BatchScratch& s) {
    check_rows(B);
    const uint32_t H = c.hidden_size, I = c.intermediate_size, E = c.num_experts, K = c.experts_per_token;
    const BatchKernelConfig& k = batch_kernel_config();
    const uint32_t mask = k.stage_mask;
    // 1. RMSNorm + router + top-k per row.
    if (mask & 8u)
        cs.dispatch(dev.kernel("sb_router"),
                    gpu::Args().buffer(0, s.x).buffer(1, L.mlp_norm.buf, L.mlp_norm.offset)
                               .buffer(2, L.router_w.buf, L.router_w.offset)
                               .buffer(3, L.router_b.buf, L.router_b.offset)
                               .buffer(4, s.normed).buffer(5, s.router_logits)
                               .value(6, RouterParams{H, E, c.rms_norm_eps, 1u, K})
                               .buffer(7, s.expert_ids).buffer(8, s.probs).buffer(9, s.router_counters),
                    {E, B}, {kRouterThreads});
    // 2. gate_up + SwiGLU over capped expert tiles (grid x = tile, an upper
    //    bound of n = B*K; surplus tiles exit).
    const uint32_t n = B * K;
    if (mask & 16u) {
        const uint32_t C = std::min(B, pick(k.moe_cap, kMoeTileCap));
        const uint32_t R = pick(k.gu_rows, kGateUpRows), sg = pick(k.gu_sg, kGateUpSimdgroups), per = R * sg;
        cs.dispatch(dev.kernel("sb_gate_up_r" + std::to_string(R) + "_c" + std::to_string(C)),
                    gpu::Args().buffer(0, s.normed)
                               .buffer(1, L.gate_up_blocks.buf, L.gate_up_blocks.offset)
                               .buffer(2, L.gate_up_scales.buf, L.gate_up_scales.offset)
                               .buffer(3, L.gate_up_bias.buf, L.gate_up_bias.offset)
                               .buffer(4, s.expert_ids).buffer(5, s.h)
                               .value(6, GateUpParams{2 * I, H, H, K, n, I, c.swiglu_limit, kSwigluAlpha}),
                    {n, (2 * I + per - 1) / per}, {sg * 32});
    }
    // 3. down + weighted sum + residual per row (row fastest).
    if (mask & 32u) {
        const uint32_t R = pick(k.dn_rows, kDownRows), sg = pick(k.dn_sg, kDownSimdgroups), per = R * sg;
        cs.dispatch(dev.kernel("sb_down_r" + std::to_string(R)),
                    gpu::Args().buffer(0, s.h)
                               .buffer(1, L.down_blocks.buf, L.down_blocks.offset)
                               .buffer(2, L.down_scales.buf, L.down_scales.offset)
                               .buffer(3, L.down_bias.buf, L.down_bias.offset)
                               .buffer(4, s.expert_ids).buffer(5, s.probs).buffer(6, s.x)
                               .value(7, DownParams{H, I, K}),
                    {B, (H + per - 1) / per}, {sg * 32});
    }
}

void encode_batch_unembed_i8(gpu::Device& dev, gpu::CommandStream& cs, const QuantI8& W, const TensorRef& norm,
                             float eps, uint32_t B, BatchScratch& s, const gpu::Buffer& logits, uint32_t ld) {
    check_rows(B);
    require(W.valid(), "int8 lm_head not quantized");
    require(logits.valid() && logits.size() >= (size_t(B - 1) * ld + W.rows) * 4, "logits buffer too small");
    require(uint32_t(norm.dim(0)) == W.K, "final norm size");
    encode_prep(dev, cs, norm, eps, B, s);
    const BatchKernelConfig& k = batch_kernel_config();
    if (k.lm_mma > 0 || (k.lm_mma < 0 && B >= kLmMmaMinRows)) {
        require(W.K % 64 == 0, "MMA lm_head needs K % 64 == 0");
        const uint32_t per = 8 * kLmMmaTiles * kLmMmaSimdgroups;
        cs.dispatch(dev.kernel("sb_lm_mma_t" + std::to_string(kLmMmaTiles)),
                    gpu::Args().buffer(0, W.q).buffer(1, W.s).buffer(2, s.xg).buffer(3, logits).buffer(4, s.inv)
                               .value(5, LmParams{W.rows, W.K, B, ld}),
                    {(W.rows + per - 1) / per}, {kLmMmaSimdgroups * 32});
        return;
    }
    const uint32_t R = pick(k.lm_rows, kGemvRows), lc = chunk_of(B, k.lm_chunk), per = R * kGemvSimdgroups;
    cs.dispatch(dev.kernel("sb_gemv_i8_norm_r" + std::to_string(R) + "_b" + std::to_string(lc)),
                gpu::Args().buffer(0, W.q).buffer(1, W.s).buffer(2, s.xg).buffer(3, logits).buffer(4, s.inv)
                           .value(5, I8Params{W.rows, W.K, eps, B, ld, 0, 0, 0}),
                {(B + lc - 1) / lc, (W.rows + per - 1) / per}, {kGemvSimdgroups * 32});
}

void encode_batch_argmax(gpu::Device& dev, gpu::CommandStream& cs, const gpu::Buffer& x, uint32_t n, uint32_t ld,
                         uint32_t B, BatchScratch& s, const gpu::Buffer& out) {
    require(B >= 1 && B <= BatchScratch::kMaxRows, "argmax rows");
    require(out.valid() && out.size() >= size_t(B) * 4, "argmax output too small");
    const uint32_t groups = std::min<uint32_t>(kArgmaxPartials, (n + 1023) / 1024);   // as encode_argmax_f32
    const ArgmaxParams p{n, ld, groups, 0};
    const size_t vi = size_t(BatchScratch::kMaxRows) * kArgmaxPartials * 4;
    cs.dispatch(dev.kernel("sb_argmax_partial"),
                gpu::Args().buffer(0, x).buffer(1, s.argmax_part, 0).buffer(2, s.argmax_part, vi).value(3, p),
                {groups, B}, {1024});
    cs.dispatch(dev.kernel("sb_argmax_final"),
                gpu::Args().buffer(0, s.argmax_part, 0).buffer(1, s.argmax_part, vi).buffer(2, out).value(3, p),
                {B}, {256});
}

} // namespace coral
