#include "prefill_ops.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace coral {

namespace {

// Must match src/kernels/prefill_gemm.metal / prefill_attn.metal / prefill_moe.metal.
constexpr uint32_t kBN = 64, kBK = 32, kGemmThreads = 128;   // row tile BM: 16, 32 or 64

// Dense GEMM row tile: 16 for M <= 16, else 32 (measured on M2 Max,
// prefill_zz_profile: BM 32 beats 64 by 3-8% at M = 128..1024 — smaller
// threadgroups keep more simdgroups resident; CORAL_PF_GEMM_BM overrides).
constexpr uint32_t kMoeBm32Rows = 24, kMoeBm64Rows = 1u << 30;   // see moe_row_tile
uint32_t dense_bm(uint32_t M) {
    if (const char* e = std::getenv("CORAL_PF_GEMM_BM")) {
        const int v = std::atoi(e);
        if (v == 16 || v == 32 || v == 64) return uint32_t(v);
    }
    return M <= 16 ? 16 : 32;
}
constexpr uint32_t kHeadDim = 64;

void require(bool ok, const std::string& what) {
    if (!ok) throw std::invalid_argument("prefill: " + what);
}
uint64_t addr(const gpu::Buffer& b, size_t off) { return b.gpu_address() + off; }

void check_bf16(const TensorRef& t, const char* what, size_t align = 8) {
    require(t.buf.valid(), std::string(what) + ": tensor '" + t.name + "' is not bound to the GPU");
    require(t.dtype == DType::BF16, std::string(what) + ": tensor '" + t.name + "' must be bf16");
    require(addr(t.buf, t.offset) % align == 0, std::string(what) + ": tensor '" + t.name + "' misaligned");
}
void check_f32(const gpu::Buffer& b, size_t off, size_t count, const char* what) {
    require(b.valid(), std::string(what) + ": buffer not allocated");
    require(addr(b, off) % 16 == 0, std::string(what) + ": fp32 buffer must be 16-byte aligned");
    require(off + count * 4 <= b.size(), std::string(what) + ": buffer too small");
}

struct GemmParams { uint32_t M, N, K, ldx, ldy, n1, n2, flags; };
struct RopeParams { uint32_t ldq, Q, KV, full; };
struct AttnParams { uint32_t M, pos0, ldq, Q, KV, window, sliding; float scale; };
struct RingParams { uint32_t M, pos0, KV, window, first; };
struct RmsParams { uint32_t n; float eps; };
struct RouterParams { uint32_t H, E, K; };
struct SortParams { uint32_t n, E, K, max_tiles, bm; };
struct MoeParams { uint32_t N, Kd, lda, K, H; float limit, alpha; uint32_t I; };
struct ReduceParams { uint32_t M, H, K; };
struct EmbedParams { uint32_t H; };
constexpr float kSwigluAlpha = 1.702f;

void check_rows(const PrefillScratch& s, uint32_t M) {
    require(M > 0 && M <= s.max_rows, "chunk of " + std::to_string(M) + " rows exceeds the scratch capacity " +
                                          std::to_string(s.max_rows));
}

void check_layout(const ModelConfig& c) {
    require(c.head_dim == kHeadDim && c.num_kv_heads * 8 == c.num_heads, "unsupported head layout");
    require(c.hidden_size % 64 == 0 && c.q_dim() % 64 == 0 && c.kv_dim() % 64 == 0 &&
            c.intermediate_size % 64 == 0, "dimensions must be multiples of 64");
    require(c.num_experts <= 256 && c.experts_per_token <= 32 && c.experts_per_token <= c.num_experts,
            "unsupported expert count / top-k");
}

struct CacheLayer { const gpu::Buffer* k; const gpu::Buffer* v; size_t base; uint32_t slots; };
CacheLayer cache_layer(const ModelConfig& c, bool sliding, uint32_t kv_slot, const KVCache& cache) {
    const size_t row = size_t(c.kv_dim()) * 2;
    CacheLayer g;
    g.k = sliding ? &cache.k_slide : &cache.k_full;
    g.v = sliding ? &cache.v_slide : &cache.v_full;
    g.slots = sliding ? c.sliding_window : cache.capacity;
    require(g.k->valid() && g.v->valid(), "KV cache not allocated");
    g.base = size_t(kv_slot) * g.slots * row;
    require(g.base + size_t(g.slots) * row <= std::min(g.k->size(), g.v->size()), "kv_slot out of range");
    return g;
}

} // namespace

PrefillScratch make_prefill_scratch(gpu::Device& dev, const ModelConfig& c, uint32_t M) {
    PrefillScratch s;
    const size_t H = c.hidden_size, Q = c.q_dim(), KV = c.kv_dim(), I = c.intermediate_size;
    const size_t k = c.experts_per_token, E = c.num_experts;
    s.max_rows = M;
    s.x = dev.alloc(M * H * 4, true);
    s.normed = dev.alloc(M * H * 4, true);
    s.qkv = dev.alloc(M * (Q + 2 * KV) * 4, true);
    s.attn = dev.alloc(M * Q * 4, true);
    s.kst = dev.alloc(M * KV * 2, true);
    s.vst = dev.alloc(M * KV * 2, true);
    s.ids = dev.alloc(M * 4, true);
    s.top_ids = dev.alloc(M * k * 4, true);
    s.top_probs = dev.alloc(M * k * 4, true);
    s.list = dev.alloc(M * k * 4, true);
    s.max_tiles = uint32_t((M * k + 15) / 16 + E);   // smallest MoE row tile (16)
    s.tiles = dev.alloc(size_t(s.max_tiles + 1) * 16, true);
    s.h = dev.alloc(M * k * I * 4, true);
    s.y = dev.alloc(M * k * H * 4, true);
    return s;
}

void encode_gemm_bf16(gpu::Device& dev, gpu::CommandStream& cs, const GemmSegment* segs, uint32_t n_segs,
                      const gpu::Buffer& X, size_t x_offset, uint32_t ldx,
                      const gpu::Buffer& Y, size_t y_offset, uint32_t ldy, uint32_t M, bool accumulate) {
    require(n_segs >= 1 && n_segs <= 3 && M > 0, "gemm: 1-3 segments and M > 0");
    const uint32_t K = uint32_t(segs[0].W->dim(1));
    require(K % kBK == 0, "gemm: K must be a multiple of 32");
    uint32_t bounds[3] = {0, 0, 0}, N = 0;
    const bool bias = segs[0].bias != nullptr;
    for (uint32_t i = 0; i < n_segs; ++i) {
        const TensorRef& W = *segs[i].W;
        check_bf16(W, "gemm W");
        require(W.shape.size() == 2 && uint32_t(W.dim(1)) == K, "gemm: segments must share K");
        require(W.dim(0) % kBN == 0, "gemm: segment rows must be a multiple of 64");
        require((segs[i].bias != nullptr) == bias, "gemm: bias on all segments or none");
        if (bias) {
            check_bf16(*segs[i].bias, "gemm bias", 2);
            require(segs[i].bias->nbytes >= size_t(W.dim(0)) * 2, "gemm: bias too small");
        }
        bounds[i] = N;
        N += uint32_t(W.dim(0));
    }
    require(ldx % 4 == 0 && ldy % 4 == 0 && ldx >= K && ldy >= N, "gemm: bad leading dimensions");
    check_f32(X, x_offset, size_t(M - 1) * ldx + K, "gemm X");
    check_f32(Y, y_offset, size_t(M - 1) * ldy + N, "gemm Y");
    const uint32_t n1 = n_segs > 1 ? bounds[1] : N, n2 = n_segs > 2 ? bounds[2] : N;
    const GemmParams p{M, N, K, ldx, ldy, n1, n2, (bias ? 1u : 0u) | (accumulate ? 2u : 0u)};
    auto seg = [&](uint32_t i) -> const TensorRef& { return *segs[std::min(i, n_segs - 1)].W; };
    auto bseg = [&](uint32_t i) -> const TensorRef& {
        const GemmSegment& g = segs[std::min(i, n_segs - 1)];
        return g.bias ? *g.bias : *g.W;
    };
    const uint32_t bm = dense_bm(M);
    cs.dispatch(dev.kernel("pf_gemm_bf16_m" + std::to_string(bm)),
                gpu::Args().buffer(0, X, x_offset)
                           .buffer(1, seg(0).buf, seg(0).offset).buffer(2, seg(1).buf, seg(1).offset)
                           .buffer(3, seg(2).buf, seg(2).offset)
                           .buffer(4, bseg(0).buf, bseg(0).offset).buffer(5, bseg(1).buf, bseg(1).offset)
                           .buffer(6, bseg(2).buf, bseg(2).offset)
                           .buffer(7, Y, y_offset).value(8, p),
                {N / kBN, (M + bm - 1) / bm}, {kGemmThreads});
}


void encode_prefill_embed(gpu::Device& dev, gpu::CommandStream& cs, const TensorRef& embed, uint32_t M,
                          PrefillScratch& s) {
    check_rows(s, M);
    check_bf16(embed, "embed", 2);
    const uint32_t H = uint32_t(embed.dim(1));
    require(H % 4 == 0, "embed: H must be a multiple of 4");
    cs.dispatch(dev.kernel("pf_embed"),
                gpu::Args().buffer(0, embed.buf, embed.offset).buffer(1, s.ids).buffer(2, s.x).value(3, EmbedParams{H}),
                {(H / 4 + 255) / 256, M}, {256});
}

void encode_prefill_rope_kv(gpu::Device& dev, gpu::CommandStream& cs, const ModelConfig& c, bool sliding,
                            uint32_t kv_slot, const KVCache& cache, uint32_t pos0, uint32_t M,
                            const AttnScratch& rope, PrefillScratch& s) {
    check_rows(s, M);
    check_layout(c);
    require(pos0 + M <= rope.max_positions, "positions beyond the rope table");
    require(sliding || pos0 + M <= cache.capacity, "positions beyond KV cache capacity");
    const CacheLayer g = cache_layer(c, sliding, kv_slot, cache);
    const uint32_t Q = c.q_dim(), KV = c.kv_dim();
    const size_t row_off = sliding ? g.base : g.base + size_t(pos0) * KV * 2;
    cs.dispatch(dev.kernel("pf_rope_kv"),
                gpu::Args().buffer(0, s.qkv).buffer(1, rope.rope, size_t(pos0) * kHeadDim / 2 * 8)
                           .buffer(2, s.kst).buffer(3, s.vst).buffer(4, *g.k, row_off).buffer(5, *g.v, row_off)
                           .value(6, RopeParams{Q + 2 * KV, Q, KV, sliding ? 0u : 1u}),
                {M}, {256});
}

void encode_prefill_attn_core(gpu::Device& dev, gpu::CommandStream& cs, const ModelConfig& c,
                              const LayerWeights& L, bool sliding, uint32_t kv_slot, const KVCache& cache,
                              uint32_t pos0, uint32_t M, PrefillScratch& s) {
    check_rows(s, M);
    check_layout(c);
    require(L.sliding == sliding, "sliding flag does not match the layer weights");
    check_bf16(L.sinks, "sinks", 2);
    const CacheLayer g = cache_layer(c, sliding, kv_slot, cache);
    const uint32_t Q = c.q_dim(), KV = c.kv_dim();
    const AttnParams p{M, pos0, Q + 2 * KV, Q, KV, c.sliding_window, sliding ? 1u : 0u,
                       1.0f / std::sqrt(float(kHeadDim))};
    cs.dispatch(dev.kernel("pf_attention"),
                gpu::Args().buffer(0, s.qkv).buffer(1, s.kst).buffer(2, s.vst)
                           .buffer(3, *g.k, g.base).buffer(4, *g.v, g.base)
                           .buffer(5, L.sinks.buf, L.sinks.offset).buffer(6, s.attn).value(7, p),
                {(M + 31) / 32, c.num_heads}, {128});
}

void encode_prefill_ring_write(gpu::Device& dev, gpu::CommandStream& cs, const ModelConfig& c,
                               uint32_t kv_slot, const KVCache& cache, uint32_t pos0, uint32_t M,
                               PrefillScratch& s) {
    check_rows(s, M);
    const CacheLayer g = cache_layer(c, true, kv_slot, cache);
    const uint32_t W = c.sliding_window, KV = c.kv_dim();
    const uint32_t first = M > W ? M - W : 0, n = M - first;
    cs.dispatch_threads(dev.kernel("pf_ring_write"),
                        gpu::Args().buffer(0, s.kst).buffer(1, s.vst).buffer(2, *g.k, g.base).buffer(3, *g.v, g.base)
                                   .value(4, RingParams{M, pos0, KV, W, first}),
                        {n * (KV / 8)}, {256});
}

void encode_prefill_attention(gpu::Device& dev, gpu::CommandStream& cs, const ModelConfig& c,
                              const LayerWeights& L, uint32_t layer_index, uint32_t kv_slot,
                              const KVCache& cache, uint32_t pos0, uint32_t M,
                              const AttnScratch& rope, PrefillScratch& s) {
    check_rows(s, M);
    const bool sliding = L.sliding;
    const uint32_t H = c.hidden_size, Q = c.q_dim(), KV = c.kv_dim();
    // 1. normed = rmsnorm(x) * attn_norm
    encode_rmsnorm_f32(dev, cs, L.attn_norm, c.rms_norm_eps, s.x, 0, s.normed, 0, M);
    // 2. qkv = normed · [Wq;Wk;Wv]^T + b
    const GemmSegment qkv[3] = {{&L.wq, &L.bq}, {&L.wk, &L.bk}, {&L.wv, &L.bv}};
    encode_gemm_bf16(dev, cs, qkv, 3, s.normed, 0, H, s.qkv, 0, Q + 2 * KV, M, false);
    // 3. RoPE + K/V staging (+ full-layer cache rows)
    encode_prefill_rope_kv(dev, cs, c, sliding, kv_slot, cache, pos0, M, rope, s);
    // 4. attention
    encode_prefill_attn_core(dev, cs, c, L, sliding, kv_slot, cache, pos0, M, s);
    // 5. ring update after attention (sliding layers)
    if (sliding) encode_prefill_ring_write(dev, cs, c, kv_slot, cache, pos0, M, s);
    // 6. x += attn · Wo^T + bo
    const GemmSegment o{&L.wo, &L.bo};
    encode_gemm_bf16(dev, cs, &o, 1, s.attn, 0, Q, s.x, 0, H, M, true);
    (void)layer_index;
}

uint32_t moe_row_tile(const ModelConfig& c, uint32_t M) {
    if (const char* e = std::getenv("CORAL_PF_MOE_BM")) {
        const int v = std::atoi(e);
        if (v == 16 || v == 32 || v == 64) return uint32_t(v);
    }
    // Expected rows per expert; tuned on M2 Max (tests/test_prefill.cpp
    // prefill_zz_profile, us/token for the 24-layer MoE at M = 64/128/256/512/1024:
    //   BM 16: 1429 1073 922 859 828   BM 32: 1765 1174 913 795 741   BM 64: 2659 1605 1086 857 761).
    const uint32_t avg = M * c.experts_per_token / c.num_experts;
    return avg < kMoeBm32Rows ? 16 : avg < kMoeBm64Rows ? 32 : 64;
}

void encode_prefill_moe_route(gpu::Device& dev, gpu::CommandStream& cs, const ModelConfig& c,
                              const LayerWeights& L, uint32_t M, PrefillScratch& s) {
    check_rows(s, M);
    check_layout(c);
    const uint32_t H = c.hidden_size, E = c.num_experts, K = c.experts_per_token;
    check_bf16(L.router_w, "router", 8);
    check_bf16(L.router_b, "router bias", 2);
    // 1. normed = rmsnorm(x) * mlp_norm
    encode_rmsnorm_f32(dev, cs, L.mlp_norm, c.rms_norm_eps, s.x, 0, s.normed, 0, M);
    // 2. router + top-k
    cs.dispatch(dev.kernel("pf_router"),
                gpu::Args().buffer(0, s.normed).buffer(1, L.router_w.buf, L.router_w.offset)
                           .buffer(2, L.router_b.buf, L.router_b.offset).buffer(3, s.top_ids)
                           .buffer(4, s.top_probs).value(5, RouterParams{H, E, K}),
                {M}, {256});
    // 3. counting sort by expert -> list + tile table
    const uint32_t bm = moe_row_tile(c, M);
    cs.dispatch(dev.kernel("pf_moe_sort"),
                gpu::Args().buffer(0, s.top_ids).buffer(1, s.list).buffer(2, s.tiles)
                           .value(3, SortParams{M * K, E, K, s.max_tiles, bm}),
                {1}, {1024});
}

void encode_prefill_moe_experts(gpu::Device& dev, gpu::CommandStream& cs, const ModelConfig& c,
                                const LayerWeights& L, uint32_t M, PrefillScratch& s) {
    check_rows(s, M);
    check_layout(c);
    const uint32_t H = c.hidden_size, I = c.intermediate_size, E = c.num_experts, K = c.experts_per_token;
    for (const TensorRef* t : {&L.gate_up_blocks, &L.gate_up_scales, &L.down_blocks, &L.down_scales})
        require(t->buf.valid() && addr(t->buf, t->offset) % 8 == 0, "expert tensor '" + t->name + "' unbound or misaligned");
    check_bf16(L.gate_up_bias, "gate_up bias", 2);
    check_bf16(L.down_bias, "down bias", 2);
    const uint32_t bm = moe_row_tile(c, M);
    // Upper bound on the tiles this chunk can produce (the rest exit at once).
    const uint32_t tiles = std::min(s.max_tiles, (M * K + bm - 1) / bm + std::min(E, M * K));
    const std::string sfx = "_m" + std::to_string(bm);
    // 4. gate_up + SwiGLU -> h (list order)
    const MoeParams gu{2 * I, H, H, K, H, c.swiglu_limit, kSwigluAlpha, I};
    cs.dispatch(dev.kernel("pf_moe_gate_up" + sfx),
                gpu::Args().buffer(0, s.normed).buffer(1, L.gate_up_blocks.buf, L.gate_up_blocks.offset)
                           .buffer(2, L.gate_up_scales.buf, L.gate_up_scales.offset)
                           .buffer(3, L.gate_up_bias.buf, L.gate_up_bias.offset)
                           .buffer(4, s.list).buffer(5, s.tiles).buffer(6, s.top_probs).buffer(7, s.h).value(8, gu),
                {2 * I / kBN, tiles}, {kGemmThreads});
    // 5. down + bias, * prob -> y slots
    const MoeParams dn{H, I, I, K, H, c.swiglu_limit, kSwigluAlpha, I};
    cs.dispatch(dev.kernel("pf_moe_down" + sfx),
                gpu::Args().buffer(0, s.h).buffer(1, L.down_blocks.buf, L.down_blocks.offset)
                           .buffer(2, L.down_scales.buf, L.down_scales.offset)
                           .buffer(3, L.down_bias.buf, L.down_bias.offset)
                           .buffer(4, s.list).buffer(5, s.tiles).buffer(6, s.top_probs).buffer(7, s.y).value(8, dn),
                {H / kBN, tiles}, {kGemmThreads});
    // 6. x += sum over slots
    cs.dispatch_threads(dev.kernel("pf_moe_reduce"),
                        gpu::Args().buffer(0, s.x).buffer(1, s.y).value(2, ReduceParams{M, H, K}),
                        {M * (H / 4)}, {256});
}

void encode_prefill_moe(gpu::Device& dev, gpu::CommandStream& cs, const ModelConfig& c,
                        const LayerWeights& L, uint32_t M, PrefillScratch& s) {
    encode_prefill_moe_route(dev, cs, c, L, M, s);
    encode_prefill_moe_experts(dev, cs, c, L, M, s);
}

} // namespace coral
