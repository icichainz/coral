#include "attention_ops.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace coral {

namespace {

constexpr uint32_t kHeadDim = 64;          // kernels are specialized for head_dim 64
constexpr uint32_t kGroup = 8;             // query heads per kv head (one simdgroup each)
constexpr uint32_t kMaxSplits = 16;        // attention split-K upper bound (see attn_split)
constexpr uint32_t kGemvSimdgroups = 8;    // GEMV_SG in gemv_bf16.metal
constexpr uint32_t kQkvPairs = 2;          // ATTN_QKV_P in rope.metal
constexpr uint32_t kArgmaxPartials = 256;
constexpr uint32_t kDefaultRows = 4;       // tuned on M2 Max (see tests/test_gemv.cpp)

void require(bool ok, const std::string& what) {
    if (!ok) throw std::invalid_argument(what);
}

uint64_t addr(const gpu::Buffer& b, size_t off) { return b.gpu_address() + off; }

void check_bf16(const TensorRef& t, const char* what, size_t align = 8) {
    require(t.buf.valid(), std::string(what) + ": tensor '" + t.name + "' is not bound to the GPU");
    require(t.dtype == DType::BF16, std::string(what) + ": tensor '" + t.name + "' must be bf16");
    require(addr(t.buf, t.offset) % align == 0,
            std::string(what) + ": tensor '" + t.name + "' is not " + std::to_string(align) + "-byte aligned");
}

void check_f32(const gpu::Buffer& b, size_t off, size_t count, const char* what) {
    require(b.valid(), std::string(what) + ": buffer not allocated");
    require(addr(b, off) % 16 == 0, std::string(what) + ": fp32 buffer must be 16-byte aligned");
    require(off + count * 4 <= b.size(), std::string(what) + ": buffer too small");
}

struct GemvParams { uint32_t rows, K, flags; float eps; };
struct RmsParams { uint32_t n; float eps; };
struct ArgmaxParams { uint32_t n; };
struct RopeParams { uint32_t n_heads; };
struct QkvParams { uint32_t H, Q, KV; float eps; };
struct AttnParams { uint32_t n, chunk, kv_heads, max_splits; float scale; };

} // namespace

// ---------------------------------------------------------------------------
// YaRN (HF modeling_rope_utils._compute_yarn_parameters), fp32 where HF is fp32.
// ---------------------------------------------------------------------------
YarnParams yarn_parameters(const ModelConfig& c) {
    const uint32_t dim = c.head_dim;
    const double base = c.rope_theta, factor = c.rope_factor;
    const double orig = c.rope_original_max_pos;
    const double beta_fast = c.rope_beta_fast, beta_slow = c.rope_beta_slow;

    YarnParams y;
    y.attention_factor = factor <= 1 ? 1.0f : float(0.1 * std::log(factor) + 1.0);

    auto correction_dim = [&](double rot) {
        return (dim * std::log(orig / (rot * 2 * M_PI))) / (2 * std::log(base));
    };
    double low = correction_dim(beta_fast), high = correction_dim(beta_slow);   // truncate = false
    low = std::max(low, 0.0);
    high = std::min(high, double(dim - 1));
    if (low == high) high += 0.001;

    const uint32_t half = dim / 2;
    y.inv_freq.resize(half);
    for (uint32_t j = 0; j < half; ++j) {
        const float pos_freq = float(std::pow(base, double(float(2 * j) / float(dim))));
        const float extrap = 1.0f / pos_freq;
        const float interp = 1.0f / (float(factor) * pos_freq);
        const float ramp = std::clamp((float(j) - float(low)) / float(high - low), 0.0f, 1.0f);
        const float extrap_factor = 1.0f - ramp;
        y.inv_freq[j] = interp * (1.0f - extrap_factor) + extrap * extrap_factor;
    }
    return y;
}

gpu::Buffer make_rope_table(gpu::Device& dev, const ModelConfig& c, uint32_t max_positions) {
    const YarnParams y = yarn_parameters(c);
    const uint32_t half = c.head_dim / 2;
    gpu::Buffer t = dev.alloc(size_t(max_positions) * half * 2 * sizeof(float));
    float* p = t.as<float>();
    for (uint32_t pos = 0; pos < max_positions; ++pos)
        for (uint32_t j = 0; j < half; ++j) {
            const float f = y.inv_freq[j] * float(pos);
            p[(size_t(pos) * half + j) * 2 + 0] = float(std::cos(double(f))) * y.attention_factor;
            p[(size_t(pos) * half + j) * 2 + 1] = float(std::sin(double(f))) * y.attention_factor;
        }
    return t;
}

AttnScratch make_attn_scratch(gpu::Device& dev, const ModelConfig& c, uint32_t max_positions) {
    require(c.head_dim == kHeadDim, "attention: kernels require head_dim 64");
    require(c.num_kv_heads * kGroup == c.num_heads, "attention: kernels require 8 query heads per kv head");
    AttnScratch s;
    s.max_positions = max_positions;
    s.max_splits = kMaxSplits;
    s.q = dev.alloc(size_t(c.q_dim()) * 4, true);
    s.attn_out = dev.alloc(size_t(c.q_dim()) * 4, true);
    s.partials = dev.alloc(size_t(c.num_heads) * kMaxSplits * (kHeadDim + 2) * 4, true);
    s.rope = make_rope_table(dev, c, max_positions);
    s.counters = dev.alloc(size_t(c.num_kv_heads) * 4, true);
    return s;
}

uint32_t kv_slot_for_layer(const ModelConfig& c, uint32_t layer) {
    require(layer < c.layer_is_sliding.size(), "kv_slot_for_layer: layer out of range");
    uint32_t n = 0;
    for (uint32_t i = 0; i < layer; ++i) n += c.layer_is_sliding[i] == c.layer_is_sliding[layer];
    return n;
}

AttnSplit attn_split(uint32_t n) {
    // One 32-position block per split up to 512 positions (short dependency
    // chains, more threadgroups), then at most kMaxSplits splits.
    uint32_t chunk = (n + kMaxSplits - 1) / kMaxSplits;
    chunk = std::max<uint32_t>(32, (chunk + 31) / 32 * 32);
    return {chunk, std::max<uint32_t>(1, (n + chunk - 1) / chunk)};
}

// ---------------------------------------------------------------------------
void encode_gemv_bf16(gpu::Device& dev, gpu::CommandStream& cs, const TensorRef& W,
                      const gpu::Buffer& x, size_t x_offset, const gpu::Buffer& y, size_t y_offset,
                      const GemvOptions& opt) {
    check_bf16(W, "gemv");
    require(W.shape.size() == 2, "gemv: W must be 2-D");
    const uint32_t rows = uint32_t(W.dim(0)), K = uint32_t(W.dim(1));
    require(K % 8 == 0 && rows > 0, "gemv: K must be a multiple of 8");
    check_f32(x, x_offset, K, "gemv x");
    check_f32(y, y_offset, rows, "gemv y");
    uint32_t flags = 0;
    if (opt.bias) {
        check_bf16(*opt.bias, "gemv bias", 2);
        require(opt.bias->nbytes >= size_t(rows) * 2, "gemv: bias too small");
        flags |= 1;
    }
    if (opt.accumulate) flags |= 2;
    if (opt.norm) {
        check_bf16(*opt.norm, "gemv norm");
        require(opt.norm->nbytes >= size_t(K) * 2, "gemv: norm weight too small");
    }
    const uint32_t R = opt.rows_per_simdgroup ? opt.rows_per_simdgroup : kDefaultRows;
    require(R == 1 || R == 2 || R == 4 || R == 8, "gemv: rows_per_simdgroup must be 1, 2, 4 or 8");
    const std::string name = std::string(opt.norm ? "gemv_bf16_norm_r" : "gemv_bf16_r") + std::to_string(R);

    const GemvParams p{rows, K, flags, opt.eps};
    const TensorRef& b = opt.bias ? *opt.bias : W;   // unused binding when absent
    const TensorRef& g = opt.norm ? *opt.norm : W;
    const uint32_t per_tg = R * kGemvSimdgroups;
    cs.dispatch(dev.kernel(name),
                gpu::Args().buffer(0, W.buf, W.offset).buffer(1, x, x_offset).buffer(2, y, y_offset)
                           .buffer(3, b.buf, b.offset).buffer(4, g.buf, g.offset).value(5, p),
                {(rows + per_tg - 1) / per_tg}, {kGemvSimdgroups * 32});
}

void encode_rmsnorm_f32(gpu::Device& dev, gpu::CommandStream& cs, const TensorRef& w, float eps,
                        const gpu::Buffer& x, size_t x_offset, const gpu::Buffer& y, size_t y_offset,
                        uint32_t rows) {
    check_bf16(w, "rmsnorm_f32", 2);
    const uint32_t n = uint32_t(w.dim(0));
    require(n % 4 == 0, "rmsnorm_f32: n must be a multiple of 4");
    check_f32(x, x_offset, size_t(rows) * n, "rmsnorm_f32 x");
    check_f32(y, y_offset, size_t(rows) * n, "rmsnorm_f32 y");
    const RmsParams p{n, eps};
    cs.dispatch(dev.kernel("rmsnorm_f32"),
                gpu::Args().buffer(0, x, x_offset).buffer(1, w.buf, w.offset).buffer(2, y, y_offset).value(3, p),
                {rows}, {256});
}

size_t argmax_scratch_bytes() { return size_t(kArgmaxPartials) * 8; }

void encode_argmax_f32(gpu::Device& dev, gpu::CommandStream& cs, const gpu::Buffer& x, size_t x_offset,
                       uint32_t n, const gpu::Buffer& out, size_t out_offset, const gpu::Buffer& scratch) {
    require(n > 0 && x.valid() && x_offset + size_t(n) * 4 <= x.size() && x_offset % 4 == 0, "argmax: bad input");
    require(out.valid() && out_offset + 4 <= out.size() && out_offset % 4 == 0, "argmax: bad output");
    require(scratch.valid() && scratch.size() >= argmax_scratch_bytes(), "argmax: scratch too small");
    const uint32_t groups = std::min<uint32_t>(kArgmaxPartials, (n + 1023) / 1024);
    const size_t vi = size_t(kArgmaxPartials) * 4;   // indices after the values
    cs.dispatch(dev.kernel("argmax_f32_partial"),
                gpu::Args().buffer(0, x, x_offset).buffer(1, scratch, 0).buffer(2, scratch, vi).value(3, ArgmaxParams{n}),
                {groups}, {1024});
    cs.dispatch(dev.kernel("argmax_f32_final"),
                gpu::Args().buffer(0, scratch, 0).buffer(1, scratch, vi).buffer(2, out, out_offset)
                           .value(3, ArgmaxParams{groups}),
                {1}, {256});
}

void encode_rope_f32(gpu::Device& dev, gpu::CommandStream& cs, const AttnScratch& s, uint32_t pos,
                     const gpu::Buffer& x, size_t x_offset, uint32_t n_heads) {
    require(pos < s.max_positions, "rope: position beyond the rope table");
    check_f32(x, x_offset, size_t(n_heads) * kHeadDim, "rope x");
    cs.dispatch_threads(dev.kernel("rope_yarn_f32"),
                        gpu::Args().buffer(0, x, x_offset).buffer(1, s.rope, size_t(pos) * kHeadDim / 2 * 8)
                                   .value(2, RopeParams{n_heads}),
                        {n_heads * kHeadDim / 2}, {256});
}

// ---------------------------------------------------------------------------
namespace {
struct CacheGeom {
    const gpu::Buffer* k; const gpu::Buffer* v;
    size_t layer_base;   // byte offset of this layer's slot run
    size_t row_off;      // byte offset of the row for `pos`
    uint32_t n;          // slots to attend: [0, n)
};

CacheGeom cache_geom(const ModelConfig& c, const LayerWeights& L, uint32_t layer_index, bool sliding,
                     uint32_t kv_slot, const KVCache& cache, uint32_t pos, const AttnScratch& s,
                     const std::string& where) {
    require(c.head_dim == kHeadDim && c.num_kv_heads * kGroup == c.num_heads, where + ": unsupported head layout");
    require(L.sliding == sliding, where + ": sliding flag does not match the layer weights");
    require(c.hidden_size % 8 == 0 && c.q_dim() % 64 == 0 && c.kv_dim() % 64 == 0, where + ": bad dims");
    require(pos < s.max_positions, where + ": position beyond the rope table");
    require(s.max_splits >= kMaxSplits, where + ": scratch too small");
    const uint32_t W = c.sliding_window;
    const size_t row_bytes = size_t(c.kv_dim()) * 2;
    const uint32_t slots = sliding ? W : cache.capacity;
    CacheGeom g;
    g.k = sliding ? &cache.k_slide : &cache.k_full;
    g.v = sliding ? &cache.v_slide : &cache.v_full;
    require(g.k->valid() && g.v->valid(), where + ": KV cache not allocated");
    require(sliding || pos < cache.capacity, where + ": position beyond KV cache capacity");
    g.layer_base = size_t(kv_slot) * slots * row_bytes;
    require(g.layer_base + size_t(slots) * row_bytes <= std::min(g.k->size(), g.v->size()),
            where + ": kv_slot out of range for the cache");
    g.row_off = g.layer_base + size_t(sliding ? pos % W : pos) * row_bytes;
    g.n = sliding ? std::min(pos + 1, W) : pos + 1;
    return g;
}
} // namespace

void encode_attn_qkv(gpu::Device& dev, gpu::CommandStream& cs, const ModelConfig& c,
                     const LayerWeights& L, uint32_t layer_index, bool sliding, uint32_t kv_slot,
                     const KVCache& cache, uint32_t pos,
                     const gpu::Buffer& residual, size_t residual_offset, AttnScratch& s) {
    const std::string where = "attention layer " + std::to_string(layer_index);
    const CacheGeom g = cache_geom(c, L, layer_index, sliding, kv_slot, cache, pos, s, where);
    for (const TensorRef* t : {&L.attn_norm, &L.wq, &L.wk, &L.wv}) check_bf16(*t, "attention");
    for (const TensorRef* t : {&L.bq, &L.bk, &L.bv}) check_bf16(*t, "attention", 2);
    const uint32_t H = c.hidden_size, Q = c.q_dim(), KV = c.kv_dim();
    check_f32(residual, residual_offset, H, "attention residual");

    const uint32_t pairs = (Q + 2 * KV) / 2, per_tg = kQkvPairs * kGemvSimdgroups;
    cs.dispatch(dev.kernel("attn_qkv_rope"),
                gpu::Args().buffer(0, residual, residual_offset).buffer(1, L.attn_norm.buf, L.attn_norm.offset)
                           .buffer(2, L.wq.buf, L.wq.offset).buffer(3, L.bq.buf, L.bq.offset)
                           .buffer(4, L.wk.buf, L.wk.offset).buffer(5, L.bk.buf, L.bk.offset)
                           .buffer(6, L.wv.buf, L.wv.offset).buffer(7, L.bv.buf, L.bv.offset)
                           .buffer(8, s.q).buffer(9, *g.k, g.row_off).buffer(10, *g.v, g.row_off)
                           .buffer(11, s.rope, size_t(pos) * kHeadDim / 2 * 8)
                           .value(12, QkvParams{H, Q, KV, c.rms_norm_eps}),
                {(pairs + per_tg - 1) / per_tg}, {kGemvSimdgroups * 32});
}

void encode_attn_core(gpu::Device& dev, gpu::CommandStream& cs, const ModelConfig& c,
                      const LayerWeights& L, uint32_t layer_index, bool sliding, uint32_t kv_slot,
                      const KVCache& cache, uint32_t pos, AttnScratch& s) {
    const std::string where = "attention layer " + std::to_string(layer_index);
    const CacheGeom g = cache_geom(c, L, layer_index, sliding, kv_slot, cache, pos, s, where);
    check_bf16(L.sinks, "attention", 2);
    const AttnSplit sp = attn_split(g.n);
    const AttnParams ap{g.n, sp.chunk, c.num_kv_heads, s.max_splits, 1.0f / std::sqrt(float(kHeadDim))};
    require(s.counters.valid() && s.counters.size() >= size_t(c.num_kv_heads) * 4, where + ": scratch counters missing");
    cs.dispatch(dev.kernel("attn_decode_fused"),
                gpu::Args().buffer(0, s.q).buffer(1, *g.k, g.layer_base).buffer(2, *g.v, g.layer_base)
                           .buffer(3, s.partials).buffer(4, L.sinks.buf, L.sinks.offset).buffer(5, s.attn_out)
                           .buffer(6, s.counters).value(7, ap),
                {c.num_kv_heads, sp.splits}, {kGroup * 32});
}

void encode_attention_decode(gpu::Device& dev, gpu::CommandStream& cs, const ModelConfig& c,
                             const LayerWeights& L, uint32_t layer_index, bool sliding, uint32_t kv_slot,
                             const KVCache& cache, uint32_t pos,
                             const gpu::Buffer& residual, size_t residual_offset, AttnScratch& s) {
    // 1. rmsnorm + QKV + bias + RoPE + KV write.
    encode_attn_qkv(dev, cs, c, L, layer_index, sliding, kv_slot, cache, pos, residual, residual_offset, s);
    // 2. attention partials over the cache slots, merge + sink -> s.attn_out.
    encode_attn_core(dev, cs, c, L, layer_index, sliding, kv_slot, cache, pos, s);
    // 3. residual += Wo . attn + bo.
    check_bf16(L.wo, "attention");
    GemvOptions o;
    o.bias = &L.bo;
    o.accumulate = true;
    encode_gemv_bf16(dev, cs, L.wo, s.attn_out, 0, residual, residual_offset, o);
}

} // namespace coral
