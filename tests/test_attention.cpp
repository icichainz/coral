// Decode attention (RoPE/YaRN, fused QKV, sinks, sliding ring, o_proj +
// residual) against CPU references built on the real layer weights.
#include "test.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "coral/gpu.h"
#include "coral/model.h"
#include "coral/safetensors.h"
#include "../src/model/attention_ops.h"
#include "../src/model/weights.h"

using namespace coral;

// Shared with test_gemv.cpp.
gpu::Device& attn_test_device();
Model& attn_test_model();

namespace {

float round_bf16(float x) { return bf16_to_f32(f32_to_bf16(x)); }
float bfv(const TensorRef& t, size_t i) { return bf16_to_f32(t.as<uint16_t>()[i]); }

// ---- CPU YaRN, ported from HF modeling_rope_utils._compute_yarn_parameters ----
struct CpuRope {
    std::vector<float> inv_freq;
    float attention_factor;
    explicit CpuRope(const ModelConfig& c) {
        const int dim = int(c.head_dim);
        const double base = c.rope_theta, factor = c.rope_factor, orig = c.rope_original_max_pos;
        attention_factor = float(0.1 * std::log(factor) + 1.0);          // get_mscale(factor)
        auto find_correction_dim = [&](double num_rot) {
            return (dim * std::log(orig / (num_rot * 2 * M_PI))) / (2 * std::log(base));
        };
        double low = find_correction_dim(c.rope_beta_fast), high = find_correction_dim(c.rope_beta_slow);
        low = std::max(low, 0.0); high = std::min(high, double(dim - 1));   // truncate = False
        for (int j = 0; j < dim / 2; ++j) {
            float pos_freqs = float(std::pow(base, double(float(2 * j) / float(dim))));
            float extra = 1.0f / pos_freqs, inter = 1.0f / (float(factor) * pos_freqs);
            float ramp = std::min(1.0f, std::max(0.0f, (float(j) - float(low)) / float(high - low)));
            float ef = 1.0f - ramp;                                          // inv_freq_extrapolation_factor
            inv_freq.push_back(inter * (1 - ef) + extra * ef);
        }
    }
    float cos_(uint32_t pos, int j) const { return float(std::cos(double(inv_freq[j] * float(pos)))) * attention_factor; }
    float sin_(uint32_t pos, int j) const { return float(std::sin(double(inv_freq[j] * float(pos)))) * attention_factor; }
    // _apply_rotary_emb: first/second half pairing.
    void apply(float* x, uint32_t n_heads, uint32_t pos) const {
        for (uint32_t h = 0; h < n_heads; ++h) {
            float* v = x + h * 64;
            for (int j = 0; j < 32; ++j) {
                const float c = cos_(pos, j), s = sin_(pos, j), a = v[j], b = v[j + 32];
                v[j] = a * c - b * s;
                v[j + 32] = b * c + a * s;
            }
        }
    }
};

// ---- CPU reference for one attention layer ----
struct CpuAttn {
    const ModelConfig& c;
    const LayerWeights& L;
    const CpuRope& rope;
    std::vector<std::vector<float>> K, V;   // per absolute position, bf16-rounded [KV]

    static std::vector<float> matvec(const TensorRef& W, const TensorRef& b, const std::vector<double>& x) {
        const size_t rows = size_t(W.dim(0)), k = size_t(W.dim(1));
        std::vector<float> y(rows);
        for (size_t r = 0; r < rows; ++r) {
            const uint16_t* w = W.as<uint16_t>() + r * k;
            double acc = 0;
            for (size_t i = 0; i < k; ++i) acc += double(bf16_to_f32(w[i])) * x[i];
            y[r] = float(acc + bfv(b, r));
        }
        return y;
    }
    std::vector<double> norm(const float* x) const {
        const uint32_t H = c.hidden_size;
        double ss = 0; for (uint32_t i = 0; i < H; ++i) ss += double(x[i]) * x[i];
        const double inv = 1.0 / std::sqrt(ss / H + c.rms_norm_eps);
        std::vector<double> xn(H);
        for (uint32_t i = 0; i < H; ++i) xn[i] = x[i] * inv * bfv(L.attn_norm, i);
        return xn;
    }
    // K/V for `pos` (always needed), returns normalized input for q.
    std::vector<double> push_kv(const float* x, uint32_t pos) {
        auto xn = norm(x);
        auto k = matvec(L.wk, L.bk, xn), v = matvec(L.wv, L.bv, xn);
        rope.apply(k.data(), c.num_kv_heads, pos);
        for (auto& e : k) e = round_bf16(e);
        for (auto& e : v) e = round_bf16(e);
        if (K.size() <= pos) { K.resize(pos + 1); V.resize(pos + 1); }
        K[pos] = std::move(k); V[pos] = std::move(v);
        return xn;
    }
    // Attention output [Q] over positions [start, pos] (HF eager_attention_forward with sinks).
    std::vector<float> attend(const std::vector<float>& q, uint32_t start, uint32_t pos) const {
        const uint32_t heads = c.num_heads, group = heads / c.num_kv_heads;
        std::vector<float> out(size_t(heads) * 64);
        for (uint32_t h = 0; h < heads; ++h) {
            const uint32_t kvh = h / group;
            std::vector<double> s;
            for (uint32_t j = start; j <= pos; ++j) {
                double d = 0;
                for (int e = 0; e < 64; ++e) d += double(q[h * 64 + e]) * K[j][kvh * 64 + e];
                s.push_back(d / 8.0);
            }
            const double sink = bfv(L.sinks, h);
            double m = sink; for (double x : s) m = std::max(m, x);
            double sum = std::exp(sink - m); for (double& x : s) { x = std::exp(x - m); sum += x; }
            for (int e = 0; e < 64; ++e) {
                double o = 0;
                for (uint32_t j = start; j <= pos; ++j) o += s[j - start] * V[j][kvh * 64 + e];
                out[h * 64 + e] = float(o / sum);
            }
        }
        return out;
    }
    std::vector<float> q_of(const std::vector<double>& xn, uint32_t pos) const {
        auto q = matvec(L.wq, L.bq, xn);
        rope.apply(q.data(), c.num_heads, pos);
        return q;
    }
    std::vector<float> o_proj(const std::vector<float>& a) const {
        std::vector<double> ad(a.begin(), a.end());
        return matvec(L.wo, L.bo, ad);
    }
};

double max_abs_diff(const float* a, const float* b, size_t n) {
    double m = 0; for (size_t i = 0; i < n; ++i) m = std::max(m, double(std::fabs(a[i] - b[i]))); return m;
}
double max_abs(const float* a, size_t n) {
    double m = 0; for (size_t i = 0; i < n; ++i) m = std::max(m, double(std::fabs(a[i]))); return m;
}

} // namespace

CORAL_TEST(attn_yarn_rope_matches_hf) {
    auto& dev = attn_test_device();
    ModelConfig c;   // gpt-oss defaults (theta 150000, factor 32, 4096, beta 32/1)
    c.head_dim = 64;
    const CpuRope ref(c);
    const YarnParams y = yarn_parameters(c);
    CHECK_NEAR(y.attention_factor, 0.1 * std::log(32.0) + 1.0, 1e-7);
    CHECK_EQ(y.inv_freq.size(), size_t(32));
    CHECK_EQ(y.inv_freq[0], 1.0f);                                  // pure extrapolation band
    CHECK_NEAR(y.inv_freq[31], 1.0 / (32.0 * std::pow(150000.0, 62.0 / 64)), 1e-12);   // pure interpolation band
    for (int j = 0; j < 32; ++j) CHECK_NEAR(y.inv_freq[j], ref.inv_freq[j], 1e-7 * ref.inv_freq[j]);

    const uint32_t positions[] = {0, 1, 127, 4095, 4096, 30000};
    AttnScratch s = make_attn_scratch(dev, c, 30001);
    const float* tab = s.rope.as<float>();
    for (uint32_t pos : positions)
        for (int j = 0; j < 32; ++j) {
            CHECK_NEAR(tab[(size_t(pos) * 32 + j) * 2], ref.cos_(pos, j), 1e-6);
            CHECK_NEAR(tab[(size_t(pos) * 32 + j) * 2 + 1], ref.sin_(pos, j), 1e-6);
        }
    CHECK_NEAR(tab[0], y.attention_factor, 1e-7);   // cos(0) * mscale
    CHECK_EQ(tab[1], 0.0f);

    // Rotate q (64 heads) and k (8 heads) on the GPU and compare.
    std::mt19937 rng(9);
    std::normal_distribution<float> nd(0.f, 2.f);
    for (uint32_t pos : positions)
        for (uint32_t heads : {64u, 8u}) {
            const uint32_t n = heads * 64;
            gpu::Buffer x = dev.alloc(n * 4);
            std::vector<float> want(n);
            for (uint32_t i = 0; i < n; ++i) want[i] = x.as<float>()[i] = nd(rng);
            ref.apply(want.data(), heads, pos);
            auto cs = dev.stream();
            cs.begin();
            encode_rope_f32(dev, cs, s, pos, x, 0, heads);
            cs.submit_and_wait();
            for (uint32_t i = 0; i < n; ++i) CHECK_NEAR(x.as<float>()[i], want[i], 1e-5 * (1 + std::fabs(want[i])));
        }
}

CORAL_TEST(attn_kv_slot_and_split) {
    ModelConfig c;
    c.num_layers = 6;
    c.layer_is_sliding = {true, false, true, false, true, false};
    CHECK_EQ(kv_slot_for_layer(c, 0), 0u);
    CHECK_EQ(kv_slot_for_layer(c, 1), 0u);
    CHECK_EQ(kv_slot_for_layer(c, 4), 2u);
    CHECK_EQ(kv_slot_for_layer(c, 5), 2u);
    for (uint32_t n : {1u, 64u, 65u, 128u, 1000u, 5001u, 131072u}) {
        AttnSplit s = attn_split(n);
        CHECK(s.chunk % 32 == 0);
        CHECK(s.splits >= 1 && s.splits <= 16);
        CHECK(s.chunk * s.splits >= n && s.chunk * (s.splits - 1) < n);
    }
}

// Positions 0..200 through one real layer; compare with the CPU reference at
// several positions. Layer 0 is sliding (window 128, ring buffer), layer 1 full.
CORAL_TEST(attn_decode_layer_matches_cpu) {
    Model& m = attn_test_model();
    auto& dev = attn_test_device();
    const ModelConfig& c = m.config();
    const Weights& w = model_weights(m);
    const CpuRope rope(c);
    const uint32_t H = c.hidden_size, Q = c.q_dim(), KV = c.kv_dim();
    const uint32_t checks[] = {0, 1, 5, 64, 126, 127, 128, 129, 150, 200};

    for (uint32_t layer : {0u, 1u}) {
        const LayerWeights& L = w.layers[layer];
        const bool sliding = L.sliding;
        CHECK_EQ(sliding, layer == 0);
        const uint32_t slot = kv_slot_for_layer(c, layer);
        KVCache cache = m.new_cache(512);
        AttnScratch s = make_attn_scratch(dev, c, 512);
        gpu::Buffer residual = dev.alloc(H * 4);
        CpuAttn ref{c, L, rope, {}, {}};
        std::mt19937 rng(100 + layer);
        std::normal_distribution<float> nd(0.f, 1.f);
        double worst_attn = 0, worst_res = 0, worst_kv = 0;

        for (uint32_t pos = 0; pos <= 200; ++pos) {
            std::vector<float> x(H);
            for (auto& e : x) e = nd(rng) * 2.0f;
            std::memcpy(residual.data(), x.data(), H * 4);
            auto cs = dev.stream();
            cs.begin();
            encode_attention_decode(dev, cs, c, L, layer, sliding, slot, cache, pos, residual, 0, s);
            CHECK_EQ(cs.dispatch_count(), size_t(3));
            cs.submit_and_wait();

            auto xn = ref.push_kv(x.data(), pos);
            // The cache row the GPU wrote for this position.
            const size_t row = sliding ? (size_t(slot) * c.sliding_window + pos % c.sliding_window)
                                       : (size_t(slot) * cache.capacity + pos);
            const uint16_t* gk = (sliding ? cache.k_slide : cache.k_full).as<uint16_t>() + row * KV;
            const uint16_t* gv = (sliding ? cache.v_slide : cache.v_full).as<uint16_t>() + row * KV;
            for (uint32_t i = 0; i < KV; ++i) {
                const double dk = std::fabs(bf16_to_f32(gk[i]) - ref.K[pos][i]);
                const double dv = std::fabs(bf16_to_f32(gv[i]) - ref.V[pos][i]);
                worst_kv = std::max({worst_kv, dk / (1e-2 * std::fabs(ref.K[pos][i]) + 1e-3),
                                     dv / (1e-2 * std::fabs(ref.V[pos][i]) + 1e-3)});
            }

            if (std::find(std::begin(checks), std::end(checks), pos) == std::end(checks)) continue;
            const uint32_t start = sliding && pos >= c.sliding_window ? pos - (c.sliding_window - 1) : 0;
            const auto q = ref.q_of(xn, pos);
            // Roped q.
            const double dq = max_abs_diff(s.q.as<float>(), q.data(), Q);
            CHECK(dq <= 1e-3 * (1 + max_abs(q.data(), Q)));
            const auto a = ref.attend(q, start, pos);
            const auto o = ref.o_proj(a);
            std::vector<float> want(H);
            for (uint32_t i = 0; i < H; ++i) want[i] = x[i] + o[i];
            const double ea = max_abs_diff(s.attn_out.as<float>(), a.data(), Q);
            const double er = max_abs_diff(residual.as<float>(), want.data(), H);
            worst_attn = std::max(worst_attn, ea);
            worst_res = std::max(worst_res, er);
            if (ea > 2e-3 * max_abs(a.data(), Q) + 1e-2 || er > 2e-2)
                std::printf("        layer %u pos %u: attn err %.3g (|a|max %.3g), residual err %.3g (|o|max %.3g)\n",
                            layer, pos, ea, max_abs(a.data(), Q), er, max_abs(o.data(), H));
            CHECK(ea <= 2e-3 * max_abs(a.data(), Q) + 1e-2);   // bf16 V rounding scales with |v|
            CHECK(er <= 2e-2);

            // The sliding window must really exclude old positions: the GPU is
            // much closer to the window-128 reference than to full causal.
            if (sliding && pos == 200) {
                const auto a_full = ref.attend(q, 0, pos);
                const double e_wrong = max_abs_diff(s.attn_out.as<float>(), a_full.data(), Q);
                std::printf("        sliding pos 200: err vs window %.3g, vs full-causal %.3g\n", ea, e_wrong);
                CHECK(e_wrong > 4 * ea);
            }
        }
        std::printf("        layer %u (%s): max attn err %.3g, max residual err %.3g, kv err/tol %.3g\n", layer,
                    sliding ? "sliding" : "full", worst_attn, worst_res, worst_kv);
        CHECK(worst_kv <= 1.0);
    }
}

// Long context: prefill the cache with random K/V, then decode one position
// with many splits (full layer) and on a wrapped ring (sliding layer).
CORAL_TEST(attn_decode_long_context_matches_cpu) {
    Model& m = attn_test_model();
    auto& dev = attn_test_device();
    const ModelConfig& c = m.config();
    const Weights& w = model_weights(m);
    const uint32_t H = c.hidden_size, Q = c.q_dim(), KV = c.kv_dim(), Wn = c.sliding_window;
    KVCache cache = m.new_cache(8192);
    AttnScratch s = make_attn_scratch(dev, c, 8192);
    gpu::Buffer residual = dev.alloc(H * 4);
    std::mt19937 rng(77);
    std::normal_distribution<float> nd(0.f, 1.f);

    struct Case { uint32_t layer, pos; };
    for (Case cs_ : {Case{3, 5000}, Case{3, 8191}, Case{2, 1000}, Case{2, 130}}) {
        const LayerWeights& L = w.layers[cs_.layer];
        const bool sliding = L.sliding;
        const uint32_t slot = kv_slot_for_layer(c, cs_.layer), pos = cs_.pos;
        const uint32_t slots = sliding ? Wn : cache.capacity;
        const size_t base = size_t(slot) * slots * KV;
        uint16_t* kc = (sliding ? cache.k_slide : cache.k_full).as<uint16_t>() + base;
        uint16_t* vc = (sliding ? cache.v_slide : cache.v_full).as<uint16_t>() + base;
        // Fill every slot (the current one is overwritten by the kernel).
        for (size_t i = 0; i < size_t(slots) * KV; ++i) { kc[i] = f32_to_bf16(nd(rng) * 3.f); vc[i] = f32_to_bf16(nd(rng)); }

        std::vector<float> x(H);
        for (auto& e : x) e = nd(rng) * 2.0f;
        std::memcpy(residual.data(), x.data(), H * 4);
        auto cs = dev.stream();
        cs.begin();
        encode_attention_decode(dev, cs, c, L, cs_.layer, sliding, slot, cache, pos, residual, 0, s);
        cs.submit_and_wait();

        // CPU attention straight from the (GPU-written) cache contents and the GPU q.
        const uint32_t n = sliding ? std::min(pos + 1, Wn) : pos + 1;
        const CpuRope rope(c);
        CpuAttn ref{c, L, rope, {}, {}};
        ref.K.resize(n); ref.V.resize(n);
        for (uint32_t j = 0; j < n; ++j) {
            ref.K[j].resize(KV); ref.V[j].resize(KV);
            for (uint32_t i = 0; i < KV; ++i) { ref.K[j][i] = bf16_to_f32(kc[size_t(j) * KV + i]); ref.V[j][i] = bf16_to_f32(vc[size_t(j) * KV + i]); }
        }
        std::vector<float> q(s.q.as<float>(), s.q.as<float>() + Q);
        const auto a = ref.attend(q, 0, n - 1);
        const auto o = ref.o_proj(a);
        std::vector<float> want(H);
        for (uint32_t i = 0; i < H; ++i) want[i] = x[i] + o[i];
        const double ea = max_abs_diff(s.attn_out.as<float>(), a.data(), Q);
        const double er = max_abs_diff(residual.as<float>(), want.data(), H);
        std::printf("        layer %u pos %u (%u splits): attn err %.3g, residual err %.3g\n", cs_.layer, pos,
                    attn_split(n).splits, ea, er);
        CHECK(ea <= 1e-4 * (1 + max_abs(a.data(), Q)));
        CHECK(er <= 1e-3 * (1 + max_abs(want.data(), H)));
    }
}

// Timing: fused QKV, o_proj and the whole attention block on real weights.
CORAL_TEST(attn_decode_bandwidth) {
    Model& m = attn_test_model();
    auto& dev = attn_test_device();
    const ModelConfig& c = m.config();
    const Weights& w = model_weights(m);
    const LayerWeights& L = w.layers[1];
    KVCache cache = m.new_cache(4096);
    AttnScratch s = make_attn_scratch(dev, c, 4096);
    gpu::Buffer residual = dev.alloc(c.hidden_size * 4, true);
    for (uint32_t i = 0; i < c.hidden_size; ++i) residual.as<float>()[i] = 0.01f * float(i % 13);
    const int iters = 20;
    auto time = [&](auto&& enc) {
        auto cs = dev.stream();
        cs.begin(); enc(cs); cs.submit_and_wait();   // warm-up
        cs.begin();
        for (int i = 0; i < iters; ++i) enc(cs);
        return cs.submit_and_wait() / iters;
    };
    const double qkv_bytes = double(L.wq.nbytes + L.wk.nbytes + L.wv.nbytes);
    const double t_qkv = time([&](gpu::CommandStream& cs) { encode_attn_qkv(dev, cs, c, L, 1, false, 0, cache, 100, residual, 0, s); });
    std::printf("        fused norm+QKV+rope+kv-write: %.1f us  %.1f GB/s\n", t_qkv * 1e6, qkv_bytes / t_qkv * 1e-9);
    GemvOptions o; o.bias = &L.bo; o.accumulate = true;
    const double t_o = time([&](gpu::CommandStream& cs) { encode_gemv_bf16(dev, cs, L.wo, s.attn_out, 0, residual, 0, o); });
    std::printf("        o_proj + residual: %.1f us  %.1f GB/s\n", t_o * 1e6, double(L.wo.nbytes) / t_o * 1e-9);
    for (uint32_t pos : {100u, 4095u}) {
        const double t = time([&](gpu::CommandStream& cs) { encode_attention_decode(dev, cs, c, L, 1, false, 0, cache, pos, residual, 0, s); });
        const double t_core = time([&](gpu::CommandStream& cs) { encode_attn_core(dev, cs, c, L, 1, false, 0, cache, pos, s); });
        const double kv_bytes = double(pos + 1) * c.kv_dim() * 2 * 2;
        std::printf("        full layer attention block, ctx %u: %.1f us (attention core %.1f us, %.1f GB/s KV)\n",
                    pos + 1, t * 1e6, t_core * 1e6, kv_bytes / t_core * 1e-9);
    }
}

CORAL_TEST(attn_zz_profile) {
    Model& m = attn_test_model();
    auto& dev = attn_test_device();
    const ModelConfig& c = m.config();
    const Weights& w = model_weights(m);
    const LayerWeights& L = w.layers[1];
    KVCache cache = m.new_cache(4096);
    AttnScratch s = make_attn_scratch(dev, c, 4096);
    gpu::Buffer residual = dev.alloc(c.hidden_size * 4, true);
    gpu::Buffer y = dev.alloc(c.hidden_size * 4, true);
    auto time = [&](int iters, auto&& enc) {
        auto cs = dev.stream();
        cs.begin(); enc(cs); cs.submit_and_wait();
        cs.begin();
        for (int i = 0; i < iters; ++i) enc(cs);
        return cs.submit_and_wait() / iters;
    };
    for (int it : {1, 20, 100})
      std::printf("        rmsnorm x%d: %.1f us\n", it, 1e6 * time(it, [&](gpu::CommandStream& cs) { encode_rmsnorm_f32(dev, cs, L.attn_norm, 1e-5f, residual, 0, y, 0); }));
    for (uint32_t R : {1u, 2u, 4u, 8u}) {
      GemvOptions o; o.bias = &L.bo; o.accumulate = true; o.rows_per_simdgroup = R;
      double t = time(100, [&](gpu::CommandStream& cs) { encode_gemv_bf16(dev, cs, L.wo, s.attn_out, 0, residual, 0, o); });
      std::printf("        o_proj R=%u: %.1f us %.1f GB/s\n", R, t * 1e6, L.wo.nbytes / t * 1e-9);
      GemvOptions o2; o2.rows_per_simdgroup = R; o2.norm = &L.attn_norm;
      t = time(100, [&](gpu::CommandStream& cs) { encode_gemv_bf16(dev, cs, L.wq, residual, 0, s.q, 0, o2); });
      std::printf("        wq R=%u: %.1f us %.1f GB/s\n", R, t * 1e6, L.wq.nbytes / t * 1e-9);
    }
    for (uint32_t pos : {100u, 1000u, 4095u}) {
      double t = time(100, [&](gpu::CommandStream& cs) { encode_attn_core(dev, cs, c, L, 1, false, 0, cache, pos, s); });
      std::printf("        core pos %u: %.1f us\n", pos, t * 1e6);
    }
}

// Per-stage timing over all 24 layers in one submit (no cache reuse between
// consecutive dispatches, as in the real forward). Set CORAL_ATTN_BENCH=1.
CORAL_TEST(attn_bench_stages) {
    if (!std::getenv("CORAL_ATTN_BENCH")) SKIP("set CORAL_ATTN_BENCH=1");
    Model& m = attn_test_model();
    auto& dev = attn_test_device();
    const ModelConfig& c = m.config();
    const Weights& w = model_weights(m);
    const uint32_t pos = std::getenv("CORAL_ATTN_POS") ? uint32_t(std::atoi(std::getenv("CORAL_ATTN_POS"))) : 300;
    KVCache cache = m.new_cache(pos + 16);
    AttnScratch s = make_attn_scratch(dev, c, pos + 16);
    gpu::Buffer residual = dev.alloc(c.hidden_size * 4, true);
    for (uint32_t i = 0; i < c.hidden_size; ++i) residual.as<float>()[i] = 0.01f * float(i % 13);
    auto cs = dev.stream();
    auto time = [&](auto&& enc) {
        double best = 1e30;
        for (int rep = 0; rep < 10; ++rep) {
            cs.begin();
            for (uint32_t l = 0; l < c.num_layers; ++l) enc(l);
            const auto t0 = std::chrono::steady_clock::now();
            cs.submit_and_wait();
            best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        return best / c.num_layers;
    };
    const auto& L0 = w.layers[0];
    const double qkv_b = double(L0.wq.nbytes + L0.wk.nbytes + L0.wv.nbytes), o_b = double(L0.wo.nbytes);
    const double tq = time([&](uint32_t l) { const auto& L = w.layers[l]; encode_attn_qkv(dev, cs, c, L, l, L.sliding, kv_slot_for_layer(c, l), cache, pos, residual, 0, s); });
    const double tc = time([&](uint32_t l) { const auto& L = w.layers[l]; encode_attn_core(dev, cs, c, L, l, L.sliding, kv_slot_for_layer(c, l), cache, pos, s); });
    GemvOptions o; o.bias = nullptr; o.accumulate = true;
    const double to = time([&](uint32_t l) { const auto& L = w.layers[l]; GemvOptions oo; oo.bias = &L.bo; oo.accumulate = true;
                                             encode_gemv_bf16(dev, cs, L.wo, s.attn_out, 0, residual, 0, oo); });
    const double ta = time([&](uint32_t l) { const auto& L = w.layers[l]; encode_attention_decode(dev, cs, c, L, l, L.sliding, kv_slot_for_layer(c, l), cache, pos, residual, 0, s); });
    std::printf("        qkv %.1f us %.0f GB/s | core %.1f us | o_proj %.1f us %.0f GB/s | block %.1f us (ctx %u)\n",
                tq * 1e6, qkv_b / tq / 1e9, tc * 1e6, to * 1e6, o_b / to / 1e9, ta * 1e6, pos + 1);
    for (uint32_t R : {1u, 2u, 4u, 8u}) {
        const double t = time([&](uint32_t l) { const auto& L = w.layers[l]; GemvOptions oo; oo.bias = &L.bo; oo.accumulate = true; oo.rows_per_simdgroup = R;
                                                encode_gemv_bf16(dev, cs, L.wo, s.attn_out, 0, residual, 0, oo); });
        const double t2 = time([&](uint32_t l) { const auto& L = w.layers[l]; GemvOptions oo; oo.norm = &L.attn_norm; oo.rows_per_simdgroup = R;
                                                encode_gemv_bf16(dev, cs, L.wq, residual, 0, s.q, 0, oo); });
        const double t3 = time([&](uint32_t l) { const auto& L = w.layers[l]; GemvOptions oo; oo.rows_per_simdgroup = R;
                                                encode_gemv_bf16(dev, cs, L.wq, residual, 0, s.q, 0, oo); });
        std::printf("        R=%u: o_proj %.1f us %.0f GB/s | wq+norm %.1f us %.0f GB/s | wq plain %.1f us %.0f GB/s\n", R,
                    t * 1e6, o_b / t / 1e9, t2 * 1e6, double(L0.wq.nbytes) / t2 / 1e9, t3 * 1e6, double(L0.wq.nbytes) / t3 / 1e9);
    }
}
