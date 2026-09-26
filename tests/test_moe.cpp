// MoE decode kernels (router top-k, MXFP4 gate_up+SwiGLU, MXFP4 down + weighted
// residual add) against fp64 CPU references built from the real gpt-oss
// weights, dequantized here independently of the kernels.
#include "test.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "coral/gpu.h"
#include "coral/kernels.h"
#include "coral/model.h"
#include "coral/safetensors.h"
#include "../src/model/moe_ops.h"
#include "../src/model/weights.h"

using namespace coral;

namespace {

struct Fixture {
    std::unique_ptr<gpu::Device> dev;
    std::unique_ptr<Model> model;
};

// Loaded once per test binary; intentionally leaked (see test_model.cpp).
Fixture& fixture() {
    std::string dir = test::model_dir_or_skip();
    static Fixture* f = [&] {
        auto* x = new Fixture;
        x->dev = gpu::Device::create();
        load_kernels(*x->dev);
        x->model = Model::load(*x->dev, dir);
        return x;
    }();
    return *f;
}

// ---- CPU reference ----------------------------------------------------------
double fp4(int n) {
    static const double lut[16] = {0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6};
    return lut[n & 15];
}
double bf(const TensorRef& t, size_t i) { return bf16_to_f32(t.as<uint16_t>()[i]); }

// Dequantize row `r` (global row index = e*rows + row) of an MXFP4 tensor with K columns.
void dequant_row(const TensorRef& blocks, const TensorRef& scales, size_t grow, size_t K, std::vector<double>& out) {
    const size_t nblk = K / 32;
    out.resize(K);
    const uint8_t* b = blocks.cpu + grow * nblk * 16;
    const uint8_t* s = scales.cpu + grow * nblk;
    for (size_t j = 0; j < nblk; ++j) {
        const double sc = std::ldexp(1.0, int(s[j]) - 127);
        for (size_t i = 0; i < 16; ++i) {
            out[j * 32 + 2 * i] = fp4(b[j * 16 + i] & 0xF) * sc;       // low nibble = even element
            out[j * 32 + 2 * i + 1] = fp4(b[j * 16 + i] >> 4) * sc;    // high nibble = odd element
        }
    }
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0; for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i]; return s;
}

std::vector<double> ref_rmsnorm(const std::vector<float>& x, const TensorRef& w, double eps) {
    double ss = 0; for (float v : x) ss += double(v) * v;
    const double inv = 1.0 / std::sqrt(ss / x.size() + eps);
    std::vector<double> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] * inv * bf(w, i);
    return y;
}

struct Routing { std::vector<int> ids; std::vector<double> probs; double gap = 0; };

// logits = W·x + b; top-k by (value desc, index asc); softmax over the k values.
Routing ref_router(const std::vector<double>& x, const TensorRef& W, const TensorRef& b, uint32_t E, uint32_t K) {
    const size_t H = x.size();
    std::vector<double> lg(E);
    for (uint32_t e = 0; e < E; ++e) {
        double s = 0; for (size_t i = 0; i < H; ++i) s += bf(W, e * H + i) * x[i];
        lg[e] = s + bf(b, e);
    }
    std::vector<int> idx(E); std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int c) { return lg[a] > lg[c]; });
    Routing r;
    r.ids.assign(idx.begin(), idx.begin() + K);
    double den = 0; for (uint32_t k = 0; k < K; ++k) den += std::exp(lg[idx[k]] - lg[idx[0]]);
    for (uint32_t k = 0; k < K; ++k) r.probs.push_back(std::exp(lg[idx[k]] - lg[idx[0]]) / den);
    r.gap = lg[idx[K - 1]] - lg[idx[K]];
    return r;
}

// h = swiglu(W_gu[e]·x + b_gu[e]) with gate = rows 0::2, up = rows 1::2.
std::vector<double> ref_gate_up(const LayerWeights& L, const ModelConfig& c, int e, const std::vector<double>& x) {
    const size_t H = c.hidden_size, I = c.intermediate_size;
    const double limit = c.swiglu_limit, alpha = 1.702;
    std::vector<double> h(I), wg, wu;
    for (size_t i = 0; i < I; ++i) {
        dequant_row(L.gate_up_blocks, L.gate_up_scales, size_t(e) * 2 * I + 2 * i, H, wg);
        dequant_row(L.gate_up_blocks, L.gate_up_scales, size_t(e) * 2 * I + 2 * i + 1, H, wu);
        double g = dot(wg, x) + bf(L.gate_up_bias, size_t(e) * 2 * I + 2 * i);
        double u = dot(wu, x) + bf(L.gate_up_bias, size_t(e) * 2 * I + 2 * i + 1);
        g = std::min(g, limit);
        u = std::clamp(u, -limit, limit);
        h[i] = (u + 1) * (g / (1 + std::exp(-alpha * g)));
    }
    return h;
}

std::vector<double> ref_down(const LayerWeights& L, const ModelConfig& c, int e, const std::vector<double>& h) {
    const size_t H = c.hidden_size, I = c.intermediate_size;
    std::vector<double> d(H), w;
    for (size_t o = 0; o < H; ++o) {
        dequant_row(L.down_blocks, L.down_scales, size_t(e) * H + o, I, w);
        d[o] = dot(w, h) + bf(L.down_bias, size_t(e) * H + o);
    }
    return d;
}

std::vector<float> random_vec(size_t n, uint32_t seed, float scale) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng);
    return v;
}

// Wall time of submit + wait, measured on the host from before the commit
// (CommandStream::wait() starts its clock after the commit returns, which
// under CPU contention can under-report).
double timed_submit(gpu::CommandStream& cs) {
    const auto t0 = std::chrono::steady_clock::now();
    cs.submit_and_wait();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

gpu::Buffer upload(gpu::Device& dev, const void* p, size_t bytes) {
    auto b = dev.alloc(bytes);
    std::memcpy(b.data(), p, bytes);
    return b;
}

} // namespace

CORAL_TEST(moe_router_topk_matches_cpu) {
    auto& f = fixture();
    auto& dev = *f.dev;
    const ModelConfig& c = f.model->config();
    const Weights& w = model_weights(*f.model);
    const uint32_t H = c.hidden_size, K = c.experts_per_token;
    MoeScratch s = make_moe_scratch(dev, c);

    double max_perr = 0, min_gap = 1e30;
    for (uint32_t layer : {0u, 1u, 23u}) {
        const LayerWeights& L = w.layers[layer];
        for (uint32_t seed = 1; seed <= 4; ++seed) {
            auto x = random_vec(H, seed * 97 + layer, 1.0f + seed);
            auto xb = upload(dev, x.data(), H * 4);
            auto cs = dev.stream();
            cs.begin();
            encode_moe_router(dev, cs, c, L.mlp_norm, L.router_w, L.router_b, xb, s, true);
            encode_moe_gate_up(dev, cs, c, L, s);
            cs.submit_and_wait();

            auto xn = ref_rmsnorm(x, L.mlp_norm, c.rms_norm_eps);
            for (uint32_t i = 0; i < H; ++i) CHECK_NEAR(s.normed.as<float>()[i], xn[i], 1e-5 * (1 + std::fabs(xn[i])));
            Routing r = ref_router(xn, L.router_w, L.router_b, c.num_experts, K);
            min_gap = std::min(min_gap, r.gap);
            for (uint32_t k = 0; k < K; ++k) {
                CHECK_EQ(s.expert_ids.as<int32_t>()[k], r.ids[k]);
                const double pe = std::fabs(s.probs.as<float>()[k] - r.probs[k]);
                max_perr = std::max(max_perr, pe);
                CHECK(pe <= 1e-5);
            }
        }
    }
    std::printf("        router: max |dprob| %.2e, min top-k/next logit gap %.3f\n", max_perr, min_gap);

    // Tie case with synthetic weights (W = 0, so logits = bias): lowest index wins.
    std::vector<uint16_t> zw(size_t(c.num_experts) * H, 0), bias(c.num_experts, f32_to_bf16(0.0f));
    for (int e : {20, 9, 5}) bias[e] = f32_to_bf16(2.0f);
    for (int e : {30, 1, 17}) bias[e] = f32_to_bf16(1.0f);
    TensorRef tw, tb;
    tw.buf = upload(dev, zw.data(), zw.size() * 2); tw.cpu = tw.buf.as<uint8_t>(); tw.name = "tie.w";
    tb.buf = upload(dev, bias.data(), bias.size() * 2); tb.cpu = tb.buf.as<uint8_t>(); tb.name = "tie.b";
    auto x = random_vec(H, 7, 1.0f);
    auto xb = upload(dev, x.data(), H * 4);
    auto cs = dev.stream();
    cs.begin();
    encode_moe_router(dev, cs, c, TensorRef{}, tw, tb, xb, s, false);
    encode_moe_gate_up(dev, cs, c, w.layers[0], s);
    cs.submit_and_wait();
    const int want[4] = {5, 9, 20, 1};
    const double e1 = std::exp(-1.0), den = 3 + e1;
    const double wantp[4] = {1 / den, 1 / den, 1 / den, e1 / den};
    for (int k = 0; k < 4; ++k) {
        CHECK_EQ(s.expert_ids.as<int32_t>()[k], want[k]);
        CHECK_NEAR(s.probs.as<float>()[k], wantp[k], 1e-6);
    }
    for (uint32_t i = 0; i < H; ++i) CHECK_EQ(s.normed.as<float>()[i], x[i]);  // no-norm path copies x
}

CORAL_TEST(moe_gate_up_swiglu_matches_cpu) {
    auto& f = fixture();
    auto& dev = *f.dev;
    const ModelConfig& c = f.model->config();
    const LayerWeights& L = model_weights(*f.model).layers[0];
    const uint32_t H = c.hidden_size, I = c.intermediate_size, K = c.experts_per_token;
    MoeScratch s = make_moe_scratch(dev, c);
    std::printf("        gate_up_blocks addr %% 16 = %llu, down_blocks addr %% 16 = %llu\n",
                (unsigned long long)((L.gate_up_blocks.buf.gpu_address() + L.gate_up_blocks.offset) % 16),
                (unsigned long long)((L.down_blocks.buf.gpu_address() + L.down_blocks.offset) % 16));

    // Forced experts (no router): write normed and ids directly.
    const int forced[4] = {0, 31, 7, 18};
    auto x = random_vec(H, 11, 1.0f);
    std::memcpy(s.normed.data(), x.data(), H * 4);
    std::memcpy(s.expert_ids.data(), forced, sizeof(forced));
    auto cs = dev.stream();
    cs.begin();
    encode_moe_gate_up(dev, cs, c, L, s, /*forced_ids=*/true);
    cs.submit_and_wait();

    std::vector<double> xd(x.begin(), x.end());
    double max_abs = 0, max_rel = 0;
    size_t clamped = 0;
    for (uint32_t k = 0; k < K; ++k) {
        auto h = ref_gate_up(L, c, forced[k], xd);
        const float* got = s.h.as<float>() + size_t(k) * I;
        for (uint32_t i = 0; i < I; ++i) {
            const double err = std::fabs(got[i] - h[i]);
            max_abs = std::max(max_abs, err);
            max_rel = std::max(max_rel, err / std::max(std::fabs(h[i]), 1e-3));
            clamped += std::fabs(h[i]) > 7.0;
            if (err > 1e-4 + 1e-3 * std::fabs(h[i]))
                CHECK_NEAR(got[i], h[i], 1e-4 + 1e-3 * std::fabs(h[i]));
        }
    }
    std::printf("        gate_up+swiglu: max abs err %.2e, max rel err %.2e (%zu outputs beyond limit)\n",
                max_abs, max_rel, clamped);
}

CORAL_TEST(moe_decode_matches_cpu) {
    auto& f = fixture();
    auto& dev = *f.dev;
    const ModelConfig& c = f.model->config();
    const Weights& w = model_weights(*f.model);
    const uint32_t H = c.hidden_size, K = c.experts_per_token;
    MoeScratch s = make_moe_scratch(dev, c);

    double max_err = 0, max_delta = 0;
    for (uint32_t layer : {0u, 1u}) {
        const LayerWeights& L = w.layers[layer];
        for (uint32_t seed = 0; seed < 3; ++seed) {
            // Residual-stream-like input: mostly O(1), a few large channels.
            auto x = random_vec(H, 1000 + seed * 7 + layer, 0.5f + seed);
            x[seed * 101 % H] = 40.0f; x[(seed * 977 + 13) % H] = -25.0f;
            auto res = upload(dev, x.data(), H * 4);
            auto cs = dev.stream();
            cs.begin();
            encode_moe_decode(dev, cs, c, L, res, s);
            cs.submit_and_wait();

            auto xn = ref_rmsnorm(x, L.mlp_norm, c.rms_norm_eps);
            Routing r = ref_router(xn, L.router_w, L.router_b, c.num_experts, K);
            for (uint32_t k = 0; k < K; ++k) CHECK_EQ(s.expert_ids.as<int32_t>()[k], r.ids[k]);
            std::vector<double> out(x.begin(), x.end());
            for (uint32_t k = 0; k < K; ++k) {
                auto d = ref_down(L, c, r.ids[k], ref_gate_up(L, c, r.ids[k], xn));
                for (uint32_t o = 0; o < H; ++o) out[o] += r.probs[k] * d[o];
            }
            for (uint32_t o = 0; o < H; ++o) {
                const double got = res.as<float>()[o];
                max_err = std::max(max_err, std::fabs(got - out[o]));
                max_delta = std::max(max_delta, std::fabs(out[o] - x[o]));
                CHECK_NEAR(got, out[o], 1e-2);
            }
        }
    }
    std::printf("        moe decode (layers 0,1 x 3 inputs): max abs err %.2e (max |moe output| %.2f)\n",
                max_err, max_delta);
}

CORAL_TEST(moe_decode_throughput) {
    auto& f = fixture();
    auto& dev = *f.dev;
    const ModelConfig& c = f.model->config();
    const Weights& w = model_weights(*f.model);
    const uint32_t H = c.hidden_size;
    MoeScratch s = make_moe_scratch(dev, c);
    auto x = random_vec(H, 5, 1.0f);
    auto res = upload(dev, x.data(), H * 4);
    const double bytes = double(moe_decode_bytes_per_layer(c));
    const int iters = 20;

    auto run = [&](bool rotate) {
        auto cs = dev.stream();
        cs.begin();   // warm-up (pipelines, residency)
        encode_moe_decode(dev, cs, c, w.layers[0], res, s);
        cs.submit_and_wait();
        double best = 1e30;
        for (int rep = 0; rep < 5; ++rep) {
            std::memcpy(res.data(), x.data(), H * 4);
            cs.begin();
            for (int i = 0; i < iters; ++i)
                encode_moe_decode(dev, cs, c, w.layers[rotate ? i % c.num_layers : 0], res, s);
            best = std::min(best, timed_submit(cs));
        }
        return best;
    };
    const double t1 = run(false), tr = run(true);
    std::printf("        moe decode, %d iters: layer 0 repeated %.1f us/layer %.0f GB/s; "
                "rotating layers %.1f us/layer %.0f GB/s (%.1f MB/layer, 3 dispatches)\n",
                iters, t1 / iters * 1e6, bytes * iters / t1 / 1e9, tr / iters * 1e6, bytes * iters / tr / 1e9,
                bytes / 1e6);
    CHECK(tr > 0);
}

// Per-stage timing (set CORAL_MOE_BENCH=1). Also compares the 8-aligned
// (checkpoint) block layout against a 16-aligned copy of layer 0.
CORAL_TEST(moe_bench_stages) {
    if (!std::getenv("CORAL_MOE_BENCH")) SKIP("set CORAL_MOE_BENCH=1");
    auto& f = fixture();
    auto& dev = *f.dev;
    const ModelConfig& c = f.model->config();
    const Weights& w = model_weights(*f.model);
    const uint32_t H = c.hidden_size, I = c.intermediate_size, K = c.experts_per_token;
    MoeScratch s = make_moe_scratch(dev, c);
    auto x = random_vec(H, 5, 1.0f);
    auto res = upload(dev, x.data(), H * 4);
    const int iters = 24;
    auto cs = dev.stream();
    cs.begin(); encode_moe_decode(dev, cs, c, w.layers[0], res, s); cs.submit_and_wait();
    const double gu_b = K * (2.0 * I * (H / 32) * 17 + 2 * I * 2), dn_b = K * (double(H) * (I / 32) * 17 + H * 2);

    auto time = [&](auto&& enc) {
        double best = 1e30;
        for (int rep = 0; rep < 10; ++rep) {
            cs.begin();
            for (int i = 0; i < iters; ++i) enc(w.layers[i % c.num_layers]);
            best = std::min(best, timed_submit(cs));
        }
        return best / iters;
    };
    const double tr = time([&](const LayerWeights& L) { encode_moe_router(dev, cs, c, L.mlp_norm, L.router_w, L.router_b, res, s, true); });
    const double tg = time([&](const LayerWeights& L) { encode_moe_gate_up(dev, cs, c, L, s); });
    const double td = time([&](const LayerWeights& L) { encode_moe_down(dev, cs, c, L, res, s); });
    const double ta = time([&](const LayerWeights& L) { encode_moe_decode(dev, cs, c, L, res, s); });
    std::printf("        router %.1f us | gate_up %.1f us %.0f GB/s | down %.1f us %.0f GB/s | all %.1f us %.0f GB/s\n",
                tr * 1e6, tg * 1e6, gu_b / tg / 1e9, td * 1e6, dn_b / td / 1e9, ta * 1e6,
                double(moe_decode_bytes_per_layer(c)) / ta / 1e9);
    for (int l = 0; l < int(c.num_layers); ++l) {
        const auto& L = w.layers[l];
        std::printf("%d:%llu/%llu ", l, (unsigned long long)((L.gate_up_blocks.buf.gpu_address() + L.gate_up_blocks.offset) % 16),
                    (unsigned long long)((L.down_blocks.buf.gpu_address() + L.down_blocks.offset) % 16));
    }
    std::printf("\n");
}

// Kernel-shape sweep (set CORAL_MOE_LAB=1): gate_up pairs/simdgroups and down
// rows/simdgroups, rotating over all 24 layers, checked against the default.
CORAL_TEST(moe_lab_sweep) {
    if (!std::getenv("CORAL_MOE_LAB")) SKIP("set CORAL_MOE_LAB=1");
    auto& f = fixture();
    auto& dev = *f.dev;
    const ModelConfig& c = f.model->config();
    const Weights& w = model_weights(*f.model);
    const uint32_t H = c.hidden_size, I = c.intermediate_size, K = c.experts_per_token;
    MoeScratch s = make_moe_scratch(dev, c);
    auto x = random_vec(H, 5, 1.0f);
    auto res = upload(dev, x.data(), H * 4);
    const int iters = 24;
    auto cs = dev.stream();
    const MoeKernelConfig def = moe_kernel_config();
    const double gu_b = K * (2.0 * I * (H / 32) * 17 + 2 * I * 2), dn_b = K * (double(H) * (I / 32) * 17 + H * 2);
    // Reference outputs with the default config on layer 3.
    std::memcpy(res.data(), x.data(), H * 4);
    cs.begin(); encode_moe_decode(dev, cs, c, w.layers[3], res, s); cs.submit_and_wait();
    std::vector<float> ref_res(res.as<float>(), res.as<float>() + H);
    std::vector<float> ref_h(s.h.as<float>(), s.h.as<float>() + size_t(K) * I);
    auto time = [&](auto&& enc) {
        double best = 1e30;
        for (int rep = 0; rep < 15; ++rep) {
            cs.begin();
            for (int i = 0; i < iters; ++i) enc(w.layers[i % c.num_layers]);
            best = std::min(best, timed_submit(cs));
        }
        return best / iters;
    };
    auto check = [&]() {
        std::memcpy(res.data(), x.data(), H * 4);
        cs.begin(); encode_moe_decode(dev, cs, c, w.layers[3], res, s); cs.submit_and_wait();
        double eh = 0, er = 0;
        for (size_t i = 0; i < ref_h.size(); ++i) eh = std::max(eh, double(std::fabs(s.h.as<float>()[i] - ref_h[i])));
        for (uint32_t i = 0; i < H; ++i) er = std::max(er, double(std::fabs(res.as<float>()[i] - ref_res[i])));
        return std::pair<double, double>(eh, er);
    };
    // Kernel shapes (see MoeKernelConfig); the lane-mapping and loop-structure
    // variants that lost are documented in src/kernels/moe.metal.
    for (uint32_t pairs : {1u, 2u, 4u})
        for (uint32_t sg : {1u, 2u, 4u, 8u}) {
            moe_kernel_config() = def;
            moe_kernel_config().gu_pairs = pairs; moe_kernel_config().gu_sg = sg;
            const double t = time([&](const LayerWeights& L) { encode_moe_gate_up(dev, cs, c, L, s); });
            auto [eh, er] = check();
            std::printf("        gate_up pairs %u sg %u: %6.1f us %5.0f GB/s  (dh %.1e)\n", pairs, sg, t * 1e6, gu_b / t / 1e9, eh);
            CHECK(eh < 1e-3);
        }
    for (uint32_t rows : {1u, 2u, 4u})
        for (uint32_t sg : {1u, 2u, 4u, 8u}) {
            moe_kernel_config() = def;
            moe_kernel_config().dn_rows = rows; moe_kernel_config().dn_sg = sg;
            const double t = time([&](const LayerWeights& L) { encode_moe_down(dev, cs, c, L, res, s); });
            auto [eh, er] = check();
            std::printf("        down rows %u sg %u: %6.1f us %5.0f GB/s  (dres %.1e)\n", rows, sg, t * 1e6, dn_b / t / 1e9, er);
            CHECK(er < 1e-2);
        }
    moe_kernel_config() = def;
    const double tr = time([&](const LayerWeights& L) { encode_moe_router(dev, cs, c, L.mlp_norm, L.router_w, L.router_b, res, s, true); });
    const double tg = time([&](const LayerWeights& L) { encode_moe_gate_up(dev, cs, c, L, s); });
    std::printf("        router (+ top-k) %.1f us | gate_up %.1f us\n", tr * 1e6, tg * 1e6);
    moe_kernel_config() = def;
}

// Raw read bandwidth roofline (set CORAL_MOE_LAB=1).
CORAL_TEST(moe_lab_roofline) {
    if (!std::getenv("CORAL_MOE_LAB")) SKIP("set CORAL_MOE_LAB=1");
    auto& f = fixture();
    auto& dev = *f.dev;
    const Weights& w = model_weights(*f.model);
    gpu::Buffer out = dev.alloc(16);
    auto cs = dev.stream();
    struct P { uint32_t n16, unroll; };
    for (double mb : {26.0, 53.0, 1160.0}) {
        const size_t bytes = size_t(mb * 1e6) / 16 * 16;
        gpu::Buffer b = dev.alloc(bytes, true);
        for (uint32_t tgs : {256u, 1024u}) {
            for (uint32_t groups : {76u * 4, 76u * 16, 76u * 64}) {
                double best = 1e30;
                for (int rep = 0; rep < 8; ++rep) {
                    cs.begin();
                    for (int i = 0; i < 10; ++i)
                        cs.dispatch(dev.kernel("lab_stream_read"), gpu::Args().buffer(0, b).buffer(1, out).value(2, P{uint32_t(bytes / 16), 1}),
                                    {groups}, {tgs});
                    best = std::min(best, timed_submit(cs) / 10);
                }
                std::printf("        stream %.0f MB tg %u groups %u: %.1f us %.0f GB/s\n", mb, tgs, groups, best * 1e6, bytes / best / 1e9);
            }
        }
        for (uint32_t chunk : {90u, 360u, 1440u}) {
            double best = 1e30;
            const uint32_t n16 = uint32_t(bytes / 16), nch = (n16 + chunk - 1) / chunk;
            for (int rep = 0; rep < 8; ++rep) {
                cs.begin();
                for (int i = 0; i < 10; ++i)
                    cs.dispatch(dev.kernel("lab_chunk_read"), gpu::Args().buffer(0, b).buffer(1, out).value(2, P{n16, chunk}),
                                {(nch + 7) / 8}, {256});
                best = std::min(best, timed_submit(cs) / 10);
            }
            std::printf("        chunk %u B/sg %.0f MB: %.1f us %.0f GB/s\n", chunk * 16, mb, best * 1e6, bytes / best / 1e9);
        }
    }
    // Rotating over the 24 real wq tensors (23.6 MB each, no SLC reuse): the
    // best a 24 MB streaming dispatch can do.
    for (uint32_t chunk : {90u, 360u, 1440u}) {
        double best = 1e30;
        const uint32_t n16 = uint32_t(w.layers[0].wq.nbytes / 16), nch = (n16 + chunk - 1) / chunk;
        for (int rep = 0; rep < 8; ++rep) {
            cs.begin();
            for (int i = 0; i < 24; ++i) {
                const auto& t = w.layers[i].wq;
                cs.dispatch(dev.kernel("lab_chunk_read"), gpu::Args().buffer(0, t.buf, t.offset / 16 * 16).buffer(1, out).value(2, P{n16, chunk}),
                            {(nch + 7) / 8}, {256});
            }
            best = std::min(best, timed_submit(cs) / 24);
        }
        std::printf("        wq x24 chunk %u B/sg: %.1f us %.0f GB/s\n", chunk * 16, best * 1e6, w.layers[0].wq.nbytes / best / 1e9);
    }
    for (uint32_t groups : {1u, 32u, 360u, 1440u}) {
        double best = 1e30;
        for (int rep = 0; rep < 8; ++rep) {
            cs.begin();
            for (int i = 0; i < 200; ++i) cs.dispatch(dev.kernel("lab_empty"), gpu::Args().buffer(0, out), {groups}, {256});
            best = std::min(best, timed_submit(cs) / 200);
        }
        std::printf("        empty dispatch + barrier, %u groups: %.2f us\n", groups, best * 1e6);
    }
    (void)w;
}
