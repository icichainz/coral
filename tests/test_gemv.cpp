// bf16 GEMV, fp32 RMSNorm and argmax kernels (src/kernels/gemv_bf16.metal)
// against CPU references, plus achieved-bandwidth reports on real weights.
#include "test.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include "coral/gpu.h"
#include "coral/kernels.h"
#include "coral/model.h"
#include "coral/safetensors.h"
#include "../src/model/attention_ops.h"
#include "../src/model/weights.h"

using namespace coral;

// Device (+ lazily, the loaded model) shared by test_gemv.cpp and
// test_attention.cpp. Leaked on purpose: the no-copy shard buffers must
// outlive every GPU object (same reasoning as test_model.cpp).
struct AttnTestFixture {
    std::unique_ptr<gpu::Device> dev;
    std::unique_ptr<Model> model;
};

gpu::Device& attn_test_device() {
    static AttnTestFixture* f = [] {
        auto* x = new AttnTestFixture;
        x->dev = gpu::Device::create();
        load_kernels(*x->dev);
        return x;
    }();
    return *f->dev;
}

Model& attn_test_model() {
    std::string dir = test::model_dir_or_skip();
    static Model* m = [&] {
        auto t0 = std::chrono::steady_clock::now();
        Model* p = Model::load(attn_test_device(), dir).release();
        attn_test_device().make_resident();
        std::printf("        Model::load (attention/gemv tests): %.1f s\n",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        return p;
    }();
    return *m;
}

namespace {

// Fast deterministic generator for large random fills.
struct XorShift {
    uint64_t s;
    explicit XorShift(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    float uniform() { return float(next() >> 40) * (1.0f / 16777216.0f) * 2.0f - 1.0f; }   // [-1, 1)
};

// A bf16 [rows, K] tensor in a fresh buffer, deliberately placed at 8 mod 16
// bytes like the real safetensors tensors.
TensorRef random_bf16(gpu::Device& dev, uint32_t rows, uint32_t K, uint64_t seed, float scale = 1.0f) {
    const size_t off = 8, n = size_t(rows) * K;
    TensorRef t;
    t.name = "random";
    t.buf = dev.alloc(off + n * 2);
    t.offset = off;
    t.cpu = static_cast<const uint8_t*>(t.buf.data()) + off;
    t.dtype = DType::BF16;
    t.shape = rows == 1 ? std::vector<int64_t>{int64_t(K)} : std::vector<int64_t>{int64_t(rows), int64_t(K)};
    t.nbytes = n * 2;
    uint16_t* p = reinterpret_cast<uint16_t*>(static_cast<uint8_t*>(t.buf.data()) + off);
    XorShift r(seed);
    for (size_t i = 0; i < n; ++i) p[i] = f32_to_bf16(r.uniform() * scale);
    return t;
}

gpu::Buffer random_f32(gpu::Device& dev, uint32_t n, uint64_t seed, float scale = 1.0f, float bias = 0.0f) {
    gpu::Buffer b = dev.alloc(size_t(n) * 4);
    XorShift r(seed);
    for (uint32_t i = 0; i < n; ++i) b.as<float>()[i] = r.uniform() * scale + bias;
    return b;
}

float bfv(const TensorRef& t, size_t i) { return bf16_to_f32(t.as<uint16_t>()[i]); }

// Check y against a double-precision reference for the listed rows.
void check_gemv(const TensorRef& W, const float* x, const float* y, const float* y0, const GemvOptions& o,
                uint32_t row_step, const char* what) {
    const uint32_t rows = uint32_t(W.dim(0)), K = uint32_t(W.dim(1));
    double inv = 1.0;
    if (o.norm) {
        double ss = 0; for (uint32_t k = 0; k < K; ++k) ss += double(x[k]) * x[k];
        inv = 1.0 / std::sqrt(ss / K + o.eps);
    }
    double worst = 0;
    auto check_row = [&](uint32_t r) {
        const uint16_t* w = W.as<uint16_t>() + size_t(r) * K;
        double acc = 0, mag = 0;
        for (uint32_t k = 0; k < K; ++k) {
            double xk = x[k] * (o.norm ? double(bfv(*o.norm, k)) : 1.0);
            double t = double(bf16_to_f32(w[k])) * xk;
            acc += t; mag += std::fabs(t);
        }
        acc *= inv; mag *= inv;
        if (o.bias) { acc += bfv(*o.bias, r); mag += std::fabs(bfv(*o.bias, r)); }
        if (o.accumulate) { acc += y0[r]; mag += std::fabs(y0[r]); }
        const double err = std::fabs(double(y[r]) - acc), tol = 2e-6 * mag + 1e-6;
        worst = std::max(worst, err / tol);
        if (err > tol) {
            std::printf("        %s row %u: got %.7g want %.7g (mag %.3g)\n", what, r, y[r], acc, mag);
            CHECK(err <= tol);
        }
    };
    for (uint32_t r = 0; r < rows; r += row_step) check_row(r);
    check_row(rows - 1);
    (void)worst;
}

} // namespace

CORAL_TEST(gemv_bf16_matches_cpu) {
    auto& dev = attn_test_device();
    struct Shape { uint32_t rows, K; };
    const Shape shapes[] = {{4096, 2880}, {512, 2880}, {2880, 4096}, {32, 2880}, {37, 2880}, {201088, 2880}};
    uint64_t seed = 1;
    for (const Shape& s : shapes) {
        const TensorRef W = random_bf16(dev, s.rows, s.K, seed++);
        const TensorRef bias = random_bf16(dev, 1, s.rows, seed++, 0.5f);
        const TensorRef norm = random_bf16(dev, 1, s.K, seed++, 0.5f);
        for (uint32_t i = 0; i < s.K; ++i)   // norm weights around 1
            const_cast<uint16_t*>(norm.as<uint16_t>())[i] = f32_to_bf16(1.0f + 0.5f * bfv(norm, i));
        gpu::Buffer x = random_f32(dev, s.K, seed++, 2.0f);
        gpu::Buffer y = dev.alloc(size_t(s.rows) * 4);
        const uint32_t step = s.rows > 10000 ? 97 : 1;

        struct Variant { bool bias, accum, norm; uint32_t R; };
        std::vector<Variant> vs = {{false, false, false, 0}, {true, false, false, 0}, {true, true, false, 0},
                                   {true, false, true, 0}};
        if (s.rows == 4096 || s.rows == 37)
            for (uint32_t R : {1u, 2u, 4u, 8u}) { vs.push_back({false, false, false, R}); vs.push_back({true, true, true, R}); }
        for (const Variant& v : vs) {
            GemvOptions o;
            o.bias = v.bias ? &bias : nullptr;
            o.accumulate = v.accum;
            o.norm = v.norm ? &norm : nullptr;
            o.rows_per_simdgroup = v.R;
            std::vector<float> y0(s.rows);
            XorShift r(seed++);
            for (auto& e : y0) e = r.uniform();
            std::memcpy(y.data(), y0.data(), y0.size() * 4);
            auto cs = dev.stream();
            cs.begin();
            encode_gemv_bf16(dev, cs, W, x, 0, y, 0, o);
            cs.submit_and_wait();
            char what[96];
            std::snprintf(what, sizeof what, "gemv %ux%u b%d a%d n%d R%u", s.rows, s.K, v.bias, v.accum, v.norm, v.R);
            check_gemv(W, x.as<float>(), y.as<float>(), y0.data(), o, step, what);
        }
    }
}

CORAL_TEST(gemv_bf16_rejects_misaligned) {
    auto& dev = attn_test_device();
    TensorRef W = random_bf16(dev, 8, 64, 3);
    gpu::Buffer x = dev.alloc(64 * 4 + 16), y = dev.alloc(8 * 4 + 16);
    auto cs = dev.stream();
    cs.begin();
    TensorRef bad = W; bad.offset += 2;
    CHECK_THROWS(encode_gemv_bf16(dev, cs, bad, x, 0, y, 0));       // weights 2-byte aligned
    CHECK_THROWS(encode_gemv_bf16(dev, cs, W, x, 4, y, 0));         // x not 16-byte aligned
    TensorRef badk = W; badk.shape = {16, 60};
    CHECK_THROWS(encode_gemv_bf16(dev, cs, badk, x, 0, y, 0));      // K % 8 != 0
    encode_gemv_bf16(dev, cs, W, x, 16, y, 16);
    cs.submit_and_wait();
}

CORAL_TEST(gemv_rmsnorm_f32_matches_cpu) {
    auto& dev = attn_test_device();
    const uint32_t n = 2880, rows = 3;
    TensorRef w = random_bf16(dev, 1, n, 11);
    gpu::Buffer x = random_f32(dev, n * rows, 12, 3.0f), y = dev.alloc(n * rows * 4);
    auto cs = dev.stream();
    cs.begin();
    encode_rmsnorm_f32(dev, cs, w, 1e-5f, x, 0, y, 0, rows);
    cs.submit_and_wait();
    for (uint32_t r = 0; r < rows; ++r) {
        const float* xr = x.as<float>() + r * n;
        double ss = 0; for (uint32_t i = 0; i < n; ++i) ss += double(xr[i]) * xr[i];
        const double inv = 1.0 / std::sqrt(ss / n + 1e-5);
        for (uint32_t i = 0; i < n; ++i) CHECK_NEAR(y.as<float>()[r * n + i], xr[i] * inv * bfv(w, i), 1e-5);
    }
}

CORAL_TEST(gemv_argmax_matches_cpu) {
    auto& dev = attn_test_device();
    gpu::Buffer out = dev.alloc(4), scratch = dev.alloc(argmax_scratch_bytes());
    auto run = [&](const std::vector<float>& v) {
        gpu::Buffer x = dev.alloc(v.size() * 4);
        std::memcpy(x.data(), v.data(), v.size() * 4);
        auto cs = dev.stream();
        cs.begin();
        encode_argmax_f32(dev, cs, x, 0, uint32_t(v.size()), out, 0, scratch);
        cs.submit_and_wait();
        return out.as<int32_t>()[0];
    };
    auto cpu = [](const std::vector<float>& v) { return int32_t(std::max_element(v.begin(), v.end()) - v.begin()); };

    const uint32_t V = 201088;
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 4.f);
    std::vector<float> v(V);
    for (auto& e : v) e = nd(rng);
    CHECK_EQ(run(v), cpu(v));

    // Ties: the lowest index wins, wherever the copies land among threads.
    const float top = 1000.f;
    v[150001] = top; v[70000] = top; v[200000] = top;
    CHECK_EQ(run(v), 70000);
    v[3] = top;
    CHECK_EQ(run(v), 3);

    // All negative, maximum in the last element.
    for (auto& e : v) e = -std::fabs(nd(rng)) - 10.f;
    v[V - 1] = -1.f;
    CHECK_EQ(run(v), int32_t(V - 1));
    // All equal -> 0.
    std::fill(v.begin(), v.end(), -3.5f);
    CHECK_EQ(run(v), 0);
    // Odd sizes.
    for (uint32_t n : {1u, 31u, 1025u, 262145u}) {
        std::vector<float> w(n);
        for (auto& e : w) e = nd(rng);
        CHECK_EQ(run(w), cpu(w));
    }
}

// Achieved bandwidth on the real lm_head (1.16 GB bf16) for each rows-per-simdgroup variant.
CORAL_TEST(gemv_bf16_lm_head_bandwidth) {
    Model& m = attn_test_model();
    auto& dev = attn_test_device();
    const Weights& w = model_weights(m);
    const uint32_t V = uint32_t(w.lm_head.dim(0)), H = uint32_t(w.lm_head.dim(1));
    gpu::Buffer x = random_f32(dev, H, 21), y = dev.alloc(size_t(V) * 4);
    const int iters = 20;
    const double bytes = double(w.lm_head.nbytes);
    double best = 0;
    for (uint32_t R : {1u, 2u, 4u, 8u, 0u}) {
        GemvOptions o;
        o.rows_per_simdgroup = R;
        o.norm = &w.final_norm;
        auto cs = dev.stream();
        cs.begin();  // warm-up
        encode_gemv_bf16(dev, cs, w.lm_head, x, 0, y, 0, o);
        cs.submit_and_wait();
        cs.begin();
        for (int i = 0; i < iters; ++i) encode_gemv_bf16(dev, cs, w.lm_head, x, 0, y, 0, o);
        const double t = cs.submit_and_wait() / iters;
        std::printf("        lm_head gemv (norm fused) R=%u%s: %.3f ms  %.1f GB/s\n", R ? R : 4, R ? "" : " (default)",
                    t * 1e3, bytes / t * 1e-9);
        best = std::max(best, bytes / t * 1e-9);
    }
    // Spot-check correctness on the real weights.
    GemvOptions o;
    o.norm = &w.final_norm;
    check_gemv(w.lm_head, x.as<float>(), y.as<float>(), nullptr, o, 1009, "lm_head");
    CHECK(best > 150.0);   // loose floor; target >= 300 GB/s on M2 Max
}
