#include "test.h"

#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "coral/gpu.h"
#include "coral/kernels.h"
#include "coral/safetensors.h"  // bf16 helpers

using namespace coral;

namespace {
// One device + compiled kernels shared by all GPU tests.
gpu::Device& device() {
    static std::unique_ptr<gpu::Device> dev = [] {
        auto d = gpu::Device::create();
        load_kernels(*d);
        return d;
    }();
    return *dev;
}

// CPU MXFP4 decode, independent of the kernel.
float fp4_lut(int i) {
    static const float lut[16] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6, -0.f, -0.5f, -1, -1.5f, -2, -3, -4, -6};
    return lut[i & 15];
}
} // namespace

CORAL_TEST(gpu_device_info) {
    const auto& i = device().info();
    CHECK(!i.name.empty());
    CHECK(i.metal4);
    CHECK(i.unified_memory);
    CHECK(i.max_buffer_bytes > (1ull << 30));
}

CORAL_TEST(gpu_smoke_axpy) {
    auto& dev = device();
    const uint32_t n = 100003;  // deliberately not a multiple of the threadgroup
    auto a = dev.alloc(n * 2), b = dev.alloc(n * 2), out = dev.alloc(n * 2, true);
    for (uint32_t i = 0; i < n; ++i) { a.as<uint16_t>()[i] = f32_to_bf16(float(i % 50)); b.as<uint16_t>()[i] = f32_to_bf16(0.5f); }
    struct { uint32_t n; float scale; } p{n, 3.0f};
    auto cs = dev.stream();
    cs.begin();
    cs.dispatch_threads(dev.kernel("smoke_axpy_bf16"), gpu::Args().buffer(0, a).buffer(1, b).buffer(2, out).value(3, p), {n}, {256});
    cs.submit_and_wait();
    auto want = [](uint32_t i) { return bf16_to_f32(f32_to_bf16(3.0f * float(i % 50) + 0.5f)); };  // bf16 output rounding
    for (uint32_t i = 0; i < n; i += 977) CHECK_EQ(bf16_to_f32(out.as<uint16_t>()[i]), want(i));
    CHECK_EQ(bf16_to_f32(out.as<uint16_t>()[n - 1]), want(n - 1));
}

CORAL_TEST(gpu_simd_sum) {
    auto& dev = device();
    const uint32_t n = 1024;
    auto out = dev.alloc(n / 32 * 4, true);
    auto cs = dev.stream();
    cs.begin();
    cs.dispatch_threads(dev.kernel("smoke_simd_sum"), gpu::Args().buffer(0, out), {n}, {256});
    cs.submit_and_wait();
    for (uint32_t g = 0; g < n / 32; ++g) {
        float want = 0; for (uint32_t i = g * 32; i < g * 32 + 32; ++i) want += float(i);
        CHECK_EQ(out.as<float>()[g], want);
    }
}

CORAL_TEST(gpu_dependent_dispatches_are_ordered) {
    // out = 2*a + b, then out2 = 2*out + b: the second dispatch must see the first.
    auto& dev = device();
    const uint32_t n = 4096;
    auto a = dev.alloc(n * 2), b = dev.alloc(n * 2), t = dev.alloc(n * 2, true), out = dev.alloc(n * 2, true);
    for (uint32_t i = 0; i < n; ++i) { a.as<uint16_t>()[i] = f32_to_bf16(1.0f); b.as<uint16_t>()[i] = f32_to_bf16(1.0f); }
    struct { uint32_t n; float scale; } p{n, 2.0f};
    auto k = dev.kernel("smoke_axpy_bf16");
    auto cs = dev.stream();
    cs.begin();
    for (int r = 0; r < 50; ++r) {
        cs.dispatch_threads(k, gpu::Args().buffer(0, a).buffer(1, b).buffer(2, t).value(3, p), {n}, {256});     // t = 2a+b = 3
        cs.dispatch_threads(k, gpu::Args().buffer(0, t).buffer(1, b).buffer(2, out).value(3, p), {n}, {256});   // out = 2t+b = 7
    }
    cs.submit_and_wait();
    CHECK_EQ(cs.dispatch_count(), size_t(100));
    for (uint32_t i = 0; i < n; i += 61) CHECK_EQ(bf16_to_f32(out.as<uint16_t>()[i]), 7.0f);
}

CORAL_TEST(gpu_rmsnorm_matches_cpu) {
    auto& dev = device();
    const uint32_t rows = 7, n = 2880;
    const float eps = 1e-5f;
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto x = dev.alloc(rows * n * 2), w = dev.alloc(n * 2), y = dev.alloc(rows * n * 2, true);
    std::vector<float> xf(rows * n), wf(n);
    for (uint32_t i = 0; i < rows * n; ++i) { float v = nd(rng) * 3.f; x.as<uint16_t>()[i] = f32_to_bf16(v); xf[i] = bf16_to_f32(x.as<uint16_t>()[i]); }
    for (uint32_t i = 0; i < n; ++i) { float v = 1.f + 0.1f * nd(rng); w.as<uint16_t>()[i] = f32_to_bf16(v); wf[i] = bf16_to_f32(w.as<uint16_t>()[i]); }

    struct { uint32_t n; float eps; } p{n, eps};
    auto cs = dev.stream();
    cs.begin();
    cs.dispatch(dev.kernel("rmsnorm_bf16"), gpu::Args().buffer(0, x).buffer(1, w).buffer(2, y).value(3, p), {rows}, {256});
    cs.submit_and_wait();

    for (uint32_t r = 0; r < rows; ++r) {
        double ss = 0; for (uint32_t i = 0; i < n; ++i) ss += double(xf[r * n + i]) * xf[r * n + i];
        float inv = 1.0f / std::sqrt(float(ss / n) + eps);
        for (uint32_t i = 0; i < n; ++i) {
            float want = xf[r * n + i] * inv * wf[i];
            float got = bf16_to_f32(y.as<uint16_t>()[r * n + i]);
            CHECK_NEAR(got, want, 0.02 * std::fabs(want) + 0.02);  // bf16 output rounding
        }
    }
}

CORAL_TEST(gpu_mxfp4_dequant_matches_cpu) {
    auto& dev = device();
    const uint32_t rows = 5, k = 2880, nblk = k / 32;
    std::mt19937 rng(7);
    auto blocks = dev.alloc(rows * nblk * 16), scales = dev.alloc(rows * nblk), out = dev.alloc(rows * k * 2, true);
    for (uint32_t i = 0; i < rows * nblk * 16; ++i) blocks.as<uint8_t>()[i] = uint8_t(rng());
    for (uint32_t i = 0; i < rows * nblk; ++i) scales.as<uint8_t>()[i] = uint8_t(110 + rng() % 30);  // 2^-17 .. 2^12

    struct { uint32_t rows, k; } p{rows, k};
    auto cs = dev.stream();
    cs.begin();
    cs.dispatch_threads(dev.kernel("mxfp4_dequant_bf16"), gpu::Args().buffer(0, blocks).buffer(1, scales).buffer(2, out).value(3, p),
                        {nblk, rows}, {32, 8});
    cs.submit_and_wait();

    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t b = 0; b < nblk; ++b) {
            float s = std::ldexp(1.0f, int(scales.as<uint8_t>()[r * nblk + b]) - 127);
            for (uint32_t i = 0; i < 16; ++i) {
                uint8_t byte = blocks.as<uint8_t>()[(r * nblk + b) * 16 + i];
                float lo = fp4_lut(byte & 15) * s, hi = fp4_lut(byte >> 4) * s;
                CHECK_EQ(bf16_to_f32(out.as<uint16_t>()[r * k + b * 32 + 2 * i]), lo);
                CHECK_EQ(bf16_to_f32(out.as<uint16_t>()[r * k + b * 32 + 2 * i + 1]), hi);
            }
        }
}

CORAL_TEST(gpu_wrap_no_copy) {
    auto& dev = device();
    const size_t pg = gpu::page_size();
    void* mem = nullptr;
    CHECK(posix_memalign(&mem, pg, pg * 2) == 0);
    auto* p = static_cast<uint16_t*>(mem);
    const uint32_t n = uint32_t(pg * 2 / 2);
    for (uint32_t i = 0; i < n; ++i) p[i] = f32_to_bf16(2.0f);
    {
        auto wrapped = dev.wrap_no_copy(mem, pg * 2);
        CHECK_EQ(wrapped.data(), mem);
        auto out = dev.alloc(n * 2, true);
        struct { uint32_t n; float scale; } prm{n, 1.0f};
        auto cs = dev.stream();
        cs.begin();
        cs.dispatch_threads(dev.kernel("smoke_axpy_bf16"), gpu::Args().buffer(0, wrapped).buffer(1, wrapped).buffer(2, out).value(3, prm), {n}, {256});
        cs.submit_and_wait();
        CHECK_EQ(bf16_to_f32(out.as<uint16_t>()[n - 1]), 4.0f);
    }  // GPU buffer released before the memory it wraps
    free(mem);
}
