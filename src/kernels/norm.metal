#include "common.h"

// RMSNorm, as used by gpt-oss: y = x * rsqrt(mean(x^2) + eps) * w
// Computed in fp32, stored as bf16. One threadgroup per row.
//
//   x   : bf16 [rows][n]
//   w   : bf16 [n]
//   y   : bf16 [rows][n]
// Threadgroup size: any multiple of 32 up to 1024; 256 is a good default for n=2880.

struct RmsNormParams { uint n; float eps; };

kernel void rmsnorm_bf16(device const bfloat*    x   [[buffer(0)]],
                         device const bfloat*    w   [[buffer(1)]],
                         device bfloat*          y   [[buffer(2)]],
                         constant RmsNormParams& p   [[buffer(3)]],
                         uint  row      [[threadgroup_position_in_grid]],
                         uint  tid      [[thread_position_in_threadgroup]],
                         uint  tg_size  [[threads_per_threadgroup]],
                         uint  lane     [[thread_index_in_simdgroup]],
                         uint  simd_id  [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[32];
    device const bfloat* xr = x + (ulong)row * p.n;
    device bfloat*       yr = y + (ulong)row * p.n;

    float ss = 0.0f;
    for (uint i = tid; i < p.n; i += tg_size) { float v = bf2f(xr[i]); ss += v * v; }
    ss = tg_sum(ss, scratch, tid, tg_size, lane, simd_id);
    const float inv = rsqrt(ss / float(p.n) + p.eps);

    for (uint i = tid; i < p.n; i += tg_size)
        yr[i] = f2bf(bf2f(xr[i]) * inv * bf2f(w[i]));
}
