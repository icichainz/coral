#include "common.h"

// Self-test kernels used by `coral selftest` and the unit tests to prove the
// runtime compile + Metal 4 dispatch path end to end.

struct SmokeParams { uint n; float scale; };

kernel void smoke_axpy_bf16(device const bfloat*  a      [[buffer(0)]],
                            device const bfloat*  b      [[buffer(1)]],
                            device bfloat*        out    [[buffer(2)]],
                            constant SmokeParams& p      [[buffer(3)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= p.n) return;
    out[id] = f2bf(p.scale * bf2f(a[id]) + bf2f(b[id]));
}

// Writes the simd-group sum of thread ids: verifies simdgroup intrinsics.
kernel void smoke_simd_sum(device float* out [[buffer(0)]],
                           uint id [[thread_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
    float v = simd_sum(float(id));
    if (lane == 0) out[id / 32] = v;
}
