// Shared MSL definitions for all coral kernels. Inlined ahead of every
// .metal file by tools/embed_shaders.cpp.
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Numeric helpers
// ---------------------------------------------------------------------------
inline float bf2f(bfloat x) { return float(x); }
inline bfloat f2bf(float x) { return bfloat(x); }

// Threadgroup-wide sum for a 1-D threadgroup of up to 1024 threads.
// `scratch` must hold at least 32 floats. All threads must call this.
inline float tg_sum(float v, threadgroup float* scratch,
                    uint tid, uint tg_size, uint simd_lane, uint simd_id) {
    v = simd_sum(v);
    if (simd_lane == 0) scratch[simd_id] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nsimd = (tg_size + 31) / 32;
    float total = 0.0f;
    if (tid < 32) {
        float x = (tid < nsimd) ? scratch[tid] : 0.0f;
        total = simd_sum(x);
        if (tid == 0) scratch[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

// ---------------------------------------------------------------------------
// MXFP4 (OCP Microscaling FP4, E2M1 elements + E8M0 shared scale, block 32)
// as stored in the gpt-oss checkpoints:
//   blocks: uint8[rows][K/32][16]  — 32 nibbles per block, low nibble first
//   scales: uint8[rows][K/32]      — value = 2^(scale - 127)
// ---------------------------------------------------------------------------
constant float kFp4Lut[16] = {
    +0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +3.0f, +4.0f, +6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// 2^(e - 127) without a table: build the float bit pattern directly.
// e == 0 would be a denormal scale in E8M0 terms (2^-127); treat it as such
// via ldexp for exactness rather than the bit trick.
inline float e8m0_to_float(uint e) {
    return (e == 0) ? ldexp(1.0f, -127) : as_type<float>(e << 23);
}

// Decode two elements from one byte.
inline float2 fp4x2(uint byte) {
    return float2(kFp4Lut[byte & 0xF], kFp4Lut[byte >> 4]);
}
