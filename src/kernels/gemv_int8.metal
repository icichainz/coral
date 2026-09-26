#include "common.h"

// int8 weight GEMV for the unembedding (lm_head), decode path (M = 1).
//
//   y[r] = s[r] * inv * sum_k q[r,k] * (x[k] * g[k])        (NORM)
//   y[r] = s[r] *       sum_k q[r,k] *  x[k]                 (plain)
//
//   q : int8 [rows][K]  row-major, private copy made at load time (16-byte
//                       aligned rows, K % 16 == 0)
//   s : fp32 [rows]     per-row scale (absmax / 127)
//   x : fp32 [K], g : bf16 [K] (RMSNorm weight, NORM only; 8-byte aligned)
// inv = rsqrt(mean(x^2) + eps), computed per threadgroup (as gemv_bf16_norm).
// One simdgroup computes R rows; GEMV_SG simdgroups per threadgroup. Each
// lane reads 16 int8 (one uint4) per row per step: 512 elements per step.
//
// Relies on gemv_bf16.metal (GEMV_SG, gemv_bf4, gemv_inv_rms), which sorts
// before this file in the concatenated translation unit.

struct GemvI8Params { uint rows; uint K; float eps; uint pad; };

inline float4 i8x4(uint u) { return float4(as_type<char4>(u)); }

template <int R, bool NORM>
kernel void gemv_i8_f32(device const char*    W    [[buffer(0)]],
                        device const float*   S    [[buffer(1)]],
                        device const float*   x    [[buffer(2)]],
                        device float*         y    [[buffer(3)]],
                        device const bfloat*  nw   [[buffer(4)]],
                        constant GemvI8Params& p   [[buffer(5)]],
                        uint tg   [[threadgroup_position_in_grid]],
                        uint tid  [[thread_position_in_threadgroup]],
                        uint tgs  [[threads_per_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sid  [[simdgroup_index_in_threadgroup]]) {
    float inv = 1.0f;
    if (NORM) {
        threadgroup float scratch[32];
        inv = gemv_inv_rms(x, p.K, p.eps, scratch, tid, tgs, lane, sid);
    }
    const uint row0 = (tg * GEMV_SG + sid) * R;
    if (row0 >= p.rows) return;
    device const uint4* wp[R];
    for (int r = 0; r < R; ++r) {
        const uint row = min(row0 + uint(r), p.rows - 1);
        wp[r] = (device const uint4*)(W + (ulong)row * p.K);
    }
    device const float4* x4 = (device const float4*)x;
    device const uint2*  g2 = (device const uint2*)nw;
    float acc[R];
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    for (uint k = lane * 16; k < p.K; k += 512) {
        uint4 w[R];
        for (int r = 0; r < R; ++r) w[r] = wp[r][k >> 4];
        const uint k4 = k >> 2;
        float4 xv[4];
        for (int j = 0; j < 4; ++j) {
            xv[j] = x4[k4 + j];
            if (NORM) xv[j] *= gemv_bf4(g2[k4 + j]);
        }
        for (int r = 0; r < R; ++r)
            acc[r] += dot(i8x4(w[r].x), xv[0]) + dot(i8x4(w[r].y), xv[1]) +
                      dot(i8x4(w[r].z), xv[2]) + dot(i8x4(w[r].w), xv[3]);
    }
    float mine = 0.0f;
    for (int r = 0; r < R; ++r) {
        const float s = simd_sum(acc[r]);
        if (lane == uint(r)) mine = s;
    }
    const uint row = row0 + lane;
    if (lane < uint(R) && row < p.rows) y[row] = mine * inv * S[row];
}

#define GEMV_I8_INST(NAME, R, N) \
template [[host_name(NAME)]] kernel void gemv_i8_f32<R, N>(device const char*, device const float*, device const float*, \
    device float*, device const bfloat*, constant GemvI8Params&, uint, uint, uint, uint, uint);
GEMV_I8_INST("gemv_i8_r1", 1, false)
GEMV_I8_INST("gemv_i8_r2", 2, false)
GEMV_I8_INST("gemv_i8_r4", 4, false)
GEMV_I8_INST("gemv_i8_r8", 8, false)
GEMV_I8_INST("gemv_i8_norm_r1", 1, true)
GEMV_I8_INST("gemv_i8_norm_r2", 2, true)
GEMV_I8_INST("gemv_i8_norm_r4", 4, true)
GEMV_I8_INST("gemv_i8_norm_r8", 8, true)

// ---------------------------------------------------------------------------
// Load-time quantization: bf16 [rows][K] -> int8 [rows][K] + fp32 scale [rows],
// symmetric per row, scale = absmax / 127, q = round(w / scale).
// One simdgroup per row; K % 4 == 0, W 8-byte aligned.
// ---------------------------------------------------------------------------
struct QuantI8Params { uint rows; uint K; };

kernel void quant_rows_i8(device const bfloat*  W   [[buffer(0)]],
                          device char*          Q   [[buffer(1)]],
                          device float*         S   [[buffer(2)]],
                          constant QuantI8Params& p [[buffer(3)]],
                          uint tg   [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sid  [[simdgroup_index_in_threadgroup]],
                          uint nsg  [[simdgroups_per_threadgroup]]) {
    const uint row = tg * nsg + sid;
    if (row >= p.rows) return;
    device const uint2* w2 = (device const uint2*)(W + (ulong)row * p.K);
    float m = 0.0f;
    for (uint i = lane; i < p.K / 4; i += 32) { const float4 v = gemv_bf4(w2[i]); m = max(m, max(max(fabs(v.x), fabs(v.y)), max(fabs(v.z), fabs(v.w)))); }
    m = simd_max(m);
    const float scale = m > 0.0f ? m / 127.0f : 1.0f;
    device char4* q4 = (device char4*)(Q + (ulong)row * p.K);
    for (uint i = lane; i < p.K / 4; i += 32) {
        const float4 v = gemv_bf4(w2[i]);
        q4[i] = char4(clamp(rint(v / scale), -127.0f, 127.0f));
    }
    if (lane == 0) S[row] = scale;
}
