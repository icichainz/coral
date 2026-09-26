#include "common.h"

// bf16-weight GEMV for the M=1 decode path, plus fp32 RMSNorm and argmax.
//
//   y[r] = (accumulate ? y[r] : 0) + inv * sum_k W[r,k] * (x[k] * g[k]) + b[r]
//
//   W : bf16 [rows][K]  row-major, bound zero-copy from the safetensors shard
//   x : fp32 [K]
//   g : bf16 [K]        optional RMSNorm weight (NORM variants); inv = rsqrt(mean(x^2)+eps),
//                       computed by every threadgroup from x (cheap: x is L2 resident)
//   b : bf16 [rows]     optional bias
//   y : fp32 [rows]
//
// Alignment: the gpt-oss safetensors shards place every tensor at 8 mod 16
// bytes, so weight / norm reads are 8-byte (uint2 = 4 x bf16) loads. Each lane
// reads 8 consecutive bf16 per step (two adjacent uint2), a simdgroup covers
// 256 elements = 512 contiguous bytes per row per step. x must be 16-byte
// aligned (float4 loads) and K a multiple of 8. The host side
// (src/model/attention_ops.cpp) checks all of this at bind time.
//
// One simdgroup computes R rows; 8 simdgroups (256 threads) per threadgroup.

#define GEMV_SG 8u

struct GemvParams { uint rows; uint K; uint flags; float eps; };
constant uint GEMV_BIAS  = 1u;
constant uint GEMV_ACCUM = 2u;

inline float4 gemv_bf4(uint2 u) {
    return float4(as_type<float>(u.x << 16), as_type<float>(u.x & 0xFFFF0000u),
                  as_type<float>(u.y << 16), as_type<float>(u.y & 0xFFFF0000u));
}

// inv RMS of an fp32 vector, computed by the whole threadgroup (all threads must call).
inline float gemv_inv_rms(device const float* x, uint K, float eps, threadgroup float* scratch,
                          uint tid, uint tgs, uint lane, uint sid) {
    device const float4* x4 = (device const float4*)x;
    float ss = 0.0f;
    for (uint i = tid; i < K / 4; i += tgs) { float4 v = x4[i]; ss += dot(v, v); }
    ss = tg_sum(ss, scratch, tid, tgs, lane, sid);
    return rsqrt(ss / float(K) + eps);
}

// acc[r] += sum over this lane's k of W_r[k] * x[k] (* g[k]); caller simd_sums.
template <int N, bool NORM>
inline void gemv_rows(thread float* acc, device const uint2* thread* wp, device const float* x,
                      device const bfloat* g, uint K, uint lane) {
    device const float4* x4 = (device const float4*)x;
    device const uint2*  g2 = (device const uint2*)g;
    for (uint k = lane * 8; k < K; k += 256) {
        const uint k4 = k >> 2;
        float4 xa = x4[k4], xb = x4[k4 + 1];
        if (NORM) { xa *= gemv_bf4(g2[k4]); xb *= gemv_bf4(g2[k4 + 1]); }
        uint2 wa[N], wb[N];
        for (int r = 0; r < N; ++r) { wa[r] = wp[r][k4]; wb[r] = wp[r][k4 + 1]; }
        for (int r = 0; r < N; ++r) acc[r] += dot(gemv_bf4(wa[r]), xa) + dot(gemv_bf4(wb[r]), xb);
    }
}

template <int R, bool NORM>
kernel void gemv_bf16_f32(device const bfloat*  W    [[buffer(0)]],
                          device const float*   x    [[buffer(1)]],
                          device float*         y    [[buffer(2)]],
                          device const bfloat*  bias [[buffer(3)]],
                          device const bfloat*  nw   [[buffer(4)]],
                          constant GemvParams&  p    [[buffer(5)]],
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

    device const uint2* wp[R];
    for (int r = 0; r < R; ++r) {
        const uint row = min(row0 + uint(r), p.rows - 1);
        wp[r] = (device const uint2*)(W + (ulong)row * p.K);
    }
    float acc[R];
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    gemv_rows<R, NORM>(acc, wp, x, nw, p.K, lane);

    float mine = 0.0f;
    for (int r = 0; r < R; ++r) {
        const float s = simd_sum(acc[r]);
        if (lane == uint(r)) mine = s;
    }
    const uint row = row0 + lane;
    if (lane < uint(R) && row < p.rows) {
        float v = mine * inv;
        if (p.flags & GEMV_BIAS) v += bf2f(bias[row]);
        if (p.flags & GEMV_ACCUM) v += y[row];
        y[row] = v;
    }
}

template [[host_name("gemv_bf16_r1")]] kernel void gemv_bf16_f32<1, false>(device const bfloat*, device const float*, device float*, device const bfloat*, device const bfloat*, constant GemvParams&, uint, uint, uint, uint, uint);
template [[host_name("gemv_bf16_r2")]] kernel void gemv_bf16_f32<2, false>(device const bfloat*, device const float*, device float*, device const bfloat*, device const bfloat*, constant GemvParams&, uint, uint, uint, uint, uint);
template [[host_name("gemv_bf16_r4")]] kernel void gemv_bf16_f32<4, false>(device const bfloat*, device const float*, device float*, device const bfloat*, device const bfloat*, constant GemvParams&, uint, uint, uint, uint, uint);
template [[host_name("gemv_bf16_r8")]] kernel void gemv_bf16_f32<8, false>(device const bfloat*, device const float*, device float*, device const bfloat*, device const bfloat*, constant GemvParams&, uint, uint, uint, uint, uint);
template [[host_name("gemv_bf16_norm_r1")]] kernel void gemv_bf16_f32<1, true>(device const bfloat*, device const float*, device float*, device const bfloat*, device const bfloat*, constant GemvParams&, uint, uint, uint, uint, uint);
template [[host_name("gemv_bf16_norm_r2")]] kernel void gemv_bf16_f32<2, true>(device const bfloat*, device const float*, device float*, device const bfloat*, device const bfloat*, constant GemvParams&, uint, uint, uint, uint, uint);
template [[host_name("gemv_bf16_norm_r4")]] kernel void gemv_bf16_f32<4, true>(device const bfloat*, device const float*, device float*, device const bfloat*, device const bfloat*, constant GemvParams&, uint, uint, uint, uint, uint);
template [[host_name("gemv_bf16_norm_r8")]] kernel void gemv_bf16_f32<8, true>(device const bfloat*, device const float*, device float*, device const bfloat*, device const bfloat*, constant GemvParams&, uint, uint, uint, uint, uint);

// ---------------------------------------------------------------------------
// RMSNorm, fp32 in / fp32 out, bf16 weight. One threadgroup per row.
//   y = x * rsqrt(mean(x^2) + eps) * w
// n must be a multiple of 4; x, y 16-byte aligned.
// ---------------------------------------------------------------------------
struct GemvRmsParams { uint n; float eps; };

kernel void rmsnorm_f32(device const float*     x   [[buffer(0)]],
                        device const bfloat*    w   [[buffer(1)]],
                        device float*           y   [[buffer(2)]],
                        constant GemvRmsParams& p   [[buffer(3)]],
                        uint row  [[threadgroup_position_in_grid]],
                        uint tid  [[thread_position_in_threadgroup]],
                        uint tgs  [[threads_per_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sid  [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[32];
    device const float* xr = x + (ulong)row * p.n;
    device float*       yr = y + (ulong)row * p.n;
    const float inv = gemv_inv_rms(xr, p.n, p.eps, scratch, tid, tgs, lane, sid);
    for (uint i = tid; i < p.n; i += tgs) yr[i] = xr[i] * inv * bf2f(w[i]);
}

// ---------------------------------------------------------------------------
// Argmax over fp32 logits -> int32, two passes. Ties resolve to the lowest
// index (torch.argmax / numpy semantics). NaNs are never selected.
//   pass 1: ARGMAX_PARTIALS threadgroups, each reduces a strided slice
//   pass 2: one threadgroup reduces the partials
// ---------------------------------------------------------------------------
struct ArgmaxParams { uint n; };

inline void gemv_argmax_pick(thread float& bv, thread uint& bi, float v, uint i) {
    if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
}

// Threadgroup-wide (value, index) max. Result valid in all threads.
inline void gemv_argmax_tg(thread float& bv, thread uint& bi, threadgroup float* sv, threadgroup uint* si,
                           uint tid, uint tgs, uint lane, uint sid) {
    float m = simd_max(bv);
    uint  i = simd_min(bv == m ? bi : 0xFFFFFFFFu);
    if (lane == 0) { sv[sid] = m; si[sid] = i; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sid == 0) {
        const uint nsg = (tgs + 31) / 32;
        float v = lane < nsg ? sv[lane] : -INFINITY;
        uint  j = lane < nsg ? si[lane] : 0xFFFFFFFFu;
        float m2 = simd_max(v);
        uint  j2 = simd_min(v == m2 ? j : 0xFFFFFFFFu);
        if (lane == 0) { sv[0] = m2; si[0] = j2; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bv = sv[0]; bi = si[0];
}

kernel void argmax_f32_partial(device const float*    x   [[buffer(0)]],
                               device float*          pv  [[buffer(1)]],
                               device uint*           pi  [[buffer(2)]],
                               constant ArgmaxParams& p   [[buffer(3)]],
                               uint tg   [[threadgroup_position_in_grid]],
                               uint ntg  [[threadgroups_per_grid]],
                               uint tid  [[thread_position_in_threadgroup]],
                               uint tgs  [[threads_per_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint sid  [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sv[32];
    threadgroup uint  si[32];
    float bv = -INFINITY; uint bi = 0xFFFFFFFFu;
    for (uint i = tg * tgs + tid; i < p.n; i += ntg * tgs) gemv_argmax_pick(bv, bi, x[i], i);
    gemv_argmax_tg(bv, bi, sv, si, tid, tgs, lane, sid);
    if (tid == 0) { pv[tg] = bv; pi[tg] = bi; }
}

kernel void argmax_f32_final(device const float*    pv  [[buffer(0)]],
                             device const uint*     pi  [[buffer(1)]],
                             device int*            out [[buffer(2)]],
                             constant ArgmaxParams& p   [[buffer(3)]],   // n = number of partials
                             uint tid  [[thread_position_in_threadgroup]],
                             uint tgs  [[threads_per_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint sid  [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sv[32];
    threadgroup uint  si[32];
    float bv = -INFINITY; uint bi = 0xFFFFFFFFu;
    for (uint i = tid; i < p.n; i += tgs) gemv_argmax_pick(bv, bi, pv[i], pi[i]);
    gemv_argmax_tg(bv, bi, sv, si, tid, tgs, lane, sid);
    if (tid == 0) out[0] = int(bi == 0xFFFFFFFFu ? 0u : bi);
}
