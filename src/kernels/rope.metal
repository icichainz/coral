#include "common.h"

// RoPE (YaRN) for gpt-oss, and the fused decode kernel
//   rmsnorm -> QKV GEMV (+bias) -> RoPE(q,k) -> write q fp32, k/v bf16 into the KV cache.
//
// RoPE convention (HF modeling_gpt_oss._apply_rotary_emb): the head vector is
// split into halves; for j in [0, head_dim/2):
//   out[j]        = x[j] * cos[j] - x[j+half] * sin[j]
//   out[j+half]   = x[j+half] * cos[j] + x[j] * sin[j]
// with cos/sin = cos/sin(pos * inv_freq[j]) * attention_factor (YaRN mscale).
//
// The cos/sin table is precomputed on the CPU in fp32 exactly like HF
// (inv_freq in fp32, freqs = fp32(inv_freq) * fp32(pos), then cos/sin, then *
// attention_factor) and stored as float2(cos, sin) [max_positions][head_dim/2].
// The host binds the table at the row of the current position, so kernels see
// a float2[half] for that position only.
//
// Relies on gemv_bf16.metal (gemv_rows, gemv_inv_rms, gemv_bf4), which sorts
// before this file in the concatenated translation unit.

// Rotate in place: x fp32 [n_heads][64], rows of the table for this position.
struct RopeParams { uint n_heads; };

kernel void rope_yarn_f32(device float*          x   [[buffer(0)]],
                          device const float2*   cs  [[buffer(1)]],
                          constant RopeParams&   p   [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    const uint half_dim = 32;
    const uint h = gid / half_dim, j = gid % half_dim;
    if (h >= p.n_heads) return;
    device float* xh = x + h * 64;
    const float2 c = cs[j];
    const float a = xh[j], b = xh[j + half_dim];
    xh[j]            = a * c.x - b * c.y;
    xh[j + half_dim] = b * c.x + a * c.y;
}

// ---------------------------------------------------------------------------
// Fused decode QKV. Rows of the virtual [Q+2KV, H] matrix are processed in
// RoPE pairs (j, j+32) inside each 64-row head block; one simdgroup handles
// ATTN_QKV_P consecutive pairs (2*P rows), 8 simdgroups per threadgroup.
// Head blocks never straddle q/k/v since Q and KV are multiples of 64.
//
//   x      fp32 [H]   residual stream (normalized here with norm_w, eps)
//   q_out  fp32 [Q]   roped q
//   k_row  bf16 [KV]  cache row for this position (host binds the slot)
//   v_row  bf16 [KV]
// ---------------------------------------------------------------------------
#define ATTN_QKV_P 2

struct QkvParams { uint H; uint Q; uint KV; float eps; };

kernel void attn_qkv_rope(device const float*  x      [[buffer(0)]],
                          device const bfloat* norm_w [[buffer(1)]],
                          device const bfloat* wq     [[buffer(2)]],
                          device const bfloat* bq     [[buffer(3)]],
                          device const bfloat* wk     [[buffer(4)]],
                          device const bfloat* bk     [[buffer(5)]],
                          device const bfloat* wv     [[buffer(6)]],
                          device const bfloat* bv     [[buffer(7)]],
                          device float*        q_out  [[buffer(8)]],
                          device bfloat*       k_row  [[buffer(9)]],
                          device bfloat*       v_row  [[buffer(10)]],
                          device const float2* cs     [[buffer(11)]],
                          constant QkvParams&  p      [[buffer(12)]],
                          uint tg   [[threadgroup_position_in_grid]],
                          uint tid  [[thread_position_in_threadgroup]],
                          uint tgs  [[threads_per_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sid  [[simdgroup_index_in_threadgroup]]) {
    constexpr int P = ATTN_QKV_P;
    threadgroup float scratch[32];
    const float inv = gemv_inv_rms(x, p.H, p.eps, scratch, tid, tgs, lane, sid);

    const uint gp0 = (tg * GEMV_SG + sid) * P;          // first pair of this simdgroup
    const uint total_pairs = (p.Q + 2 * p.KV) / 2;
    if (gp0 >= total_pairs) return;
    const uint j0 = gp0 % 32;                            // pair index inside the head block
    uint rb = (gp0 / 32) * 64 + j0;                      // row in the virtual [Q+2KV] matrix

    device const bfloat* W; device const bfloat* B; uint kind;
    if (rb < p.Q)               { W = wq; B = bq; kind = 0; }
    else if (rb < p.Q + p.KV)   { rb -= p.Q; W = wk; B = bk; kind = 1; }
    else                        { rb -= p.Q + p.KV; W = wv; B = bv; kind = 2; }

    device const uint2* wp[2 * P];
    for (int i = 0; i < P; ++i) {
        wp[i]     = (device const uint2*)(W + (ulong)(rb + i) * p.H);
        wp[P + i] = (device const uint2*)(W + (ulong)(rb + 32 + i) * p.H);
    }
    float acc[2 * P];
    for (int i = 0; i < 2 * P; ++i) acc[i] = 0.0f;
    gemv_rows<2 * P, true>(acc, wp, x, norm_w, p.H, lane);

    float a = 0.0f, b = 0.0f;
    for (int i = 0; i < P; ++i) {
        const float s0 = simd_sum(acc[i]);
        const float s1 = simd_sum(acc[P + i]);
        if (lane == uint(i)) { a = s0; b = s1; }
    }
    if (lane >= uint(P)) return;
    const uint r0 = rb + lane, r1 = rb + 32 + lane;
    a = a * inv + bf2f(B[r0]);
    b = b * inv + bf2f(B[r1]);
    if (kind != 2) {
        const float2 c = cs[j0 + lane];
        const float ra = a * c.x - b * c.y;
        const float rbv = b * c.x + a * c.y;
        a = ra; b = rbv;
    }
    if (kind == 0)      { q_out[r0] = a; q_out[r1] = b; }
    else if (kind == 1) { k_row[r0] = f2bf(a); k_row[r1] = f2bf(b); }
    else                { v_row[r0] = f2bf(a); v_row[r1] = f2bf(b); }
}
