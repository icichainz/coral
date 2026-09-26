#include "common.h"

// MoE decode (M = 1) for gpt-oss: 3 dispatches per layer.
//
//   1. moe_router            rmsnorm(residual) * mlp_norm -> normed; router logits;
//                            the last threadgroup: top-k + softmax -> ids, probs
//   2. mx_gemv_gate_up_swiglu h[k] = swiglu(W_gu[ids[k]]·normed + b_gu)
//   3. mx_gemv_down          residual += sum_k probs[k] * (W_d[ids[k]]·h[k] + b_d)
//
// Relies on the mx_* MXFP4 helpers from gemv_mxfp4.metal, which precedes this
// file in the (sorted) concatenated translation unit.
//
// Semantics follow HF transformers GptOssTopKRouter / GptOssExperts:
//   logits = W_r·x + b_r;  (vals, ids) = topk(logits, K);  probs = softmax(vals)
//   gu = W_gu[e]·x + b_gu[e];  gate = gu[0::2], up = gu[1::2]
//   gate = min(gate, limit); up = clamp(up, -limit, limit)
//   h = (up + 1) * gate * sigmoid(alpha * gate)
//   out = sum_k probs[k] * (W_d[e_k]·h_k + b_d[e_k])
// Top-k ties are broken toward the lowest expert index; ids are ordered by
// descending logit.

// Top-K over E <= 256 logits within one simdgroup (all lanes call it).
// Lane k (< K) receives the k-th selected expert and its softmax weight.
inline void moe_select_topk(device const float* logits, uint E, uint K, uint lane,
                            thread int& my_id, thread float& my_prob) {
    const uint per_lane = (E + 31) / 32;
    uint taken = 0;
    float top0 = 0.0f, denom = 0.0f, my_val = 0.0f;
    my_id = 0;
    for (uint k = 0; k < K; ++k) {
        float best = -INFINITY;
        uint best_e = 0xFFFFFFFFu;
        for (uint t = 0; t < per_lane; ++t) {
            const uint e = lane + 32 * t;
            if (e < E && !(taken & (1u << t))) {
                const float v = logits[e];
                if (best_e == 0xFFFFFFFFu || v > best) { best = v; best_e = e; }
            }
        }
        const float m = simd_max(best_e == 0xFFFFFFFFu ? -INFINITY : best);
        const uint win = simd_min((best_e != 0xFFFFFFFFu && best == m) ? best_e : 0xFFFFFFFFu);
        if (win != 0xFFFFFFFFu && win % 32 == lane) taken |= 1u << (win / 32);
        if (k == 0) top0 = m;
        denom += exp(m - top0);
        if (lane == k) { my_val = m; my_id = int(win); }
    }
    my_prob = exp(my_val - top0) / denom;
}

// ---------------------------------------------------------------------------
// 1. RMSNorm + router logits. One threadgroup per expert row; every
// threadgroup recomputes the (cheap, L2-resident) norm so no second pass or
// grid-wide sync is needed. Thread t owns float4 chunks t, t+T, ... for both
// the norm and the dot product, so normed values never leave registers.
//   residual : fp32 [H]    norm_w : bf16 [H]    router_w : bf16 [E][H]   router_b : bf16 [E]
//   normed   : fp32 [H]    (written by threadgroup 0)
//   logits   : fp32 [E]
//   ids, probs : int32/fp32 [K]   top-k (ties: lowest index) and softmax over it
//   counter  : uint, zero between dispatches
// Grid: E threadgroups x MOE_ROUTER_THREADS. H % 4 == 0, H/4 <= 4*MOE_ROUTER_THREADS.
// ---------------------------------------------------------------------------
struct MoeRouterParams { uint H; uint E; float eps; uint do_norm; uint K; };

constant constexpr uint MOE_ROUTER_THREADS = 256;
constant constexpr uint MOE_ROUTER_CHUNKS = 4;   // float4 chunks per thread (H <= 4096)

kernel void moe_router(device const float*       residual [[buffer(0)]],
                       device const bfloat*      norm_w   [[buffer(1)]],
                       device const bfloat*      router_w [[buffer(2)]],
                       device const bfloat*      router_b [[buffer(3)]],
                       device float*             normed   [[buffer(4)]],
                       device float*             logits   [[buffer(5)]],
                       constant MoeRouterParams& p        [[buffer(6)]],
                       device int*               ids      [[buffer(7)]],
                       device float*             probs    [[buffer(8)]],
                       device atomic_uint*       counter  [[buffer(9)]],
                       uint e       [[threadgroup_position_in_grid]],
                       uint tid     [[thread_position_in_threadgroup]],
                       uint tg_size [[threads_per_threadgroup]],
                       uint lane    [[thread_index_in_simdgroup]],
                       uint sg      [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[32];
    threadgroup uint last;
    const uint n4 = p.H / 4;
    device const float4* r4 = (device const float4*)residual;
    device const bfloat4* w4 = (device const bfloat4*)(router_w + (ulong)e * p.H);

    // Issue every load (residual, router row, norm weight) before the first
    // reduction so the two memory round trips overlap.
    float4 xv[MOE_ROUTER_CHUNKS], wv[MOE_ROUTER_CHUNKS], nv[MOE_ROUTER_CHUNKS];
    device const bfloat4* n4w = (device const bfloat4*)norm_w;
    for (uint c = 0; c < MOE_ROUTER_CHUNKS; ++c) {
        const uint j = tid + c * tg_size;
        const bool ok = j < n4;
        xv[c] = ok ? r4[j] : float4(0.0f);
        wv[c] = ok ? float4(w4[j]) : float4(0.0f);
        nv[c] = (ok && p.do_norm) ? float4(n4w[j]) : float4(1.0f);
    }
    float ss = 0.0f;
    for (uint c = 0; c < MOE_ROUTER_CHUNKS; ++c) ss += dot(xv[c], xv[c]);
    float inv = 1.0f;
    if (p.do_norm) {
        ss = tg_sum(ss, scratch, tid, tg_size, lane, sg);
        inv = rsqrt(ss / float(p.H) + p.eps);
    }
    float acc = 0.0f;
    for (uint c = 0; c < MOE_ROUTER_CHUNKS; ++c) {
        const uint j = tid + c * tg_size;
        const float4 xn = xv[c] * inv * nv[c];
        acc += dot(wv[c], xn);
        if (e == 0 && j < n4) ((device float4*)normed)[j] = xn;
    }
    acc = tg_sum(acc, scratch, tid, tg_size, lane, sg);
    if (tid == 0) logits[e] = acc + bf2f(router_b[e]);

    // The last threadgroup to finish selects the top-k (device-scope counter,
    // seq_cst fences around it) and resets the counter for the next layer.
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (tid == 0) last = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed) == p.E - 1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!last) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    if (sg == 0) {
        int my_id; float my_prob;
        moe_select_topk(logits, p.E, p.K, lane, my_id, my_prob);
        if (lane < p.K) { ids[lane] = my_id; probs[lane] = my_prob; }
        if (lane == 0) atomic_store_explicit(counter, 0u, memory_order_relaxed);
    }
}


// ---------------------------------------------------------------------------
// MXFP4 GEMV lane mapping ("quarter blocks"): 4 lanes share one 32-element
// block, each lane owns 4 bytes (8 elements) of it. A simdgroup covers 8
// consecutive blocks = 128 contiguous bytes per row per step, and the
// matching activation loads are 8 contiguous floats (two float4) per lane.
// Measured on M2 Max against one-lane-per-block (16-byte loads, 128 bytes of
// activations per lane per block) and two-lanes-per-block mappings, with
// 1-8 rows per simdgroup, 1-16 simdgroups per threadgroup, experts split
// across simdgroups, all-experts-per-simdgroup and 2-4x unrolled loads
// (tests/test_moe.cpp moe_lab_sweep history): this mapping with the shapes in
// src/model/moe_ops.cpp is the fastest (gate_up 128 -> 107 us, down 104 ->
// 62 us per layer; gate_up reads 35.3 MB, down 17.6 MB).
// Word w of a block holds elements 8w..8w+7, element k at bits 4k..4k+3.
// Weight words are 4-byte loads: tensor offsets are 8 mod 16 in the
// checkpoint, which this mapping never cares about.
// ---------------------------------------------------------------------------
inline float mx_quarter_dot(uint w, float4 xa, float4 xb) {
    const float2 v0 = float2(as_type<half2>(mx_half2_bits(w)));        // e0, e4
    const float2 v1 = float2(as_type<half2>(mx_half2_bits(w >> 4)));   // e1, e5
    const float2 v2 = float2(as_type<half2>(mx_half2_bits(w >> 8)));   // e2, e6
    const float2 v3 = float2(as_type<half2>(mx_half2_bits(w >> 12)));  // e3, e7
    return dot(float4(v0.x, v1.x, v2.x, v3.x), xa) + dot(float4(v0.y, v1.y, v2.y, v3.y), xb);
}


// ---------------------------------------------------------------------------
// 2. gate_up GEMV + clamped SwiGLU for the K selected experts.
//   x      : fp32 [H]      normed activations
//   ids    : int32 [K]     selected experts (from moe_router)
//   h      : fp32 [K][I]   output
// (logits, probs and p.E / p.forced are unused; kept for a stable binding layout.)
// Grid: (ceil(I / (NSG*PAIRS)), K) threadgroups of NSG simdgroups. Each
// simdgroup computes PAIRS (gate, up) row pairs = 2*PAIRS adjacent rows
// sharing one activation load per step.
// ---------------------------------------------------------------------------
struct MoeGateUpParams { uint H; uint I; uint E; uint K; float limit; float alpha; uint forced; };

template <uint PAIRS>
kernel void mx_gemv_gate_up_swiglu_t(device const float*       x      [[buffer(0)]],
                                        device const uchar*       blocks [[buffer(1)]],
                                        device const uchar*       scales [[buffer(2)]],
                                        device const bfloat*      bias   [[buffer(3)]],
                                        device const float*       logits [[buffer(4)]],
                                        device int*               ids    [[buffer(5)]],
                                        device float*             probs  [[buffer(6)]],
                                        device float*             h      [[buffer(7)]],
                                        constant MoeGateUpParams& p      [[buffer(8)]],
                                        uint2 tg   [[threadgroup_position_in_grid]],
                                        uint  sg   [[simdgroup_index_in_threadgroup]],
                                        uint  nsg  [[simdgroups_per_threadgroup]],
                                        uint  lane [[thread_index_in_simdgroup]]) {
    constexpr uint R = 2 * PAIRS;
    const uint slot = tg.y;
    const uint e = uint(ids[slot]);

    const uint nblk = p.H / 32;
    const uint i0 = (tg.x * nsg + sg) * PAIRS;
    if (i0 >= p.I) return;
    const uint npairs = min(PAIRS, p.I - i0);
    const uint nrows = 2 * npairs;
    const ulong row0 = (ulong)e * (2 * p.I) + 2 * i0;
    device const uint* bp = (device const uint*)(blocks + row0 * nblk * 16);
    device const uchar* sp = scales + row0 * nblk;
    const uint rstride4 = nblk * 4;   // uints per row
    const uint q = lane & 3, bl = lane >> 2;

    float acc[R];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;
    for (uint b = bl; b < nblk; b += 8) {
        uint w[R];
        float sc[R];
        for (uint r = 0; r < R; ++r) {
            const uint rr = min(r, nrows - 1);
            w[r] = bp[rr * rstride4 + b * 4 + q];
            sc[r] = e8m0_to_float(sp[rr * nblk + b]);
        }
        device const float4* x4 = (device const float4*)(x + b * 32 + q * 8);
        const float4 xa = x4[0], xb = x4[1];
        for (uint r = 0; r < R; ++r) acc[r] += mx_quarter_dot(w[r], xa, xb) * sc[r];
    }
    for (uint r = 0; r < R; ++r) acc[r] = simd_sum(acc[r]);

    if (lane < npairs) {
        float g = 0.0f, u = 0.0f;
        for (uint j = 0; j < PAIRS; ++j)
            if (j == lane) { g = acc[2 * j]; u = acc[2 * j + 1]; }
        const uint i = i0 + lane;
        device const bfloat* bb = bias + (ulong)e * (2 * p.I) + 2 * i;
        g = g * kMxUnscale + bf2f(bb[0]);
        u = u * kMxUnscale + bf2f(bb[1]);
        g = min(g, p.limit);
        u = clamp(u, -p.limit, p.limit);
        const float glu = g / (1.0f + exp(-p.alpha * g));
        h[(ulong)slot * p.I + i] = (u + 1.0f) * glu;
    }
}

#define MX_GU_INST(NAME, P) \
template [[host_name(NAME)]] kernel void mx_gemv_gate_up_swiglu_t<P>( \
    device const float*, device const uchar*, device const uchar*, device const bfloat*, device const float*, \
    device int*, device float*, device float*, constant MoeGateUpParams&, uint2, uint, uint, uint);
MX_GU_INST("mx_gemv_gate_up_swiglu_p1", 1)
MX_GU_INST("mx_gemv_gate_up_swiglu_p2", 2)
MX_GU_INST("mx_gemv_gate_up_swiglu_p4", 4)

// ---------------------------------------------------------------------------
// 3. down GEMV fused with the routing-weighted sum and the residual add:
//   residual[o] += sum_k probs[k] * (W_d[ids[k]][o] · h[k] + b_d[ids[k]][o])
// One simdgroup owns R output rows across all K experts, so the weighted sum
// needs no atomics and no extra pass. Grid: ceil(H / (NSG*R)) threadgroups.
// ---------------------------------------------------------------------------
struct MoeDownParams { uint H; uint I; uint K; };

template <uint R>
kernel void mx_gemv_down_t(device const float*     h        [[buffer(0)]],
                              device const uchar*     blocks   [[buffer(1)]],
                              device const uchar*     scales   [[buffer(2)]],
                              device const bfloat*    bias     [[buffer(3)]],
                              device const int*       ids      [[buffer(4)]],
                              device const float*     probs    [[buffer(5)]],
                              device float*           residual [[buffer(6)]],
                              constant MoeDownParams& p        [[buffer(7)]],
                              uint tg   [[threadgroup_position_in_grid]],
                              uint sg   [[simdgroup_index_in_threadgroup]],
                              uint nsg  [[simdgroups_per_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    const uint o0 = (tg * nsg + sg) * R;
    if (o0 >= p.H) return;
    const uint nrows = min(R, p.H - o0);
    const uint nblk = p.I / 32;
    const uint rstride4 = nblk * 4;
    const uint q = lane & 3, bl = lane >> 2;

    float acc[R];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;
    for (uint s = 0; s < p.K; ++s) {
        const uint e = uint(ids[s]);
        const float pr = probs[s];
        const ulong row0 = (ulong)e * p.H + o0;
        device const uint* bp = (device const uint*)(blocks + row0 * nblk * 16);
        device const uchar* sp = scales + row0 * nblk;
        device const float* hs = h + (ulong)s * p.I;
        for (uint b = bl; b < nblk; b += 8) {
            uint w[R];
            float sc[R];
            for (uint r = 0; r < R; ++r) {
                const uint rr = min(r, nrows - 1);
                w[r] = bp[rr * rstride4 + b * 4 + q];
                sc[r] = e8m0_to_float(sp[rr * nblk + b]) * pr;
            }
            device const float4* x4 = (device const float4*)(hs + b * 32 + q * 8);
            const float4 xa = x4[0], xb = x4[1];
            for (uint r = 0; r < R; ++r) acc[r] += mx_quarter_dot(w[r], xa, xb) * sc[r];
        }
    }
    for (uint r = 0; r < R; ++r) acc[r] = simd_sum(acc[r]);

    if (lane < nrows) {
        float v = 0.0f;
        for (uint r = 0; r < R; ++r) if (r == lane) v = acc[r];
        const uint o = o0 + lane;
        float bsum = 0.0f;
        for (uint s = 0; s < p.K; ++s) bsum += probs[s] * bf2f(bias[(ulong)uint(ids[s]) * p.H + o]);
        residual[o] += v * kMxUnscale + bsum;
    }
}

#define MX_DN_INST(NAME, R) \
template [[host_name(NAME)]] kernel void mx_gemv_down_t<R>( \
    device const float*, device const uchar*, device const uchar*, device const bfloat*, device const int*, \
    device const float*, device float*, constant MoeDownParams&, uint, uint, uint, uint);
MX_DN_INST("mx_gemv_down_r1", 1)
MX_DN_INST("mx_gemv_down_r2", 2)
MX_DN_INST("mx_gemv_down_r4", 4)
