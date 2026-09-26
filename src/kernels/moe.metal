#include "common.h"

// MoE decode (M = 1) for gpt-oss: 3 dispatches per layer.
//
//   1. moe_router            rmsnorm(residual) * mlp_norm -> normed; router logits
//   2. mx_gemv_gate_up_swiglu top-k + softmax (per simdgroup, from the logits),
//                            h[k] = swiglu(W_gu[ids[k]]·normed + b_gu)
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

// ---------------------------------------------------------------------------
// 1. RMSNorm + router logits. One threadgroup per expert row; every
// threadgroup recomputes the (cheap, L2-resident) norm so no second pass or
// grid-wide sync is needed. Thread t owns float4 chunks t, t+T, ... for both
// the norm and the dot product, so normed values never leave registers.
//   residual : fp32 [H]    norm_w : bf16 [H]    router_w : bf16 [E][H]   router_b : bf16 [E]
//   normed   : fp32 [H]    (written by threadgroup 0)
//   logits   : fp32 [E]
// Grid: E threadgroups x MOE_ROUTER_THREADS. H % 4 == 0, H/4 <= 4*MOE_ROUTER_THREADS.
// ---------------------------------------------------------------------------
struct MoeRouterParams { uint H; uint E; float eps; uint do_norm; };

constant constexpr uint MOE_ROUTER_THREADS = 256;
constant constexpr uint MOE_ROUTER_CHUNKS = 4;   // float4 chunks per thread (H <= 4096)

kernel void moe_router(device const float*       residual [[buffer(0)]],
                       device const bfloat*      norm_w   [[buffer(1)]],
                       device const bfloat*      router_w [[buffer(2)]],
                       device const bfloat*      router_b [[buffer(3)]],
                       device float*             normed   [[buffer(4)]],
                       device float*             logits   [[buffer(5)]],
                       constant MoeRouterParams& p        [[buffer(6)]],
                       uint e       [[threadgroup_position_in_grid]],
                       uint tid     [[thread_position_in_threadgroup]],
                       uint tg_size [[threads_per_threadgroup]],
                       uint lane    [[thread_index_in_simdgroup]],
                       uint sg      [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[32];
    const uint n4 = p.H / 4;
    device const float4* r4 = (device const float4*)residual;
    device const bfloat4* w4 = (device const bfloat4*)(router_w + (ulong)e * p.H);

    float4 xv[MOE_ROUTER_CHUNKS];
    float ss = 0.0f;
    for (uint c = 0; c < MOE_ROUTER_CHUNKS; ++c) {
        const uint j = tid + c * tg_size;
        xv[c] = (j < n4) ? r4[j] : float4(0.0f);
        ss += dot(xv[c], xv[c]);
    }
    if (p.do_norm) {
        ss = tg_sum(ss, scratch, tid, tg_size, lane, sg);
        const float inv = rsqrt(ss / float(p.H) + p.eps);
        device const bfloat4* n4w = (device const bfloat4*)norm_w;
        for (uint c = 0; c < MOE_ROUTER_CHUNKS; ++c) {
            const uint j = tid + c * tg_size;
            if (j < n4) xv[c] = xv[c] * inv * float4(n4w[j]);
        }
    }
    float acc = 0.0f;
    for (uint c = 0; c < MOE_ROUTER_CHUNKS; ++c) {
        const uint j = tid + c * tg_size;
        if (j < n4) {
            acc += dot(float4(w4[j]), xv[c]);
            if (e == 0) ((device float4*)normed)[j] = xv[c];
        }
    }
    acc = tg_sum(acc, scratch, tid, tg_size, lane, sg);
    if (tid == 0) logits[e] = acc + bf2f(router_b[e]);
}

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
// 2. Top-k + gate_up GEMV + clamped SwiGLU for the K selected experts.
//   x      : fp32 [H]      normed activations
//   logits : fp32 [E]      router logits (ignored when p.forced != 0)
//   ids    : int32 [K]     written by threadgroup (0,0) (read when p.forced != 0)
//   probs  : fp32 [K]      written by threadgroup (0,0)
//   h      : fp32 [K][I]   output
// Grid: (ceil(I / (NSG*PAIRS)), K) threadgroups of NSG simdgroups.
// Each simdgroup computes PAIRS (gate, up) row pairs = 2*PAIRS adjacent
// rows sharing one x load per block; lanes own whole 32-element blocks.
// ---------------------------------------------------------------------------
struct MoeGateUpParams { uint H; uint I; uint E; uint K; float limit; float alpha; uint forced; };

template <bool A16, uint PAIRS>
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
    uint e;
    if (p.forced) {
        e = uint(ids[slot]);
    } else {
        int my_id; float my_prob;
        moe_select_topk(logits, p.E, p.K, lane, my_id, my_prob);
        e = uint(simd_shuffle(my_id, ushort(slot)));
        if (tg.x == 0 && slot == 0 && sg == 0 && lane < p.K) { ids[lane] = my_id; probs[lane] = my_prob; }
    }

    const uint nblk = p.H / 32;
    const uint i0 = (tg.x * nsg + sg) * PAIRS;   // first output (pair) index
    if (i0 >= p.I) return;
    const uint npairs = min(PAIRS, p.I - i0);
    const uint nrows = 2 * npairs;

    const ulong row0 = (ulong)e * (2 * p.I) + 2 * i0;
    device const uchar* bp = blocks + row0 * nblk * 16;
    device const uchar* sp = scales + row0 * nblk;
    const ulong rstride = (ulong)nblk * 16;

    float acc[R];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;

    for (uint b = lane; b < nblk; b += 32) {
        uint4 q[R];
        float sc[R];
        for (uint r = 0; r < R; ++r) {
            const uint rr = min(r, nrows - 1);
            q[r] = mx_load_block<A16>(bp + rr * rstride + b * 16);
            sc[r] = e8m0_to_float(sp[rr * nblk + b]);
        }
        float4 xr[8];
        mx_load_x(x + b * 32, xr);
        for (uint r = 0; r < R; ++r) acc[r] += mx_block_dot(q[r], xr) * sc[r];
    }
    for (uint r = 0; r < R; ++r) acc[r] = simd_sum(acc[r]);

    if (lane < npairs) {   // lane j finalizes pair j
        float g = 0.0f, u = 0.0f;
        for (uint j = 0; j < PAIRS; ++j)
            if (j == lane) { g = acc[2 * j]; u = acc[2 * j + 1]; }
        const uint i = i0 + lane;
        device const bfloat* bb = bias + (ulong)e * (2 * p.I) + 2 * i;
        g = g * kMxUnscale + bf2f(bb[0]);
        u = u * kMxUnscale + bf2f(bb[1]);
        g = min(g, p.limit);
        u = clamp(u, -p.limit, p.limit);
        const float glu = g / (1.0f + exp(-p.alpha * g));   // g * sigmoid(alpha * g)
        h[(ulong)slot * p.I + i] = (u + 1.0f) * glu;
    }
}

#define MX_GU_INST(NAME, A16, P) \
template [[host_name(NAME)]] kernel void mx_gemv_gate_up_swiglu_t<A16, P>( \
    device const float*, device const uchar*, device const uchar*, device const bfloat*, device const float*, \
    device int*, device float*, device float*, constant MoeGateUpParams&, uint2, uint, uint, uint);
MX_GU_INST("mx_gemv_gate_up_swiglu_p1", true, 1)
MX_GU_INST("mx_gemv_gate_up_swiglu_p2", true, 2)
MX_GU_INST("mx_gemv_gate_up_swiglu_p4", true, 4)
MX_GU_INST("mx_gemv_gate_up_swiglu_p1_a8", false, 1)
MX_GU_INST("mx_gemv_gate_up_swiglu_p2_a8", false, 2)
MX_GU_INST("mx_gemv_gate_up_swiglu_p4_a8", false, 4)

// ---------------------------------------------------------------------------
// 3. down GEMV fused with the routing-weighted sum and the residual add:
//   residual[o] += sum_k probs[k] * (W_d[ids[k]][o] · h[k] + b_d[ids[k]][o])
// One simdgroup owns R output rows across all K experts, so the weighted sum
// needs no atomics and no extra pass over d. The (expert, block) work items
// of a row are flattened so each lane has ~K*nblk/32 independent block loads.
// Grid: ceil(H / (NSG*R)) threadgroups of NSG simdgroups.
// ---------------------------------------------------------------------------
struct MoeDownParams { uint H; uint I; uint K; };

template <bool A16, uint R>
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
    const ulong rstride = (ulong)nblk * 16;
    const uint nitems = p.K * nblk;

    float acc[R];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;

    for (uint it = lane; it < nitems; it += 32) {
        const uint s = it / nblk, b = it - s * nblk;
        const uint e = uint(ids[s]);
        const float pr = probs[s];
        const ulong row0 = (ulong)e * p.H + o0;
        uint4 q[R];
        float sc[R];
        for (uint r = 0; r < R; ++r) {
            const ulong rr = row0 + min(r, nrows - 1);
            q[r] = mx_load_block<A16>(blocks + rr * rstride + b * 16);
            sc[r] = e8m0_to_float(scales[rr * nblk + b]) * pr;
        }
        float4 xr[8];
        mx_load_x(h + (ulong)s * p.I + b * 32, xr);
        for (uint r = 0; r < R; ++r) acc[r] += mx_block_dot(q[r], xr) * sc[r];
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

#define MX_DN_INST(NAME, A16, R) \
template [[host_name(NAME)]] kernel void mx_gemv_down_t<A16, R>( \
    device const float*, device const uchar*, device const uchar*, device const bfloat*, device const int*, \
    device const float*, device float*, constant MoeDownParams&, uint, uint, uint, uint);
MX_DN_INST("mx_gemv_down_r1", true, 1)
MX_DN_INST("mx_gemv_down_r2", true, 2)
MX_DN_INST("mx_gemv_down_r4", true, 4)
MX_DN_INST("mx_gemv_down_r1_a8", false, 1)
MX_DN_INST("mx_gemv_down_r2_a8", false, 2)
MX_DN_INST("mx_gemv_down_r4_a8", false, 4)

// ---------------------------------------------------------------------------
// "Quarter-block" lane mapping: 4 lanes share one 32-element block, each lane
// owns 4 bytes (8 elements) of it. A simdgroup covers 8 consecutive blocks =
// 128 contiguous bytes per row per step, and the matching activation loads
// are 8 contiguous floats per lane (much better coalesced than one lane
// reading a whole 128-byte block of x).
// ---------------------------------------------------------------------------
inline float mx_quarter_dot(uint w, float4 xa, float4 xb) {
    const float2 v0 = float2(as_type<half2>(mx_half2_bits(w)));        // e0, e4
    const float2 v1 = float2(as_type<half2>(mx_half2_bits(w >> 4)));   // e1, e5
    const float2 v2 = float2(as_type<half2>(mx_half2_bits(w >> 8)));   // e2, e6
    const float2 v3 = float2(as_type<half2>(mx_half2_bits(w >> 12)));  // e3, e7
    return dot(float4(v0.x, v1.x, v2.x, v3.x), xa) + dot(float4(v0.y, v1.y, v2.y, v3.y), xb);
}

template <uint PAIRS>
kernel void mx_gemv_gate_up_swiglu_q4_t(device const float*       x      [[buffer(0)]],
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
    uint e;
    if (p.forced) {
        e = uint(ids[slot]);
    } else {
        int my_id; float my_prob;
        moe_select_topk(logits, p.E, p.K, lane, my_id, my_prob);
        e = uint(simd_shuffle(my_id, ushort(slot)));
        if (tg.x == 0 && slot == 0 && sg == 0 && lane < p.K) { ids[lane] = my_id; probs[lane] = my_prob; }
    }

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

#define MX_GUQ_INST(NAME, P) \
template [[host_name(NAME)]] kernel void mx_gemv_gate_up_swiglu_q4_t<P>( \
    device const float*, device const uchar*, device const uchar*, device const bfloat*, device const float*, \
    device int*, device float*, device float*, constant MoeGateUpParams&, uint2, uint, uint, uint);
MX_GUQ_INST("mx_gemv_gate_up_swiglu_q4_p1", 1)
MX_GUQ_INST("mx_gemv_gate_up_swiglu_q4_p2", 2)
MX_GUQ_INST("mx_gemv_gate_up_swiglu_q4_p4", 4)

template <uint R>
kernel void mx_gemv_down_q4_t(device const float*     h        [[buffer(0)]],
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

#define MX_DNQ_INST(NAME, R) \
template [[host_name(NAME)]] kernel void mx_gemv_down_q4_t<R>( \
    device const float*, device const uchar*, device const uchar*, device const bfloat*, device const int*, \
    device const float*, device float*, constant MoeDownParams&, uint, uint, uint, uint);
MX_DNQ_INST("mx_gemv_down_q4_r1", 1)
MX_DNQ_INST("mx_gemv_down_q4_r2", 2)
MX_DNQ_INST("mx_gemv_down_q4_r4", 4)
MX_DNQ_INST("mx_gemv_down_q4_r8", 8)

// Down with the experts split across simdgroups: a threadgroup of K*SPLIT
// simdgroups owns R output rows; simdgroup (s, part) accumulates expert slot
// s over blocks part, part+SPLIT*8, ...; partial sums meet in threadgroup
// memory and are added in a fixed order (deterministic).
template <uint R, uint SPLIT>
kernel void mx_gemv_down_ks_t(device const float*     h        [[buffer(0)]],
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
    threadgroup float part_sum[32][R];
    const uint o0 = tg * R;
    const uint nrows = min(R, p.H - o0);
    const uint nblk = p.I / 32;
    const uint rstride4 = nblk * 4;
    const uint q = lane & 3, bl = lane >> 2;
    const uint s = sg / SPLIT, part = sg % SPLIT;

    float acc[R];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;
    if (s < p.K) {
        const uint e = uint(ids[s]);
        const float pr = probs[s];
        const ulong row0 = (ulong)e * p.H + o0;
        device const uint* bp = (device const uint*)(blocks + row0 * nblk * 16);
        device const uchar* sp = scales + row0 * nblk;
        device const float* hs = h + (ulong)s * p.I;
        for (uint b = bl + part * 8; b < nblk; b += 8 * SPLIT) {
            uint w[R];
            float sc[R];
            for (uint r = 0; r < R; ++r) {
                const uint rr = min(r, nrows - 1);
                w[r] = bp[rr * rstride4 + b * 4 + q];
                sc[r] = e8m0_to_float(sp[rr * nblk + b]);
            }
            device const float4* x4 = (device const float4*)(hs + b * 32 + q * 8);
            const float4 xa = x4[0], xb = x4[1];
            for (uint r = 0; r < R; ++r) acc[r] += mx_quarter_dot(w[r], xa, xb) * sc[r];
        }
        for (uint r = 0; r < R; ++r) acc[r] = simd_sum(acc[r]) * pr;
    }
    if (lane == 0) for (uint r = 0; r < R; ++r) part_sum[sg][r] = acc[r];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane < nrows) {
        float v = 0.0f;
        for (uint i = 0; i < nsg; ++i) v += part_sum[i][lane];
        const uint o = o0 + lane;
        float bsum = 0.0f;
        for (uint k = 0; k < p.K; ++k) bsum += probs[k] * bf2f(bias[(ulong)uint(ids[k]) * p.H + o]);
        residual[o] += v * kMxUnscale + bsum;
    }
}

#define MX_DNK_INST(NAME, R, S) \
template [[host_name(NAME)]] kernel void mx_gemv_down_ks_t<R, S>( \
    device const float*, device const uchar*, device const uchar*, device const bfloat*, device const int*, \
    device const float*, device float*, constant MoeDownParams&, uint, uint, uint, uint);
MX_DNK_INST("mx_gemv_down_ks_r1_s1", 1, 1)
MX_DNK_INST("mx_gemv_down_ks_r2_s1", 2, 1)
MX_DNK_INST("mx_gemv_down_ks_r4_s1", 4, 1)
MX_DNK_INST("mx_gemv_down_ks_r8_s1", 8, 1)
MX_DNK_INST("mx_gemv_down_ks_r1_s2", 1, 2)
MX_DNK_INST("mx_gemv_down_ks_r2_s2", 2, 2)
MX_DNK_INST("mx_gemv_down_ks_r4_s2", 4, 2)
MX_DNK_INST("mx_gemv_down_ks_r8_s2", 8, 2)

// ---------------------------------------------------------------------------
// Generic lane mapping: LPB lanes share one block; each lane owns NW = 4/LPB
// 32-bit words (8 elements each) of it. 32/LPB blocks per simdgroup step.
// Weight words are loaded as uint (NW=1) or uint2 (NW=2,4; 8-byte aligned).
// ---------------------------------------------------------------------------
template <uint NW>
inline void mx_load_words(device const uchar* p, thread uint* w) {
    if (NW == 1) { w[0] = *(device const uint*)p; }
    else {
        for (uint j = 0; j < NW; j += 2) {
            const uint2 u = ((device const uint2*)p)[j / 2];
            w[j] = u.x; w[j + 1] = u.y;
        }
    }
}

template <uint LPB, uint PAIRS>
kernel void mx_gemv_gate_up_swiglu_l_t(device const float*       x      [[buffer(0)]],
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
    constexpr uint R = 2 * PAIRS, NW = 4 / LPB, BPS = 32 / LPB;
    const uint slot = tg.y;
    uint e;
    if (p.forced) {
        e = uint(ids[slot]);
    } else {
        int my_id; float my_prob;
        moe_select_topk(logits, p.E, p.K, lane, my_id, my_prob);
        e = uint(simd_shuffle(my_id, ushort(slot)));
        if (tg.x == 0 && slot == 0 && sg == 0 && lane < p.K) { ids[lane] = my_id; probs[lane] = my_prob; }
    }

    const uint nblk = p.H / 32;
    const uint i0 = (tg.x * nsg + sg) * PAIRS;
    if (i0 >= p.I) return;
    const uint npairs = min(PAIRS, p.I - i0);
    const uint nrows = 2 * npairs;
    const ulong row0 = (ulong)e * (2 * p.I) + 2 * i0;
    device const uchar* bp = blocks + row0 * nblk * 16;
    device const uchar* sp = scales + row0 * nblk;
    const uint rstride = nblk * 16;
    const uint q = lane % LPB, bl = lane / LPB;

    float acc[R];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;
    for (uint b = bl; b < nblk; b += BPS) {
        uint w[R][NW];
        float sc[R];
        for (uint r = 0; r < R; ++r) {
            const uint rr = min(r, nrows - 1);
            mx_load_words<NW>(bp + rr * rstride + b * 16 + q * NW * 4, w[r]);
            sc[r] = e8m0_to_float(sp[rr * nblk + b]);
        }
        device const float4* x4 = (device const float4*)(x + b * 32 + q * NW * 8);
        float4 xv[2 * NW];
        for (uint j = 0; j < 2 * NW; ++j) xv[j] = x4[j];
        for (uint r = 0; r < R; ++r) {
            float d = 0.0f;
            for (uint j = 0; j < NW; ++j) d += mx_quarter_dot(w[r][j], xv[2 * j], xv[2 * j + 1]);
            acc[r] += d * sc[r];
        }
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

#define MX_GUL_INST(NAME, L, P) \
template [[host_name(NAME)]] kernel void mx_gemv_gate_up_swiglu_l_t<L, P>( \
    device const float*, device const uchar*, device const uchar*, device const bfloat*, device const float*, \
    device int*, device float*, device float*, constant MoeGateUpParams&, uint2, uint, uint, uint);
MX_GUL_INST("mx_gu_l1_p1", 1, 1)
MX_GUL_INST("mx_gu_l1_p2", 1, 2)
MX_GUL_INST("mx_gu_l1_p4", 1, 4)
MX_GUL_INST("mx_gu_l2_p1", 2, 1)
MX_GUL_INST("mx_gu_l2_p2", 2, 2)
MX_GUL_INST("mx_gu_l2_p4", 2, 4)
MX_GUL_INST("mx_gu_l4_p1", 4, 1)
MX_GUL_INST("mx_gu_l4_p2", 4, 2)
MX_GUL_INST("mx_gu_l4_p4", 4, 4)

template <uint LPB, uint R>
kernel void mx_gemv_down_l_t(device const float*     h        [[buffer(0)]],
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
    constexpr uint NW = 4 / LPB, BPS = 32 / LPB;
    const uint o0 = (tg * nsg + sg) * R;
    if (o0 >= p.H) return;
    const uint nrows = min(R, p.H - o0);
    const uint nblk = p.I / 32;
    const uint rstride = nblk * 16;
    const uint q = lane % LPB, bl = lane / LPB;
    const uint nitems = p.K * nblk;

    float acc[R];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;
    for (uint it = bl; it < nitems; it += BPS) {
        const uint s = it / nblk, b = it - s * nblk;
        const uint e = uint(ids[s]);
        const float pr = probs[s];
        const ulong row0 = (ulong)e * p.H + o0;
        device const uchar* bp = blocks + row0 * rstride + b * 16 + q * NW * 4;
        device const uchar* sp = scales + row0 * nblk + b;
        uint w[R][NW];
        float sc[R];
        for (uint r = 0; r < R; ++r) {
            const uint rr = min(r, nrows - 1);
            mx_load_words<NW>(bp + rr * rstride, w[r]);
            sc[r] = e8m0_to_float(sp[rr * nblk]) * pr;
        }
        device const float4* x4 = (device const float4*)(h + (ulong)s * p.I + b * 32 + q * NW * 8);
        float4 xv[2 * NW];
        for (uint j = 0; j < 2 * NW; ++j) xv[j] = x4[j];
        for (uint r = 0; r < R; ++r) {
            float d = 0.0f;
            for (uint j = 0; j < NW; ++j) d += mx_quarter_dot(w[r][j], xv[2 * j], xv[2 * j + 1]);
            acc[r] += d * sc[r];
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

#define MX_DNL_INST(NAME, L, R) \
template [[host_name(NAME)]] kernel void mx_gemv_down_l_t<L, R>( \
    device const float*, device const uchar*, device const uchar*, device const bfloat*, device const int*, \
    device const float*, device float*, constant MoeDownParams&, uint, uint, uint, uint);
MX_DNL_INST("mx_dn_l1_r2", 1, 2)
MX_DNL_INST("mx_dn_l1_r4", 1, 4)
MX_DNL_INST("mx_dn_l2_r2", 2, 2)
MX_DNL_INST("mx_dn_l2_r4", 2, 4)
MX_DNL_INST("mx_dn_l2_r8", 2, 8)
MX_DNL_INST("mx_dn_l4_r2", 4, 2)
MX_DNL_INST("mx_dn_l4_r4", 4, 4)
MX_DNL_INST("mx_dn_l4_r8", 4, 8)

// Quarter-block mapping with U steps unrolled: all weight/scale/x loads of U
// steps are issued before any is consumed (more bytes in flight per lane).
template <uint PAIRS, uint U>
kernel void mx_gu_q4u_t(device const float*       x      [[buffer(0)]],
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
    uint e;
    if (p.forced) {
        e = uint(ids[slot]);
    } else {
        int my_id; float my_prob;
        moe_select_topk(logits, p.E, p.K, lane, my_id, my_prob);
        e = uint(simd_shuffle(my_id, ushort(slot)));
        if (tg.x == 0 && slot == 0 && sg == 0 && lane < p.K) { ids[lane] = my_id; probs[lane] = my_prob; }
    }
    const uint nblk = p.H / 32;
    const uint i0 = (tg.x * nsg + sg) * PAIRS;
    if (i0 >= p.I) return;
    const uint npairs = min(PAIRS, p.I - i0);
    const uint nrows = 2 * npairs;
    const ulong row0 = (ulong)e * (2 * p.I) + 2 * i0;
    device const uint* bp = (device const uint*)(blocks + row0 * nblk * 16);
    device const uchar* sp = scales + row0 * nblk;
    const uint rstride4 = nblk * 4;
    const uint q = lane & 3, bl = lane >> 2;

    float acc[R];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;
    for (uint b0 = bl; b0 < nblk; b0 += 8 * U) {
        uint w[U][R];
        float sc[U][R];
        float4 xa[U], xb[U];
        for (uint u = 0; u < U; ++u) {
            const uint b = b0 + 8 * u;
            const bool ok = b < nblk;
            const uint bb = ok ? b : b0;
            for (uint r = 0; r < R; ++r) {
                const uint rr = min(r, nrows - 1);
                w[u][r] = bp[rr * rstride4 + bb * 4 + q];
                sc[u][r] = ok ? e8m0_to_float(sp[rr * nblk + bb]) : 0.0f;
            }
            device const float4* x4 = (device const float4*)(x + bb * 32 + q * 8);
            xa[u] = x4[0]; xb[u] = x4[1];
        }
        for (uint u = 0; u < U; ++u)
            for (uint r = 0; r < R; ++r) acc[r] += mx_quarter_dot(w[u][r], xa[u], xb[u]) * sc[u][r];
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

#define MX_GUU_INST(NAME, P, U) \
template [[host_name(NAME)]] kernel void mx_gu_q4u_t<P, U>( \
    device const float*, device const uchar*, device const uchar*, device const bfloat*, device const float*, \
    device int*, device float*, device float*, constant MoeGateUpParams&, uint2, uint, uint, uint);
MX_GUU_INST("mx_gu_q4u_p1_u2", 1, 2)
MX_GUU_INST("mx_gu_q4u_p2_u2", 2, 2)
MX_GUU_INST("mx_gu_q4u_p1_u3", 1, 3)
MX_GUU_INST("mx_gu_q4u_p2_u3", 2, 3)
MX_GUU_INST("mx_gu_q4u_p1_u4", 1, 4)
MX_GUU_INST("mx_gu_q4u_p2_u4", 2, 4)

template <uint R, uint U>
kernel void mx_dn_q4u_t(device const float*     h        [[buffer(0)]],
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
        for (uint b0 = bl; b0 < nblk; b0 += 8 * U) {
            uint w[U][R];
            float sc[U][R];
            float4 xa[U], xb[U];
            for (uint u = 0; u < U; ++u) {
                const uint b = b0 + 8 * u;
                const bool ok = b < nblk;
                const uint bb = ok ? b : b0;
                for (uint r = 0; r < R; ++r) {
                    const uint rr = min(r, nrows - 1);
                    w[u][r] = bp[rr * rstride4 + bb * 4 + q];
                    sc[u][r] = ok ? e8m0_to_float(sp[rr * nblk + bb]) * pr : 0.0f;
                }
                device const float4* x4 = (device const float4*)(hs + bb * 32 + q * 8);
                xa[u] = x4[0]; xb[u] = x4[1];
            }
            for (uint u = 0; u < U; ++u)
                for (uint r = 0; r < R; ++r) acc[r] += mx_quarter_dot(w[u][r], xa[u], xb[u]) * sc[u][r];
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

#define MX_DNU_INST(NAME, R, U) \
template [[host_name(NAME)]] kernel void mx_dn_q4u_t<R, U>( \
    device const float*, device const uchar*, device const uchar*, device const bfloat*, device const int*, \
    device const float*, device float*, constant MoeDownParams&, uint, uint, uint, uint);
MX_DNU_INST("mx_dn_q4u_r2_u2", 2, 2)
MX_DNU_INST("mx_dn_q4u_r4_u2", 4, 2)
MX_DNU_INST("mx_dn_q4u_r2_u3", 2, 3)
MX_DNU_INST("mx_dn_q4u_r4_u3", 4, 3)
MX_DNU_INST("mx_dn_q4u_r2_u4", 2, 4)
MX_DNU_INST("mx_dn_q4u_r4_u4", 4, 4)
