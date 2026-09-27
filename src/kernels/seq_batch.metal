#include "common.h"

// Multi-sequence decode ("continuous batching"): one forward step for B
// independent sequences (2 <= B <= SB_MAX_ROWS), each row at its own position
// in its own KV cache. Host side: src/model/batch_ops.h.
//
// Every kernel but sb_lm_mma is the batched twin of a decode (M = 1) kernel
// with the SAME per-row arithmetic (lane mapping, accumulation order,
// reductions), so a row's residual stream and KV-cache rows are bitwise those
// of decoding the sequence alone:
//
//   sb_norm_prep       the RMSNorm (inv, x * w) the fused M = 1 kernels compute inline
//   sb_qkv_rope        attn_qkv_rope        (rope.metal)        dense weights read once for all rows
//   sb_attn            attn_decode_fused    (attention.metal)   per-row cache, position, split count
//   sb_gemv_bf16_*     gemv_bf16_r*         (gemv_bf16.metal)   dense weights read once for all rows
//   sb_router          moe_router           (moe.metal)         grid (E, rows)
//   sb_gate_up_*       mx_gemv_gate_up_swiglu_* (moe.metal), grouped by expert:
//                      tiles of <= 2 rows that selected it, weights decoded once per tile
//   sb_down_*          mx_gemv_down_*       (moe.metal)         grid (rows, blocks), row fastest
//   sb_gemv_i8_norm_*  gemv_i8_norm_r*      (gemv_int8.metal)   B = 2 lm_head
//   sb_lm_mma          lm_head on simdgroup MMA (B >= 3): only the logits
//                      see a different summation order (~1e-6 relative)
//   sb_argmax_*        argmax_f32_*         (gemv_bf16.metal)   grid (partials, rows)
//
// The dense GEMVs process "chunks" of NB <= B batch rows per threadgroup,
// the chunk index being the fastest grid dimension (the chunks of one weight
// block run back to back and share it through the cache).
//
// Per-row state comes from a small table (SbRow, bound as inline bytes): the
// row's cache buffers as raw GPU addresses (Metal 4 binds by address, so a
// kernel can follow them), its position and its full-layer capacity.
//
// Relies on gemv_bf16.metal (GEMV_SG, gemv_bf4, gemv_inv_rms, argmax helpers),
// gemv_int8.metal (i8x4), gemv_mxfp4.metal / moe.metal (mx_half2_bits,
// kMxUnscale, moe_select_topk, MoeRouterParams), attention.metal (ATTN_*,
// attn_bf4), rope.metal (ATTN_QKV_P) and prefill_core.h (pg_frag,
// pg_frag_coord, PG_UNROLL), all of which precede this file.

#define SB_MAX_ROWS 8u

struct SbRow {
    device bfloat* k_full;     // KVCache::k_full base (bf16 [layers_full][capacity][KV])
    device bfloat* v_full;
    device bfloat* k_slide;    // KVCache::k_slide base (bf16 [layers_slide][window][KV])
    device bfloat* v_slide;
    uint pos;                  // position decoded this step
    uint capacity;             // full-layer slots
    uint pad0, pad1;
};

// ---------------------------------------------------------------------------
// sb_norm_prep: per row b (one threadgroup of 256, as the M = 1 kernels that
// fold the RMSNorm in): inv[b] = rsqrt(mean(x[b]^2) + eps) via gemv_inv_rms
// and xg[b] = x[b] * g — the exact values attn_qkv_rope / gemv_i8_norm
// compute inline, computed once per step instead of once per threadgroup.
// ---------------------------------------------------------------------------
struct SbPrepParams { uint K; float eps; };

kernel void sb_norm_prep(device const float*  x   [[buffer(0)]],
                         device const bfloat* g   [[buffer(1)]],
                         device float*        xg  [[buffer(2)]],
                         device float*        inv [[buffer(3)]],
                         constant SbPrepParams& p [[buffer(4)]],
                         uint row  [[threadgroup_position_in_grid]],
                         uint tid  [[thread_position_in_threadgroup]],
                         uint tgs  [[threads_per_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sid  [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[32];
    device const float* xr = x + (ulong)row * p.K;
    const float v = gemv_inv_rms(xr, p.K, p.eps, scratch, tid, tgs, lane, sid);
    if (tid == 0) inv[row] = v;
    device const float4* x4 = (device const float4*)xr;
    device const uint2* g2 = (device const uint2*)g;
    device float4* o4 = (device float4*)(xg + (ulong)row * p.K);
    for (uint i = tid; i < p.K / 4; i += tgs) {
        float4 t = x4[i];
        t *= gemv_bf4(g2[i]);
        o4[i] = t;
    }
}

// ---------------------------------------------------------------------------
// sb_qkv_rope: rmsnorm + QKV GEMV (+bias) + RoPE + KV-cache write for NB rows.
//   x : fp32 [nb][H]    q_out : fp32 [nb][Q]    cs : rope table (row 0)
// Grid / threadgroup: as attn_qkv_rope.
// ---------------------------------------------------------------------------
struct SbQkvParams { uint H, Q, KV; float eps; uint nb, kv_slot, sliding, window; };

template <uint NB>
kernel void sb_qkv_rope(device const float*  x      [[buffer(0)]],
                        device const float*  invb   [[buffer(1)]],
                        device const bfloat* wq     [[buffer(2)]],
                        device const bfloat* bq     [[buffer(3)]],
                        device const bfloat* wk     [[buffer(4)]],
                        device const bfloat* bk     [[buffer(5)]],
                        device const bfloat* wv     [[buffer(6)]],
                        device const bfloat* bv     [[buffer(7)]],
                        device float*        q_out  [[buffer(8)]],
                        device const SbRow*  rows   [[buffer(9)]],
                        device const float2* cs     [[buffer(10)]],
                        constant SbQkvParams& p     [[buffer(11)]],
                        uint2 tg2  [[threadgroup_position_in_grid]],
                        uint2 tgs2 [[threads_per_threadgroup]],
                        uint tid   [[thread_index_in_threadgroup]],
                        uint lane  [[thread_index_in_simdgroup]],
                        uint sid   [[simdgroup_index_in_threadgroup]]) {
    constexpr int P = ATTN_QKV_P;
    // Row chunk tg2.x: batch rows [tg2.x * NB, +NB) of p.nb; weight block tg2.y.
    const uint rb0 = tg2.x * NB, tg = tg2.y, tgs = tgs2.x;
    x += (ulong)rb0 * p.H;
    q_out += (ulong)rb0 * p.Q;
    rows += rb0;
    const uint nbc = min(NB, p.nb - rb0);
    (void)tid; (void)tgs;
    invb += rb0;

    const uint gp0 = (tg * GEMV_SG + sid) * P;
    const uint total_pairs = (p.Q + 2 * p.KV) / 2;
    if (gp0 >= total_pairs) return;
    const uint j0 = gp0 % 32;
    uint rb = (gp0 / 32) * 64 + j0;

    device const bfloat* W; device const bfloat* B; uint kind;
    if (rb < p.Q)               { W = wq; B = bq; kind = 0; }
    else if (rb < p.Q + p.KV)   { rb -= p.Q; W = wk; B = bk; kind = 1; }
    else                        { rb -= p.Q + p.KV; W = wv; B = bv; kind = 2; }

    device const uint2* wp[2 * P];
    for (int i = 0; i < P; ++i) {
        wp[i]     = (device const uint2*)(W + (ulong)(rb + i) * p.H);
        wp[P + i] = (device const uint2*)(W + (ulong)(rb + 32 + i) * p.H);
    }
    float acc[NB][2 * P];
    for (uint b = 0; b < NB; ++b)
        for (int i = 0; i < 2 * P; ++i) acc[b][i] = 0.0f;
    for (uint k = lane * 8; k < p.H; k += 256) {
        const uint k4 = k >> 2;
        uint2 wa[2 * P], wb[2 * P];
        for (int r = 0; r < 2 * P; ++r) { wa[r] = wp[r][k4]; wb[r] = wp[r][k4 + 1]; }
        for (uint b = 0; b < NB; ++b) {
            device const float4* x4 = (device const float4*)(x + (ulong)min(b, nbc - 1) * p.H);
            const float4 xa = x4[k4], xb = x4[k4 + 1];   // x * norm weight (sb_norm_prep)
            for (int r = 0; r < 2 * P; ++r) acc[b][r] += dot(gemv_bf4(wa[r]), xa) + dot(gemv_bf4(wb[r]), xb);
        }
    }

    for (uint b = 0; b < NB; ++b) {
        if (b >= nbc) break;
        float a = 0.0f, c2 = 0.0f;
        for (int i = 0; i < P; ++i) {
            const float s0 = simd_sum(acc[b][i]);
            const float s1 = simd_sum(acc[b][P + i]);
            if (lane == uint(i)) { a = s0; c2 = s1; }
        }
        if (lane < uint(P)) {
            const uint r0 = rb + lane, r1 = rb + 32 + lane;
            const float inv = invb[b];
            a = a * inv + bf2f(B[r0]);
            c2 = c2 * inv + bf2f(B[r1]);
            const uint pos = rows[b].pos;
            if (kind != 2) {
                const float2 c = cs[pos * 32 + j0 + lane];
                const float ra = a * c.x - c2 * c.y;
                const float rbv = c2 * c.x + a * c.y;
                a = ra; c2 = rbv;
            }
            if (kind == 0) {
                q_out[(ulong)b * p.Q + r0] = a;
                q_out[(ulong)b * p.Q + r1] = c2;
            } else {
                const ulong row_off = p.sliding ? ((ulong)p.kv_slot * p.window + pos % p.window) * p.KV
                                                : ((ulong)p.kv_slot * rows[b].capacity + pos) * p.KV;
                device bfloat* dst = kind == 1 ? (p.sliding ? rows[b].k_slide : rows[b].k_full)
                                               : (p.sliding ? rows[b].v_slide : rows[b].v_full);
                dst[row_off + r0] = f2bf(a);
                dst[row_off + r1] = f2bf(c2);
            }
        }
    }
}

#define SB_QKV_INST(NB) \
template [[host_name("sb_qkv_rope_b" #NB)]] kernel void sb_qkv_rope<NB>( \
    device const float*, device const float*, device const bfloat*, device const bfloat*, device const bfloat*, \
    device const bfloat*, device const bfloat*, device const bfloat*, device float*, device const SbRow*, \
    device const float2*, constant SbQkvParams&, uint2, uint2, uint, uint, uint);
SB_QKV_INST(1) SB_QKV_INST(2) SB_QKV_INST(3) SB_QKV_INST(4) SB_QKV_INST(5) SB_QKV_INST(6) SB_QKV_INST(7) SB_QKV_INST(8)

// ---------------------------------------------------------------------------
// sb_attn: attn_decode_fused for row z = tg.z, over that row's cache layer.
// The split count follows attn_split (attention_ops.cpp) from the row's own
// attended length; threadgroups past it exit at once.
//   q, out : fp32 [nb][Q]   part : fp32 [nb][heads][max_splits][66]
//   cnt : uint [nb][kv_heads], zero on entry and exit
// Grid: (kv_heads, max split count over rows, nb) x 256 threads.
// ---------------------------------------------------------------------------
struct SbAttnParams { uint kv_heads, max_splits, kv_slot, sliding, window, split_cap; float scale; uint pad; };

kernel void sb_attn(device const float*   q     [[buffer(0)]],
                    device const SbRow*   rows  [[buffer(1)]],
                    device float*         part  [[buffer(2)]],
                    device const bfloat*  sinks [[buffer(3)]],
                    device float*         out   [[buffer(4)]],
                    device atomic_uint*   cnt   [[buffer(5)]],
                    constant SbAttnParams& p    [[buffer(6)]],
                    uint3 tg   [[threadgroup_position_in_grid]],
                    uint  tid  [[thread_index_in_threadgroup]],
                    uint  lane [[thread_index_in_simdgroup]],
                    uint  sid  [[simdgroup_index_in_threadgroup]]) {
    threadgroup float qs[ATTN_GROUP][ATTN_HD];
    threadgroup float ks[ATTN_BLOCK][ATTN_HD + 1];
    threadgroup float vs[ATTN_BLOCK][ATTN_HD];
    threadgroup uint last;

    const uint b = tg.z;
    const uint pos = rows[b].pos;
    const uint n = p.sliding ? min(pos + 1, p.window) : pos + 1;
    uint chunk = (n + p.split_cap - 1) / p.split_cap;
    chunk = max(32u, (chunk + 31) / 32 * 32);
    const uint nsplit = max(1u, (n + chunk - 1) / chunk);
    const uint kvh = tg.x, split = tg.y;
    if (split >= nsplit) return;

    const uint KVd = p.kv_heads * ATTN_HD;
    device const bfloat* K = p.sliding ? rows[b].k_slide + (ulong)p.kv_slot * p.window * KVd
                                       : rows[b].k_full + (ulong)p.kv_slot * rows[b].capacity * KVd;
    device const bfloat* V = p.sliding ? rows[b].v_slide + (ulong)p.kv_slot * p.window * KVd
                                       : rows[b].v_full + (ulong)p.kv_slot * rows[b].capacity * KVd;
    const uint heads = p.kv_heads * ATTN_GROUP;
    q += (ulong)b * heads * ATTN_HD;
    out += (ulong)b * heads * ATTN_HD;
    part += (ulong)b * heads * p.max_splits * ATTN_PSTRIDE;
    cnt += b * p.kv_heads;

    const uint head = kvh * ATTN_GROUP + sid;
    const uint c0 = split * chunk;
    const uint c1 = min(c0 + chunk, n);
    {
        device const float* qg = q + kvh * ATTN_GROUP * ATTN_HD;
        for (uint i = tid; i < ATTN_GROUP * ATTN_HD; i += 256)
            qs[i / ATTN_HD][i % ATTN_HD] = qg[i] * p.scale;
    }
    float m = -INFINITY, l = 0.0f, acc0 = 0.0f, acc1 = 0.0f;
    const ulong row_stride = (ulong)KVd;
    const uint lpos = tid / 8, lseg = (tid % 8) * 8;
    for (uint b0 = c0; b0 < c1; b0 += ATTN_BLOCK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            const uint pidx = b0 + lpos;
            float4 k0 = 0, k1 = 0, v0 = 0, v1 = 0;
            if (pidx < c1) {
                const ulong off = pidx * row_stride + kvh * ATTN_HD + lseg;
                const uint4 ku = *(device const uint4*)(K + off);
                const uint4 vu = *(device const uint4*)(V + off);
                k0 = attn_bf4(ku.xy); k1 = attn_bf4(ku.zw);
                v0 = attn_bf4(vu.xy); v1 = attn_bf4(vu.zw);
            }
            for (uint e = 0; e < 4; ++e) { ks[lpos][lseg + e] = k0[e]; ks[lpos][lseg + 4 + e] = k1[e]; }
            *(threadgroup float4*)&vs[lpos][lseg]     = v0;
            *(threadgroup float4*)&vs[lpos][lseg + 4] = v1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float s = 0.0f;
        for (uint d = 0; d < ATTN_HD; ++d) s += qs[sid][d] * ks[lane][d];
        if (b0 + lane >= c1) s = -INFINITY;
        const float bm = simd_max(s);
        const float m_new = max(m, bm);
        const float alpha = exp(m - m_new);
        const float pj = exp(s - m_new);
        l = l * alpha + simd_sum(pj);
        acc0 *= alpha; acc1 *= alpha;
        const uint nb = min(ATTN_BLOCK, c1 - b0);
        for (uint j = 0; j < nb; ++j) {
            const float pb = simd_shuffle(pj, ushort(j));
            acc0 += pb * vs[j][lane];
            acc1 += pb * vs[j][lane + 32];
        }
        m = m_new;
    }

    const float sink = bf2f(sinks[head]);
    if (nsplit == 1) {
        const float M = max(sink, m);
        const float w = exp(m - M);
        const float L = exp(sink - M) + l * w;
        out[head * ATTN_HD + lane] = acc0 * w / L;
        out[head * ATTN_HD + lane + 32] = acc1 * w / L;
        return;
    }
    device float* rec = part + ((ulong)head * p.max_splits + split) * ATTN_PSTRIDE;
    if (lane == 0) { rec[0] = m; rec[1] = l; }
    rec[2 + lane] = acc0;
    rec[2 + lane + 32] = acc1;

    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (tid == 0) last = atomic_fetch_add_explicit(&cnt[kvh], 1u, memory_order_relaxed) == nsplit - 1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!last) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);

    device const float* hrec = part + (ulong)head * p.max_splits * ATTN_PSTRIDE;
    float M = sink;
    for (uint s = 0; s < nsplit; ++s) M = max(M, hrec[s * ATTN_PSTRIDE]);
    float L = exp(sink - M), o0 = 0.0f, o1 = 0.0f;
    for (uint s = 0; s < nsplit; ++s) {
        const float w = exp(hrec[s * ATTN_PSTRIDE] - M);
        L += hrec[s * ATTN_PSTRIDE + 1] * w;
        o0 += hrec[s * ATTN_PSTRIDE + 2 + lane] * w;
        o1 += hrec[s * ATTN_PSTRIDE + 2 + lane + 32] * w;
    }
    out[head * ATTN_HD + lane] = o0 / L;
    out[head * ATTN_HD + lane + 32] = o1 / L;
    if (tid == 0) atomic_store_explicit(&cnt[kvh], 0u, memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// sb_gemv_bf16: y[b][r] = (accumulate ? y[b][r] : 0) + W[r,:] . x[b] + bias[r]
// for b < nb, as gemv_bf16_r* (no norm). x : fp32 [nb][ldx], y : fp32 [nb][ldy].
// Grid: ceil(rows / (R * GEMV_SG)) threadgroups of 256.
// ---------------------------------------------------------------------------
struct SbGemvParams { uint rows, K, flags, nb, ldx, ldy; };

template <int R, uint NB>
kernel void sb_gemv_bf16(device const bfloat*  W    [[buffer(0)]],
                         device const float*   x    [[buffer(1)]],
                         device float*         y    [[buffer(2)]],
                         device const bfloat*  bias [[buffer(3)]],
                         constant SbGemvParams& p   [[buffer(4)]],
                         uint2 tg2 [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sid  [[simdgroup_index_in_threadgroup]]) {
    // Row chunk tg2.x: batch rows [tg2.x * NB, +NB) of p.nb; weight block tg2.y.
    const uint rb0 = tg2.x * NB, tg = tg2.y;
    x += (ulong)rb0 * p.ldx;
    y += (ulong)rb0 * p.ldy;
    const uint nbc = min(NB, p.nb - rb0);
    const uint row0 = (tg * GEMV_SG + sid) * R;
    if (row0 >= p.rows) return;
    device const uint2* wp[R];
    for (int r = 0; r < R; ++r) {
        const uint row = min(row0 + uint(r), p.rows - 1);
        wp[r] = (device const uint2*)(W + (ulong)row * p.K);
    }
    float acc[NB][R];
    for (uint b = 0; b < NB; ++b)
        for (int r = 0; r < R; ++r) acc[b][r] = 0.0f;
    for (uint k = lane * 8; k < p.K; k += 256) {
        const uint k4 = k >> 2;
        uint2 wa[R], wb[R];
        for (int r = 0; r < R; ++r) { wa[r] = wp[r][k4]; wb[r] = wp[r][k4 + 1]; }
        for (uint b = 0; b < NB; ++b) {
            device const float4* x4 = (device const float4*)(x + (ulong)min(b, nbc - 1) * p.ldx);
            const float4 xa = x4[k4], xb = x4[k4 + 1];
            for (int r = 0; r < R; ++r) acc[b][r] += dot(gemv_bf4(wa[r]), xa) + dot(gemv_bf4(wb[r]), xb);
        }
    }
    for (uint b = 0; b < NB; ++b) {
        if (b >= nbc) break;
        float mine = 0.0f;
        for (int r = 0; r < R; ++r) {
            const float s = simd_sum(acc[b][r]);
            if (lane == uint(r)) mine = s;
        }
        const uint row = row0 + lane;
        if (lane < uint(R) && row < p.rows) {
            float v = mine * 1.0f;
            if (p.flags & GEMV_BIAS) v += bf2f(bias[row]);
            device float* yp = y + (ulong)b * p.ldy + row;
            if (p.flags & GEMV_ACCUM) v += *yp;
            *yp = v;
        }
    }
}

#define SB_GEMV_INST(R, NB) \
template [[host_name("sb_gemv_bf16_r" #R "_b" #NB)]] kernel void sb_gemv_bf16<R, NB>( \
    device const bfloat*, device const float*, device float*, device const bfloat*, constant SbGemvParams&, \
    uint2, uint, uint);
#define SB_GEMV_INST_R(R) SB_GEMV_INST(R, 1) SB_GEMV_INST(R, 2) SB_GEMV_INST(R, 3) SB_GEMV_INST(R, 4) SB_GEMV_INST(R, 5) \
                          SB_GEMV_INST(R, 6) SB_GEMV_INST(R, 7) SB_GEMV_INST(R, 8)
SB_GEMV_INST_R(2) SB_GEMV_INST_R(4)

// ---------------------------------------------------------------------------
// sb_router: moe_router for row tg.y (grid (E, nb)); per-row outputs
//   normed [nb][H], logits [nb][E], ids/probs [nb][K], counter [nb].
// ---------------------------------------------------------------------------
kernel void sb_router(device const float*       residual [[buffer(0)]],
                      device const bfloat*      norm_w   [[buffer(1)]],
                      device const bfloat*      router_w [[buffer(2)]],
                      device const bfloat*      router_b [[buffer(3)]],
                      device float*             normed   [[buffer(4)]],
                      device float*             logits   [[buffer(5)]],
                      constant MoeRouterParams& p        [[buffer(6)]],
                      device int*               ids      [[buffer(7)]],
                      device float*             probs    [[buffer(8)]],
                      device atomic_uint*       counter  [[buffer(9)]],
                      uint2 tg      [[threadgroup_position_in_grid]],
                      uint2 tgs2    [[threads_per_threadgroup]],
                      uint tid      [[thread_index_in_threadgroup]],
                      uint lane     [[thread_index_in_simdgroup]],
                      uint sg       [[simdgroup_index_in_threadgroup]]) {
    const uint tg_size = tgs2.x;
    threadgroup float scratch[32];
    threadgroup uint last;
    const uint e = tg.x, row = tg.y;
    residual += (ulong)row * p.H;
    normed += (ulong)row * p.H;
    logits += row * p.E;
    ids += row * p.K;
    probs += row * p.K;
    counter += row;
    const uint n4 = p.H / 4;
    device const float4* r4 = (device const float4*)residual;
    device const bfloat4* w4 = (device const bfloat4*)(router_w + (ulong)e * p.H);

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
// sb_gemv_i8_norm: final RMSNorm folded into the int8 lm_head GEMV for nb rows
// (gemv_i8_norm_r*). x : fp32 [nb][K], y : fp32 [nb][ldy].
// ---------------------------------------------------------------------------
struct SbI8Params { uint rows, K; float eps; uint nb, ldy, pad0, pad1, pad2; };

template <int R, uint NB>
kernel void sb_gemv_i8_norm(device const char*    W    [[buffer(0)]],
                            device const float*   S    [[buffer(1)]],
                            device const float*   x    [[buffer(2)]],
                            device float*         y    [[buffer(3)]],
                            device const float*   invb [[buffer(4)]],
                            constant SbI8Params&  p    [[buffer(5)]],
                            uint2 tg2  [[threadgroup_position_in_grid]],
                            uint2 tgs2 [[threads_per_threadgroup]],
                            uint tid   [[thread_index_in_threadgroup]],
                            uint lane  [[thread_index_in_simdgroup]],
                            uint sid   [[simdgroup_index_in_threadgroup]]) {
    // Row chunk tg2.x: batch rows [tg2.x * NB, +NB) of p.nb; weight block tg2.y.
    const uint rb0 = tg2.x * NB, tg = tg2.y, tgs = tgs2.x;
    x += (ulong)rb0 * p.K;
    y += (ulong)rb0 * p.ldy;
    const uint nbc = min(NB, p.nb - rb0);
    (void)tid; (void)tgs;
    invb += rb0;
    const uint row0 = (tg * GEMV_SG + sid) * R;
    if (row0 >= p.rows) return;
    device const uint4* wp[R];
    for (int r = 0; r < R; ++r) {
        const uint row = min(row0 + uint(r), p.rows - 1);
        wp[r] = (device const uint4*)(W + (ulong)row * p.K);
    }
    float acc[NB][R];
    PG_UNROLL for (uint b = 0; b < NB; ++b)
        PG_UNROLL for (int r = 0; r < R; ++r) acc[b][r] = 0.0f;
    for (uint k = lane * 16; k < p.K; k += 512) {
        float4 wf[R][4];   // int8 -> float once per step (the values the M = 1 kernel converts inline)
        PG_UNROLL for (int r = 0; r < R; ++r) {
            const uint4 w = wp[r][k >> 4];
            wf[r][0] = i8x4(w.x); wf[r][1] = i8x4(w.y); wf[r][2] = i8x4(w.z); wf[r][3] = i8x4(w.w);
        }
        const uint k4 = k >> 2;
        PG_UNROLL for (uint b = 0; b < NB; ++b) {
            device const float4* x4 = (device const float4*)(x + (ulong)min(b, nbc - 1) * p.K);
            float4 xv[4];   // x * norm weight (sb_norm_prep)
            PG_UNROLL for (int j = 0; j < 4; ++j) xv[j] = x4[k4 + j];
            PG_UNROLL for (int r = 0; r < R; ++r)
                acc[b][r] += dot(wf[r][0], xv[0]) + dot(wf[r][1], xv[1]) + dot(wf[r][2], xv[2]) + dot(wf[r][3], xv[3]);
        }
    }
    PG_UNROLL for (uint b = 0; b < NB; ++b) {
        if (b >= nbc) continue;
        float mine = 0.0f;
        PG_UNROLL for (int r = 0; r < R; ++r) {
            const float s = simd_sum(acc[b][r]);
            if (lane == uint(r)) mine = s;
        }
        const uint row = row0 + lane;
        if (lane < uint(R) && row < p.rows) y[(ulong)b * p.ldy + row] = mine * invb[b] * S[row];
    }
}

#define SB_I8_INST(R, NB) \
template [[host_name("sb_gemv_i8_norm_r" #R "_b" #NB)]] kernel void sb_gemv_i8_norm<R, NB>( \
    device const char*, device const float*, device const float*, device float*, device const float*, \
    constant SbI8Params&, uint2, uint2, uint, uint, uint);
#define SB_I8_INST_R(R) SB_I8_INST(R, 1) SB_I8_INST(R, 2) SB_I8_INST(R, 3) SB_I8_INST(R, 4) SB_I8_INST(R, 5) \
                        SB_I8_INST(R, 6) SB_I8_INST(R, 7) SB_I8_INST(R, 8)
SB_I8_INST_R(2) SB_I8_INST_R(4)

// ---------------------------------------------------------------------------
// Batched argmax (argmax_f32_partial / _final per row tg.y).
//   x : fp32 [nb][ldx]   pv/pi : [nb][256]   out : int32 [nb]
// ---------------------------------------------------------------------------
struct SbArgmaxParams { uint n, ldx, groups, pad; };

kernel void sb_argmax_partial(device const float*      x   [[buffer(0)]],
                              device float*            pv  [[buffer(1)]],
                              device uint*             pi  [[buffer(2)]],
                              constant SbArgmaxParams& p   [[buffer(3)]],
                              uint2 tg   [[threadgroup_position_in_grid]],
                              uint2 ntg  [[threadgroups_per_grid]],
                              uint2 tgs2 [[threads_per_threadgroup]],
                              uint tid   [[thread_index_in_threadgroup]],
                              uint lane  [[thread_index_in_simdgroup]],
                              uint sid   [[simdgroup_index_in_threadgroup]]) {
    const uint tgs = tgs2.x;
    threadgroup float sv[32];
    threadgroup uint  si[32];
    x += (ulong)tg.y * p.ldx;
    float bv = -INFINITY; uint bi = 0xFFFFFFFFu;
    for (uint i = tg.x * tgs + tid; i < p.n; i += ntg.x * tgs) gemv_argmax_pick(bv, bi, x[i], i);
    gemv_argmax_tg(bv, bi, sv, si, tid, tgs, lane, sid);
    if (tid == 0) { pv[tg.y * 256 + tg.x] = bv; pi[tg.y * 256 + tg.x] = bi; }
}

kernel void sb_argmax_final(device const float*      pv  [[buffer(0)]],
                            device const uint*       pi  [[buffer(1)]],
                            device int*              out [[buffer(2)]],
                            constant SbArgmaxParams& p   [[buffer(3)]],
                            uint row  [[threadgroup_position_in_grid]],
                            uint tid  [[thread_position_in_threadgroup]],
                            uint tgs  [[threads_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sid  [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sv[32];
    threadgroup uint  si[32];
    pv += row * 256;
    pi += row * 256;
    float bv = -INFINITY; uint bi = 0xFFFFFFFFu;
    for (uint i = tid; i < p.groups; i += tgs) gemv_argmax_pick(bv, bi, pv[i], pi[i]);
    gemv_argmax_tg(bv, bi, sv, si, tid, tgs, lane, sid);
    if (tid == 0) out[row] = int(bi == 0xFFFFFFFFu ? 0u : bi);
}


// ---------------------------------------------------------------------------
// MoE: the n = nb*K <= 32 (row, slot) entries of a step (entry = row*K + slot)
// are grouped by expert on the GPU, from the router's ids, by every simdgroup
// with simd ops (no sort dispatch).
//
// Capped expert tiles: tile t = chunk of at most C entries of one expert.
// Experts in ascending order; an expert with m entries owns ceil(m / C)
// consecutive tiles, chunk q holding its entries [qC, qC + C) (ascending
// entry order). Returns the expert (or -1 past the last tile) and the chunk's
// entry mask.
// ---------------------------------------------------------------------------
inline int sb_tile_capped(device const int* ids, uint n, uint t, uint C, uint lane, thread uint& mask) {
    const int e = lane < n ? ids[lane] : 0x7FFFFFFF;
    bool first = lane < n;
    uint start = 0;
    for (ushort j = 0; j < 32; ++j) {
        const int ej = simd_shuffle(e, j);
        if (uint(j) < lane && ej == e) first = false;
    }
    // members of my expert (entries with the same id)
    uint mine = 0;
    for (ushort j = 0; j < 32; ++j) {
        const int ej = simd_shuffle(e, j);
        if (uint(j) < n && ej == e) mine |= 1u << j;
    }
    const uint ntiles = (popcount(mine) + C - 1) / C;
    for (ushort j = 0; j < 32; ++j) {
        const int ej = simd_shuffle(e, j);
        const uint fj = simd_shuffle(first ? 1u : 0u, j);
        const uint tj = simd_shuffle(ntiles, j);
        if (fj && ej < e) start += tj;
    }
    const bool owner = first && t >= start && t < start + ntiles;
    const ulong hit = ulong(simd_vote::vote_t(simd_ballot(owner)));
    if (hit == 0) { mask = 0; return -1; }
    const ushort src = ushort(ctz(uint(hit)));
    const int ex = simd_shuffle(e, src);
    const uint m = simd_shuffle(mine, src);
    const uint q = t - simd_shuffle(start, src);
    // entries [qC, qC + C) of m
    uint out = 0, k = 0, mm = m;
    for (uint i = 0; i < 32 && mm; ++i) {
        const uint bit = ctz(mm);
        if (k >= q * C && k < q * C + C) out |= 1u << bit;
        mm &= mm - 1;
        ++k;
    }
    mask = out;
    return ex;
}

// ---------------------------------------------------------------------------
// sb_gate_up: gate_up GEMV + clamped SwiGLU over capped tiles, the bitwise
// twin of mx_gemv_gate_up_swiglu for every entry of the tile: each lane
// decodes its quarter block of R weight rows once and applies it to the
// tile's <= NB activation rows with the M = 1 arithmetic (mx_quarter_dot's
// values and order).
//   x = normed [nb][H], N = 2I weight rows (gate, up interleaved)
//   h [nb*K][I] (entry order), + bias, clamped SwiGLU
// Register-lean: decoded values stay half2 (converted per use: same values),
// the next step's words and scales are loaded before the current math, loops
// fully unrolled. Grid (tiles = nb*K upper bound, row blocks): the chunks of
// one expert run back to back.
//
// Measured on M2 Max (tests/test_batch.cpp batch_zz_profile, B = 8 real
// routing, ~12 distinct experts / 32 entries per layer): the per-tile cost
// is set by registers (occupancy), not DRAM — a tile compiled for 8 entries
// runs ~2x slower than one for 1 even with a single live entry, and a tile
// whose weights hit in the cache is no faster than a fresh one. Tiles of <= 2
// entries with R = 8 rows per simdgroup were best: 14.2 ms per step at B = 8,
// vs 18.8 for one tile per expert, 14.2 for an MMA version (8 columns, ~2.7
// used), 16.6 for one threadgroup per (row, slot) sharing weights through the
// cache and 20.5 for one tile per entry. Also slower: the tile's activations
// staged in threadgroup memory (per step in lockstep, or whole: occupancy),
// and a byte -> float2 LUT decode in threadgroup memory. The DRAM floor for
// ~12 distinct experts is ~7.7 ms: the grouped GEMV is compute-issue bound.
// ---------------------------------------------------------------------------
struct SbGateUpParams { uint N, Kd, lda, K, n, I; float limit, alpha; };

struct SbHalfQ { half2 v0, v1, v2, v3; };
inline SbHalfQ sb_decode_h(uint w) {
    return SbHalfQ{as_type<half2>(mx_half2_bits(w)), as_type<half2>(mx_half2_bits(w >> 4)),
                   as_type<half2>(mx_half2_bits(w >> 8)), as_type<half2>(mx_half2_bits(w >> 12))};
}
inline float sb_qdot(SbHalfQ d, float4 xa, float4 xb) {
    const float2 v0 = float2(d.v0), v1 = float2(d.v1), v2 = float2(d.v2), v3 = float2(d.v3);
    return dot(float4(v0.x, v1.x, v2.x, v3.x), xa) + dot(float4(v0.y, v1.y, v2.y, v3.y), xb);
}

template <uint R, uint NB>
kernel void sb_gate_up(device const float*      x      [[buffer(0)]],
                       device const uchar*      blocks [[buffer(1)]],
                       device const uchar*      scales [[buffer(2)]],
                       device const bfloat*     bias   [[buffer(3)]],
                       device const int*        ids    [[buffer(4)]],
                       device float*            out    [[buffer(5)]],
                       constant SbGateUpParams& p      [[buffer(6)]],
                       uint2 tg   [[threadgroup_position_in_grid]],
                       uint  sg   [[simdgroup_index_in_threadgroup]],
                       uint  nsg  [[simdgroups_per_threadgroup]],
                       uint  lane [[thread_index_in_simdgroup]]) {
    constexpr uint PAIRS = R / 2;
    uint mask;
    const int ex = sb_tile_capped(ids, p.n, tg.x, NB, lane, mask);
    if (ex < 0) return;
    const uint e = uint(ex);
    uint ent[NB];
    uint cnt = 0;
    PG_UNROLL for (uint j = 0; j < NB; ++j) {
        ent[j] = mask ? ctz(mask) : 0;
        if (mask) { mask &= mask - 1; cnt = j + 1; }
    }
    const uint r0 = (tg.y * nsg + sg) * R;     // first weight row (of N = 2I)
    if (r0 >= p.N) return;
    const uint nrows = min(R, p.N - r0);
    const uint nblk = p.Kd / 32;
    const uint rstride4 = nblk * 4;
    const uint q = lane & 3, bl = lane >> 2;
    const ulong row0 = (ulong)e * p.N + r0;
    device const uint* bp = (device const uint*)(blocks + row0 * nblk * 16);
    device const uchar* sp = scales + row0 * nblk;
    device const float* xr[NB];
    PG_UNROLL for (uint j = 0; j < NB; ++j) xr[j] = x + (ulong)(ent[j] / p.K) * p.lda;

    float acc[NB][R];
    PG_UNROLL for (uint j = 0; j < NB; ++j)
        PG_UNROLL for (uint r = 0; r < R; ++r) acc[j][r] = 0.0f;
    uint w[R];
    uchar s8[R];
    uint b = bl;
    if (b < nblk)
        PG_UNROLL for (uint r = 0; r < R; ++r) {
            const uint rr = min(r, nrows - 1);
            w[r] = bp[rr * rstride4 + b * 4 + q];
            s8[r] = sp[rr * nblk + b];
        }
    for (; b < nblk; b += 8) {
        SbHalfQ d[R];
        float sc[R];
        PG_UNROLL for (uint r = 0; r < R; ++r) { d[r] = sb_decode_h(w[r]); sc[r] = e8m0_to_float(s8[r]); }
        const uint bn = b + 8;
        if (bn < nblk)
            PG_UNROLL for (uint r = 0; r < R; ++r) {
                const uint rr = min(r, nrows - 1);
                w[r] = bp[rr * rstride4 + bn * 4 + q];
                s8[r] = sp[rr * nblk + bn];
            }
        PG_UNROLL for (uint j = 0; j < NB; ++j) {
            if (j < cnt) {
                device const float4* x4 = (device const float4*)(xr[j] + b * 32 + q * 8);
                const float4 xa = x4[0], xb = x4[1];
                PG_UNROLL for (uint r = 0; r < R; ++r) acc[j][r] += sb_qdot(d[r], xa, xb) * sc[r];
            }
        }
    }
    PG_UNROLL for (uint j = 0; j < NB; ++j) {
        if (j >= cnt) continue;
        PG_UNROLL for (uint r = 0; r < R; ++r) acc[j][r] = simd_sum(acc[j][r]);
        const uint i0 = r0 / 2, npairs = nrows / 2;
        if (lane < npairs) {
            float g = 0.0f, u = 0.0f;
            PG_UNROLL for (uint jj = 0; jj < PAIRS; ++jj)
                if (jj == lane) { g = acc[j][2 * jj]; u = acc[j][2 * jj + 1]; }
            const uint i = i0 + lane;
            device const bfloat* bb = bias + (ulong)e * p.N + 2 * i;
            g = g * kMxUnscale + bf2f(bb[0]);
            u = u * kMxUnscale + bf2f(bb[1]);
            g = min(g, p.limit);
            u = clamp(u, -p.limit, p.limit);
            const float glu = g / (1.0f + exp(-p.alpha * g));
            out[(ulong)ent[j] * p.I + i] = (u + 1.0f) * glu;
        }
    }
}

#define SB_GU_INST(R, NB) \
template [[host_name("sb_gate_up_r" #R "_c" #NB)]] kernel void sb_gate_up<R, NB>( \
    device const float*, device const uchar*, device const uchar*, device const bfloat*, device const int*, \
    device float*, constant SbGateUpParams&, uint2, uint, uint, uint);
SB_GU_INST(4, 1) SB_GU_INST(4, 2) SB_GU_INST(8, 1) SB_GU_INST(8, 2)

// ---------------------------------------------------------------------------
// sb_down: mx_gemv_down (fused routing-weighted sum over the row's K experts
// + residual add) for row tg.x, grid (nb, blocks): the row index is the
// fastest grid dimension, so the rows' threadgroups for one block of output
// features run back to back and share the experts they have in common
// through the cache. Measured faster than grouping the down GEMV by expert
// like gate_up (which, to stay exact, needs per-slot outputs and a reduce
// dispatch: 7.9 vs 8.2 ms per step at B = 8, 4.3 vs 4.8 at B = 4), and the
// M = 1 kernel stays untouched.
//   h [nb][K][I], ids/probs [nb][K], residual [nb][H]
// ---------------------------------------------------------------------------
template <uint R>
kernel void sb_down(device const float*     h        [[buffer(0)]],
                    device const uchar*     blocks   [[buffer(1)]],
                    device const uchar*     scales   [[buffer(2)]],
                    device const bfloat*    bias     [[buffer(3)]],
                    device const int*       ids      [[buffer(4)]],
                    device const float*     probs    [[buffer(5)]],
                    device float*           residual [[buffer(6)]],
                    constant MoeDownParams& p        [[buffer(7)]],
                    uint2 tg  [[threadgroup_position_in_grid]],
                    uint sg   [[simdgroup_index_in_threadgroup]],
                    uint nsg  [[simdgroups_per_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tg.x;
    h += (ulong)row * p.K * p.I;
    ids += row * p.K;
    probs += row * p.K;
    residual += (ulong)row * p.H;
    const uint o0 = (tg.y * nsg + sg) * R;
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

template [[host_name("sb_down_r2")]] kernel void sb_down<2>(device const float*, device const uchar*, device const uchar*,
    device const bfloat*, device const int*, device const float*, device float*, constant MoeDownParams&, uint2, uint, uint, uint);
template [[host_name("sb_down_r4")]] kernel void sb_down<4>(device const float*, device const uchar*, device const uchar*,
    device const bfloat*, device const int*, device const float*, device float*, constant MoeDownParams&, uint2, uint, uint, uint);

// ---------------------------------------------------------------------------
// sb_lm_mma: int8 lm_head on simdgroup MMA for nb <= 8 rows (all 8 MMA columns
// useful at B = 8, unlike the expert GEMMs). Only the final logits see the
// different K summation order (~1e-6 relative); the residual stream and KV
// caches stay bitwise equal to the M = 1 path.
//   C[t] (8 vocab rows x 8 batch rows) += A (8 rows x 8 k) . B (8 k x 8 rows)
// K walked 64 per step: lane (row r = fc.y, pair c = fc.x/2) loads 16 int8 of
// its row at k0 + 16c (uint4); fragment f (0..7) takes elements (f, f + 8) as
// columns (2c, 2c+1), so its column j is k = k0 + 16 (j/2) + f + 8 (j%2); the
// matching B row k' = fc.y is xg[b][k0 + 16 (fc.y/2) + 8 (fc.y%2) + f]: two
// float4 per batch row per step.
//   logits[b][v] = C * inv[b] * S[v]   (xg = x * final_norm, inv from sb_norm_prep)
// Grid: V / (8 T nsg) threadgroups of nsg simdgroups.
// ---------------------------------------------------------------------------
struct SbLmParams { uint rows, K, nb, ldy; };

template <uint T>
kernel void sb_lm_mma(device const char*    W    [[buffer(0)]],
                      device const float*   S    [[buffer(1)]],
                      device const float*   xg   [[buffer(2)]],
                      device float*         y    [[buffer(3)]],
                      device const float*   invb [[buffer(4)]],
                      constant SbLmParams&  p    [[buffer(5)]],
                      uint tg   [[threadgroup_position_in_grid]],
                      uint sg   [[simdgroup_index_in_threadgroup]],
                      uint nsg  [[simdgroups_per_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    const ushort2 fc = pg_frag_coord(ushort(lane));
    const uint n0 = (tg * nsg + sg) * (8 * T);
    if (n0 >= p.rows) return;
    const uint ba = min(uint(fc.x), p.nb - 1), bb = min(uint(fc.x) + 1u, p.nb - 1);
    device const float* xa_row = xg + (ulong)ba * p.K + 16 * (fc.y / 2) + 8 * (fc.y % 2);
    device const float* xb_row = xg + (ulong)bb * p.K + 16 * (fc.y / 2) + 8 * (fc.y % 2);
    const uint c = fc.x / 2;
    device const uint4* wrow[T];
    PG_UNROLL for (uint t = 0; t < T; ++t) {
        const uint row = min(n0 + 8 * t + fc.y, p.rows - 1);
        wrow[t] = (device const uint4*)(W + (ulong)row * p.K + 16 * c);
    }
    pg_frag acc[T];
    PG_UNROLL for (uint t = 0; t < T; ++t) acc[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = 0; k0 < p.K; k0 += 64) {
        uint4 w[T];
        PG_UNROLL for (uint t = 0; t < T; ++t) w[t] = wrow[t][k0 / 16];
        const float4 a0 = *(device const float4*)(xa_row + k0), a1 = *(device const float4*)(xa_row + k0 + 4);
        const float4 b0 = *(device const float4*)(xb_row + k0), b1 = *(device const float4*)(xb_row + k0 + 4);
        PG_UNROLL for (uint f = 0; f < 8; ++f) {
            pg_frag bf;
            bf.thread_elements()[0] = f < 4 ? a0[f] : a1[f - 4];
            bf.thread_elements()[1] = f < 4 ? b0[f] : b1[f - 4];
            PG_UNROLL for (uint t = 0; t < T; ++t) {
                // element f and f + 8 of the lane's 16 int8: word f/4 byte f%4, word 2 + f/4 byte f%4
                const uint lo = f < 4 ? w[t].x : w[t].y, hi = f < 4 ? w[t].z : w[t].w;
                pg_frag af;
                af.thread_elements()[0] = float(as_type<char4>(lo)[f % 4]);
                af.thread_elements()[1] = float(as_type<char4>(hi)[f % 4]);
                simdgroup_multiply_accumulate(acc[t], af, bf, acc[t]);
            }
        }
    }
    PG_UNROLL for (uint t = 0; t < T; ++t) {
        const uint row = n0 + 8 * t + fc.y;
        if (row >= p.rows) continue;
        const float s = S[row];
        if (uint(fc.x) < p.nb) y[(ulong)fc.x * p.ldy + row] = acc[t].thread_elements()[0] * invb[fc.x] * s;
        if (uint(fc.x) + 1u < p.nb) y[(ulong)(fc.x + 1) * p.ldy + row] = acc[t].thread_elements()[1] * invb[fc.x + 1] * s;
    }
}

template [[host_name("sb_lm_mma_t2")]] kernel void sb_lm_mma<2>(device const char*, device const float*, device const float*,
    device float*, device const float*, constant SbLmParams&, uint, uint, uint, uint);
template [[host_name("sb_lm_mma_t4")]] kernel void sb_lm_mma<4>(device const char*, device const float*, device const float*,
    device float*, device const float*, constant SbLmParams&, uint, uint, uint, uint);
