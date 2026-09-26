#include "common.h"

// Batched-prefill attention for gpt-oss (M > 1 positions per pass):
//
//   pf_rope_kv     RoPE (YaRN table, as rope.metal) on q in place and on k;
//                  k, v -> bf16 staging rows [M][KV] (+ the full-layer cache rows)
//   pf_attention   causal / sliding-window flash attention with sinks and GQA
//   pf_ring_write  sliding layers: staging rows of the chunk's last `window`
//                  positions -> ring slots (after attention, see below)
//
// Key sources. Query position p (absolute) attends keys j in [start(p), p],
// start = 0 (full) or max(0, p - window + 1) (sliding). Keys j >= pos0 (this
// chunk) always come from the staging rows; keys j < pos0 from the cache
// (slot j, or ring slot j % window). A chunk longer than the window
// overwrites ring slots that earlier queries in the same chunk still need,
// so the ring is only updated after the attention dispatch; before it, ring
// slot j % window still holds position j for every j in [pos0 - window, pos0).
//
// Layout constants: head_dim 64, 8 query heads per kv head (host checks).

// ---------------------------------------------------------------------------
// pf_rope_kv: one threadgroup per row m (position pos0 + m).
//   qkv : fp32 [M][ldq]  (q at [0, Q), k at [Q, Q+KV), v at [Q+KV, Q+2KV))
//   cs  : float2 rope table, bound at row pos0
//   kst, vst : bf16 [M][KV]
//   kc, vc : full layers: bf16 cache rows for pos0.. (bound at row pos0); unused when sliding
// ---------------------------------------------------------------------------
struct PfRopeParams { uint ldq, Q, KV, full; };

kernel void pf_rope_kv(device float*        qkv [[buffer(0)]],
                       device const float2* cs  [[buffer(1)]],
                       device bfloat*       kst [[buffer(2)]],
                       device bfloat*       vst [[buffer(3)]],
                       device bfloat*       kc  [[buffer(4)]],
                       device bfloat*       vc  [[buffer(5)]],
                       constant PfRopeParams& p [[buffer(6)]],
                       uint m   [[threadgroup_position_in_grid]],
                       uint tid [[thread_position_in_threadgroup]],
                       uint tgs [[threads_per_threadgroup]]) {
    device float* row = qkv + (ulong)m * p.ldq;
    device const float2* c = cs + m * 32;
    const uint qpairs = p.Q / 2, kpairs = p.KV / 2;
    for (uint i = tid; i < qpairs + kpairs; i += tgs) {
        const bool isq = i < qpairs;
        const uint ii = isq ? i : i - qpairs;
        const uint h = ii / 32, j = ii % 32;
        const uint col = h * 64 + j;
        device float* src = isq ? row + col : row + p.Q + col;
        const float a = src[0], b = src[32];
        const float2 r = c[j];
        const float ra = a * r.x - b * r.y, rb = b * r.x + a * r.y;
        if (isq) { src[0] = ra; src[32] = rb; }
        else {
            const bfloat ka = bfloat(ra), kb = bfloat(rb);
            kst[(ulong)m * p.KV + col] = ka; kst[(ulong)m * p.KV + col + 32] = kb;
            if (p.full) { kc[(ulong)m * p.KV + col] = ka; kc[(ulong)m * p.KV + col + 32] = kb; }
        }
    }
    for (uint i = tid; i < p.KV; i += tgs) {
        const bfloat v = bfloat(row[p.Q + p.KV + i]);
        vst[(ulong)m * p.KV + i] = v;
        if (p.full) vc[(ulong)m * p.KV + i] = v;
    }
}

// ---------------------------------------------------------------------------
// pf_attention. Threadgroup = (block of 32 query rows, one query head),
// 4 simdgroups x 8 query rows. Per key block of 32:
//   S = Q K^T (8 x 32 per simdgroup, simdgroup MMA, fp32), scale folded into Q,
//   online softmax in registers (a row's 32 scores live in 4 lanes: xor 1, 8),
//   O = O * alpha + P V (8 x 64, fp32 MMA).
// K is staged transposed [64][32], V natural [32][64], both bf16 (exact).
// Finalize with the sink: out = O / (l + exp(sink - m)) (sink carries no value).
//   q   : fp32 [M][ldq] (roped)      out : fp32 [M][Q]
//   kst, vst : bf16 [M][KV] (this chunk)   kc, vc : bf16 cache layer base
// Grid: (ceil(M / 32), heads) threadgroups of 128 threads.
// ---------------------------------------------------------------------------
#define PA_BQ 32u
#define PA_BK 32u
#define PA_LDK (PA_BK + 8u)
#define PA_LDV (64u + 8u)

struct PfAttnParams { uint M, pos0, ldq, Q, KV, window, sliding; float scale; };

kernel void pf_attention(device const float*  q     [[buffer(0)]],
                         device const bfloat* kst   [[buffer(1)]],
                         device const bfloat* vst   [[buffer(2)]],
                         device const bfloat* kc    [[buffer(3)]],
                         device const bfloat* vc    [[buffer(4)]],
                         device const bfloat* sinks [[buffer(5)]],
                         device float*        out   [[buffer(6)]],
                         constant PfAttnParams& p   [[buffer(7)]],
                         uint2 tg   [[threadgroup_position_in_grid]],
                         uint  t    [[thread_index_in_threadgroup]],
                         uint  sg   [[simdgroup_index_in_threadgroup]],
                         uint  lane [[thread_index_in_simdgroup]]) {
    threadgroup bfloat Kt[64 * PA_LDK];
    threadgroup bfloat Vs[PA_BK * PA_LDV];

    const uint h = tg.y, kvh = h / 8;
    const uint qb = tg.x * PA_BQ;                       // first query row of the block
    const uint qa = p.pos0 + qb;                        // its absolute position
    const uint qz = p.pos0 + min(qb + PA_BQ, p.M) - 1;  // last absolute position in the block
    const uint kstart = p.sliding ? (qa >= p.window ? qa - p.window + 1 : 0) : 0;
    const uint kb0 = kstart / PA_BK * PA_BK;

    const ushort2 fc = pg_frag_coord(ushort(lane));
    const uint r_local = sg * 8 + fc.y;                 // this lane's query row within the block
    const uint qrow = min(qb + r_local, p.M - 1);
    const uint qpos = p.pos0 + qb + r_local;            // absolute position (may be past the chunk: ignored)
    const uint qlo = p.sliding ? (qpos >= p.window ? qpos - p.window + 1 : 0) : 0;

    // Q fragments (8 rows x 64 dims), pre-scaled.
    pg_frag qf[8];
    {
        device const float* qp = q + (ulong)qrow * p.ldq + h * 64 + fc.x;
        PG_UNROLL for (uint d = 0; d < 8; ++d) {
            const float2 v = *(device const float2*)(qp + d * 8) * p.scale;
            qf[d].thread_elements()[0] = v.x;
            qf[d].thread_elements()[1] = v.y;
        }
    }
    pg_frag o[8];
    PG_UNROLL for (uint d = 0; d < 8; ++d) o[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float m = -INFINITY, l = 0.0f;

    // Loader mapping: thread t -> key t/4, dims (t%4)*16 .. +16.
    const uint lk = t / 4, ld0 = (t % 4) * 16;
    for (uint kb = kb0; kb <= qz; kb += PA_BK) {
        {
            const uint j = kb + lk;
            uint4 k0 = 0, k1 = 0, v0 = 0, v1 = 0;
            if (j >= kstart && j <= qz) {
                device const bfloat* ks; device const bfloat* vs;
                if (j >= p.pos0) {
                    const ulong off = (ulong)(j - p.pos0) * p.KV + kvh * 64 + ld0;
                    ks = kst + off; vs = vst + off;
                } else {
                    const uint slot = p.sliding ? j % p.window : j;
                    const ulong off = (ulong)slot * p.KV + kvh * 64 + ld0;
                    ks = kc + off; vs = vc + off;
                }
                k0 = ((device const uint4*)ks)[0]; k1 = ((device const uint4*)ks)[1];
                v0 = ((device const uint4*)vs)[0]; v1 = ((device const uint4*)vs)[1];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            threadgroup ushort* kd = (threadgroup ushort*)Kt + ld0 * PA_LDK + lk;
            const uint kw[8] = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w};
            PG_UNROLL for (uint e = 0; e < 8; ++e) {
                kd[(2 * e) * PA_LDK] = ushort(kw[e] & 0xFFFFu);
                kd[(2 * e + 1) * PA_LDK] = ushort(kw[e] >> 16);
            }
            threadgroup uint4* vd = (threadgroup uint4*)((threadgroup ushort*)Vs + lk * PA_LDV + ld0);
            vd[0] = v0; vd[1] = v1;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        // S = Q K^T: 4 fragments of 8 keys.
        pg_frag s[4];
        PG_UNROLL for (uint j = 0; j < 4; ++j) {
            s[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            PG_UNROLL for (uint d = 0; d < 8; ++d) {
                pg_frag kf;
                threadgroup const bfloat* kp = Kt + (d * 8 + fc.y) * PA_LDK + j * 8 + fc.x;
                kf.thread_elements()[0] = float(kp[0]);
                kf.thread_elements()[1] = float(kp[1]);
                simdgroup_multiply_accumulate(s[j], qf[d], kf, s[j]);
            }
        }
        // Mask + online softmax (row = fc.y; this lane holds columns j*8 + fc.x, +1).
        float mx = -INFINITY;
        PG_UNROLL for (uint j = 0; j < 4; ++j) {
            PG_UNROLL for (uint e = 0; e < 2; ++e) {
                const uint kp = kb + j * 8 + fc.x + e;
                const bool ok = kp <= qpos && kp >= qlo;
                const float v = ok ? float(s[j].thread_elements()[e]) : -INFINITY;
                s[j].thread_elements()[e] = v;
                mx = max(mx, v);
            }
        }
        mx = max(mx, simd_shuffle_xor(mx, ushort(1)));
        mx = max(mx, simd_shuffle_xor(mx, ushort(8)));
        const float m_new = max(m, mx);
        const float alpha = m_new == -INFINITY ? 1.0f : exp(m - m_new);
        float rs = 0.0f;
        PG_UNROLL for (uint j = 0; j < 4; ++j) {
            PG_UNROLL for (uint e = 0; e < 2; ++e) {
                const float v = s[j].thread_elements()[e];
                const float pv = v == -INFINITY ? 0.0f : exp(v - m_new);
                s[j].thread_elements()[e] = pv;
                rs += pv;
            }
        }
        rs += simd_shuffle_xor(rs, ushort(1));
        rs += simd_shuffle_xor(rs, ushort(8));
        l = l * alpha + rs;
        m = m_new;
        // O = O * alpha + P V.
        PG_UNROLL for (uint d = 0; d < 8; ++d) {
            o[d].thread_elements()[0] *= alpha;
            o[d].thread_elements()[1] *= alpha;
            PG_UNROLL for (uint j = 0; j < 4; ++j) {
                pg_frag vf;
                threadgroup const bfloat* vp = Vs + (j * 8 + fc.y) * PA_LDV + d * 8 + fc.x;
                vf.thread_elements()[0] = float(vp[0]);
                vf.thread_elements()[1] = float(vp[1]);
                simdgroup_multiply_accumulate(o[d], s[j], vf, o[d]);
            }
        }
    }

    if (qb + r_local >= p.M) return;
    const float sink = float(sinks[h]);
    const float M_ = max(m, sink);
    const float w = exp(m - M_);
    const float inv = w / (l * w + exp(sink - M_));
    device float* op = out + (ulong)(qb + r_local) * p.Q + h * 64 + fc.x;
    PG_UNROLL for (uint d = 0; d < 8; ++d)
        *(device float2*)(op + d * 8) = float2(o[d].thread_elements()[0], o[d].thread_elements()[1]) * inv;
}

// ---------------------------------------------------------------------------
// pf_ring_write: ring[(pos0 + m) % window] = staging[m] for the last
// min(M, window) rows. One thread per 8 bf16 (uint4). kr/vr: ring layer base.
// ---------------------------------------------------------------------------
struct PfRingParams { uint M, pos0, KV, window, first; };

kernel void pf_ring_write(device const bfloat* kst [[buffer(0)]],
                          device const bfloat* vst [[buffer(1)]],
                          device bfloat*       kr  [[buffer(2)]],
                          device bfloat*       vr  [[buffer(3)]],
                          constant PfRingParams& p [[buffer(4)]],
                          uint gid [[thread_position_in_grid]]) {
    const uint per = p.KV / 8;
    const uint m = p.first + gid / per, c = gid % per;
    if (m >= p.M) return;
    const uint slot = (p.pos0 + m) % p.window;
    ((device uint4*)(kr + (ulong)slot * p.KV))[c] = ((device const uint4*)(kst + (ulong)m * p.KV))[c];
    ((device uint4*)(vr + (ulong)slot * p.KV))[c] = ((device const uint4*)(vst + (ulong)m * p.KV))[c];
}
