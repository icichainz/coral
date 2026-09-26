#include "common.h"

// Decode attention (M=1) with learned sinks, GQA, bf16 KV cache.
//
// Semantics (HF modeling_gpt_oss.eager_attention_forward):
//   s_j   = (q . k_j) / sqrt(head_dim)       for the attended positions j
//   probs = softmax([s_0 .. s_{n-1}, sink_h]); the sink's mass is discarded
//   out   = sum_j probs_j v_j
//
// The host passes the KV rows to attend as a contiguous run of cache slots
// [0, n): for full layers n = pos+1 (slot = position); for sliding layers the
// 128-slot ring holds exactly the window [pos-127, pos] once pos >= 127, and
// slots [0, pos] before that, so n = min(pos+1, 128). Softmax is order
// invariant, so the ring order does not matter.
//
// Split-K ("flash decoding"): grid = (kv_heads, n_splits). A threadgroup of 8
// simdgroups handles one kv head and one chunk of positions; simdgroup g owns
// query head kvh*8+g (all 8 share the K/V tiles staged in threadgroup memory).
// Each writes a partial (max m, sum l, acc[64]); attn_decode_combine merges the
// splits and the sink.
//
// Layout constants: head_dim 64, 8 query heads per kv head (host checks).

#define ATTN_HD 64u
#define ATTN_GROUP 8u
#define ATTN_BLOCK 32u
#define ATTN_PSTRIDE (ATTN_HD + 2u)   // floats per partial record: m, l, acc[64]

inline float4 attn_bf4(uint2 u) {
    return float4(as_type<float>(u.x << 16), as_type<float>(u.x & 0xFFFF0000u),
                  as_type<float>(u.y << 16), as_type<float>(u.y & 0xFFFF0000u));
}

struct AttnParams {
    uint  n;          // positions (slots) to attend: [0, n)
    uint  chunk;      // positions per split (multiple of 32)
    uint  kv_heads;
    uint  max_splits; // partial record stride per head
    float scale;      // 1/sqrt(head_dim)
};

kernel void attn_decode_partial(device const float*   q    [[buffer(0)]],   // fp32 [heads][64]
                                device const bfloat*  K    [[buffer(1)]],   // bf16 [slots][kv_heads][64]
                                device const bfloat*  V    [[buffer(2)]],
                                device float*         part [[buffer(3)]],   // [heads][max_splits][66]
                                constant AttnParams&  p    [[buffer(4)]],
                                uint2 tg   [[threadgroup_position_in_grid]],
                                uint  tid  [[thread_index_in_threadgroup]],
                                uint  lane [[thread_index_in_simdgroup]],
                                uint  sid  [[simdgroup_index_in_threadgroup]]) {
    threadgroup float qs[ATTN_GROUP][ATTN_HD];
    threadgroup float ks[ATTN_BLOCK][ATTN_HD + 1];   // +1: conflict-free column reads
    threadgroup float vs[ATTN_BLOCK][ATTN_HD];

    const uint kvh = tg.x, split = tg.y;
    const uint head = kvh * ATTN_GROUP + sid;
    const uint c0 = split * p.chunk;
    const uint c1 = min(c0 + p.chunk, p.n);

    // Stage the 8 query heads of this kv group, pre-scaled.
    {
        device const float* qg = q + kvh * ATTN_GROUP * ATTN_HD;
        for (uint i = tid; i < ATTN_GROUP * ATTN_HD; i += 256)
            qs[i / ATTN_HD][i % ATTN_HD] = qg[i] * p.scale;
    }

    float m = -INFINITY, l = 0.0f, acc0 = 0.0f, acc1 = 0.0f;
    const ulong row_stride = (ulong)p.kv_heads * ATTN_HD;   // elements between positions

    // Loader mapping: 256 threads cover 32 positions x 8 segments of 8 bf16.
    const uint lpos = tid / 8, lseg = (tid % 8) * 8;

    for (uint b0 = c0; b0 < c1; b0 += ATTN_BLOCK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);   // previous tile consumed (and qs staged)
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

        // Scores: lane j <-> position b0 + j.
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

    device float* rec = part + ((ulong)head * p.max_splits + split) * ATTN_PSTRIDE;
    if (lane == 0) { rec[0] = m; rec[1] = l; }
    rec[2 + lane] = acc0;
    rec[2 + lane + 32] = acc1;
}

// Merge split partials with the sink: out[h][d] = sum_s acc_s[d] e^{m_s-M} / (e^{sink-M} + sum_s l_s e^{m_s-M}).
// Grid: heads threadgroups of 64 threads.
struct AttnCombineParams { uint n_splits; uint max_splits; };

kernel void attn_decode_combine(device const float*  part  [[buffer(0)]],
                                device const bfloat* sinks [[buffer(1)]],
                                device float*        out   [[buffer(2)]],
                                constant AttnCombineParams& p [[buffer(3)]],
                                uint head [[threadgroup_position_in_grid]],
                                uint d    [[thread_position_in_threadgroup]]) {
    device const float* rec = part + (ulong)head * p.max_splits * ATTN_PSTRIDE;
    const float sink = bf2f(sinks[head]);
    float M = sink;
    for (uint s = 0; s < p.n_splits; ++s) M = max(M, rec[s * ATTN_PSTRIDE]);
    float L = exp(sink - M), o = 0.0f;
    for (uint s = 0; s < p.n_splits; ++s) {
        const float w = exp(rec[s * ATTN_PSTRIDE] - M);
        L += rec[s * ATTN_PSTRIDE + 1] * w;
        o += rec[s * ATTN_PSTRIDE + 2 + d] * w;
    }
    out[head * ATTN_HD + d] = o / L;
}
