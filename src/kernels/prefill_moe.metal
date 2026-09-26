#include "common.h"

// Batched-prefill MoE (M > 1 rows): router + top-k per row, a deterministic
// counting sort of the (row, slot) pairs by expert, grouped MXFP4 GEMMs over
// the per-expert row tiles, and a fixed-order reduction into the residual.
// Semantics are those of moe.metal (HF GptOssTopKRouter / GptOssExperts).
//
// Uses the GEMM core of prefill_gemm.metal (sorted before this file): MXFP4
// blocks are decoded once per (tile, K step) into the staged bf16 B tile
// (fp4 * 2^(s-127) is exact in bf16) and reused by every row of the tile.
//
// Tile table (written by pf_moe_sort, read by the grouped GEMMs):
//   tiles[0]     = (n_tiles, 0, 0, 0)
//   tiles[1 + t] = (expert, first list index, rows (<= BM), 0)
// BM (16, 32 or 64) is chosen by the host per chunk and must match between
// pf_moe_sort and the pf_moe_gate_up_m* / pf_moe_down_m* dispatch.
// list[i] = row << 8 | slot, grouped by expert, stable (row-major) order.

// Top-K over E <= 256 logits in threadgroup memory, one simdgroup (all lanes).
// Same algorithm and tie-breaking as moe_select_topk (moe.metal).
inline void pf_select_topk(threadgroup const float* logits, uint E, uint K, uint lane,
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
// Router: one threadgroup (256 threads) per row of the already-normalized
// activations. Simdgroup g computes the logits of experts g, g+8, ...
//   x : fp32 [M][H]   w : bf16 [E][H]   b : bf16 [E]
//   ids, probs : [M][K]
// ---------------------------------------------------------------------------
struct PfRouterParams { uint H, E, K; };

kernel void pf_router(device const float*   x     [[buffer(0)]],
                      device const bfloat*  w     [[buffer(1)]],
                      device const bfloat*  b     [[buffer(2)]],
                      device int*           ids   [[buffer(3)]],
                      device float*         probs [[buffer(4)]],
                      constant PfRouterParams& p  [[buffer(5)]],
                      uint row  [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg   [[simdgroup_index_in_threadgroup]],
                      uint nsg  [[simdgroups_per_threadgroup]]) {
    threadgroup float logits[256];
    device const float4* x4 = (device const float4*)(x + (ulong)row * p.H);
    const uint n4 = p.H / 4;
    for (uint e = sg; e < p.E; e += nsg) {
        device const uint2* w2 = (device const uint2*)(w + (ulong)e * p.H);
        float acc = 0.0f;
        for (uint j = lane; j < n4; j += 32) acc += dot(gemv_bf4(w2[j]), x4[j]);
        acc = simd_sum(acc);
        if (lane == 0) logits[e] = acc + float(b[e]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        int my_id; float my_prob;
        pf_select_topk(logits, p.E, p.K, lane, my_id, my_prob);
        if (lane < p.K) { ids[row * p.K + lane] = my_id; probs[row * p.K + lane] = my_prob; }
    }
}

// ---------------------------------------------------------------------------
// Counting sort by expert, one threadgroup of 1024 threads (32 simdgroups).
// Simdgroup g owns experts g, g+32, ...: it scans the n = M*K entries in
// order, 32 at a time, and ranks its matches with a ballot, so the list order
// is deterministic (row-major within an expert).
//   ids : int32 [n]    list : uint [n]    tiles : uint4 [1 + max_tiles]
// ---------------------------------------------------------------------------
struct PfSortParams { uint n, E, K, max_tiles, bm; };

kernel void pf_moe_sort(device const int*   ids   [[buffer(0)]],
                        device uint*        list  [[buffer(1)]],
                        device uint4*       tiles [[buffer(2)]],
                        constant PfSortParams& p  [[buffer(3)]],
                        uint tid  [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sg   [[simdgroup_index_in_threadgroup]],
                        uint nsg  [[simdgroups_per_threadgroup]]) {
    threadgroup uint cnt[256];
    threadgroup uint off[256];
    for (uint e = sg; e < p.E; e += nsg) {
        uint c = 0;
        for (uint i0 = 0; i0 < p.n; i0 += 32) {
            const uint i = i0 + lane;
            c += (i < p.n && uint(ids[i]) == e) ? 1u : 0u;
        }
        c = simd_sum(c);
        if (lane == 0) cnt[e] = c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint o = 0, t = 0;
        for (uint e = 0; e < p.E; ++e) {
            off[e] = o;
            for (uint s = 0; s < cnt[e]; s += p.bm) {
                if (t < p.max_tiles) tiles[1 + t] = uint4(e, o + s, min(p.bm, cnt[e] - s), 0);
                ++t;
            }
            o += cnt[e];
        }
        tiles[0] = uint4(min(t, p.max_tiles), 0, 0, 0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = sg; e < p.E; e += nsg) {
        uint base = off[e];
        for (uint i0 = 0; i0 < p.n; i0 += 32) {
            const uint i = i0 + lane;
            const bool match = i < p.n && uint(ids[i]) == e;
            const ulong bal = ulong(simd_vote::vote_t(simd_ballot(match)));
            if (match) {
                const uint rank = popcount(bal & ((1ul << lane) - 1ul));
                list[base + rank] = ((i / p.K) << 8) | (i % p.K);
            }
            base += popcount(bal);
        }
    }
}

// ---------------------------------------------------------------------------
// MXFP4 B loader: thread t owns W row t/2 of the tile, half (t%2) of the one
// 32-element block of this K step (8 bytes = 16 nibbles) + the block scale.
// ---------------------------------------------------------------------------
struct PgMxRegs { uint2 q; uint s; };

inline void pg_load_b_mx(thread PgMxRegs& r, device const uchar* blocks, device const uchar* scales,
                         ulong grow, uint nblk, uint k0, uint half_) {
    const uint blk = k0 / 32;
    r.q = *(device const uint2*)(blocks + (grow * nblk + blk) * 16 + half_ * 8);
    r.s = scales[grow * nblk + blk];
}
// Elements half*16 + 8w + e of the block come from nibble e of word w;
// fp4 * 2^(s-127) is exact in bf16. (The ALU-only fp16 bit trick of
// gemv_mxfp4.metal measured 2-3% slower here than the constant LUT.)
inline void pg_store_b_mx(threadgroup bfloat* Bs, thread const PgMxRegs& r, uint t) {
    threadgroup bfloat* d = Bs + (t % 2) * 16 * PG_LDB + t / 2;
    const float sc = e8m0_to_float(r.s);
    PG_UNROLL for (uint w = 0; w < 2; ++w) {
        const uint word = w ? r.q.y : r.q.x;
        PG_UNROLL for (uint e = 0; e < 8; ++e)
            d[(w * 8 + e) * PG_LDB] = bfloat(kFp4Lut[(word >> (4 * e)) & 0xFu] * sc);
    }
}

// ---------------------------------------------------------------------------
// Grouped MXFP4 GEMM over the tile table. MODE 0 = gate_up (+bias, clamped
// SwiGLU -> h in list order), MODE 1 = down (+bias, * routing prob -> y slots).
//   gate_up: A = x rows gathered through list (fp32 [M][H]), W = W_gu[e] [2I][H]
//   down   : A = h rows [list start, +rows) (fp32 [M*K][I]), W = W_d[e] [H][I]
// Grid: (N / PG_BN, max_tiles) threadgroups of PG_THREADS; threadgroups past
// the tile count exit.
// ---------------------------------------------------------------------------
struct PfMoeParams { uint N, Kd, lda, K, H; float limit, alpha; uint I; };

template <uint MODE, uint BM>
kernel void pf_moe_gemm(device const float*   A      [[buffer(0)]],
                        device const uchar*   blocks [[buffer(1)]],
                        device const uchar*   scales [[buffer(2)]],
                        device const bfloat*  bias   [[buffer(3)]],
                        device const uint*    list   [[buffer(4)]],
                        device const uint4*   tiles  [[buffer(5)]],
                        device const float*   probs  [[buffer(6)]],
                        device float*         out    [[buffer(7)]],
                        constant PfMoeParams& p      [[buffer(8)]],
                        uint2 tg   [[threadgroup_position_in_grid]],
                        uint  t    [[thread_index_in_threadgroup]],
                        uint  sg   [[simdgroup_index_in_threadgroup]],
                        uint  lane [[thread_index_in_simdgroup]]) {
    if (tg.y >= tiles[0].x) return;
    const uint4 tile = tiles[1 + tg.y];
    const uint e = tile.x, l0 = tile.y, rows = tile.z;

    threadgroup float  As[BM * PG_LDA];
    threadgroup bfloat Bs[PG_BK * PG_LDB];

    const uint n0 = tg.x * PG_BN;
    const uint sm = sg / 2, sn = sg % 2;
    const ushort2 fc = pg_frag_coord(ushort(lane));
    const uint ar_row = PgA<BM>::row(t);
    const bool a_ok = ar_row < rows;
    const uint li = l0 + min(ar_row, rows - 1);
    device const float* arow = MODE == 0 ? A + (ulong)(list[li] >> 8) * p.lda : A + (ulong)li * p.lda;
    const uint nblk = p.Kd / 32;
    const ulong grow = (ulong)e * p.N + n0 + t / 2;

    pg_frag c[BM / 16][PG_FN];
    pg_zero<BM>(c);

    PgA<BM> ar; PgMxRegs br;
    ar.load(arow, a_ok, 0, t);
    pg_load_b_mx(br, blocks, scales, grow, nblk, 0, t % 2);
    for (uint k0 = 0; k0 < p.Kd; k0 += PG_BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ar.store(As, t);
        pg_store_b_mx(Bs, br, t);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k0 + PG_BK < p.Kd) {
            ar.load(arow, a_ok, k0 + PG_BK, t);
            pg_load_b_mx(br, blocks, scales, grow, nblk, k0 + PG_BK, t % 2);
        }
        pg_mma_step<BM>(As, Bs, c, sm, sn, fc);
    }

    PG_UNROLL for (uint j = 0; j < PG_FN; ++j) {
        const uint n = n0 + sn * (PG_BN / 2) + j * 8 + fc.x;   // even
        device const bfloat* bp = bias + (ulong)e * p.N + n;
        const float2 bb = float2(float(bp[0]), float(bp[1]));
        PG_UNROLL for (uint i = 0; i < BM / 16; ++i) {
            const uint r = sm * (BM / 2) + i * 8 + fc.y;
            if (r >= rows) continue;
            const float2 v = float2(c[i][j].thread_elements()[0], c[i][j].thread_elements()[1]) + bb;
            if (MODE == 0) {
                // (even, odd) = (gate, up) of intermediate feature n/2.
                const float g = min(v.x, p.limit);
                const float u = clamp(v.y, -p.limit, p.limit);
                const float glu = g / (1.0f + exp(-p.alpha * g));
                out[(ulong)(l0 + r) * p.I + n / 2] = (u + 1.0f) * glu;
            } else {
                const uint ent = list[l0 + r];
                const uint row = ent >> 8, slot = ent & 0xFFu;
                const float pr = probs[row * p.K + slot];
                *(device float2*)(out + ((ulong)row * p.K + slot) * p.H + n) = v * pr;
            }
        }
    }
}

#define PF_MOE_INST(NAME, MODE, BM) \
template [[host_name(NAME)]] kernel void pf_moe_gemm<MODE, BM>( \
    device const float*, device const uchar*, device const uchar*, device const bfloat*, device const uint*, \
    device const uint4*, device const float*, device float*, constant PfMoeParams&, uint2, uint, uint, uint);
PF_MOE_INST("pf_moe_gate_up_m16", 0, 16)
PF_MOE_INST("pf_moe_gate_up_m32", 0, 32)
PF_MOE_INST("pf_moe_gate_up_m64", 0, 64)
PF_MOE_INST("pf_moe_down_m16", 1, 16)
PF_MOE_INST("pf_moe_down_m32", 1, 32)
PF_MOE_INST("pf_moe_down_m64", 1, 64)

// ---------------------------------------------------------------------------
// x[m] += y[m][0] + ... + y[m][K-1] (fixed order). One thread per float4.
// ---------------------------------------------------------------------------
struct PfReduceParams { uint M, H, K; };

kernel void pf_moe_reduce(device float*        x [[buffer(0)]],
                          device const float*  y [[buffer(1)]],
                          constant PfReduceParams& p [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    const uint h4 = p.H / 4;
    if (gid >= p.M * h4) return;
    const uint m = gid / h4, c = gid % h4;
    device const float4* ym = (device const float4*)(y + (ulong)m * p.K * p.H) + c;
    float4 acc = ym[0];
    for (uint k = 1; k < p.K; ++k) acc += ym[(ulong)k * h4];
    ((device float4*)x)[gid] += acc;
}

// ---------------------------------------------------------------------------
// Embedding rows: x[m] = float(table[ids[m]]). Grid: (ceil(H/4/256), M).
// ---------------------------------------------------------------------------
struct PfEmbedParams { uint H; };

kernel void pf_embed(device const bfloat* table [[buffer(0)]],
                     device const int*    ids   [[buffer(1)]],
                     device float*        x     [[buffer(2)]],
                     constant PfEmbedParams& p  [[buffer(3)]],
                     uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x * 4, m = gid.y;
    if (i >= p.H) return;
    device const bfloat* src = table + (ulong)uint(ids[m]) * p.H + i;
    *(device float4*)(x + (ulong)m * p.H + i) = float4(src[0], src[1], src[2], src[3]);
}
