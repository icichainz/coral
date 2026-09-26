#include "common.h"
#include "prefill_core.h"

// Dense bf16 GEMM for batched prefill; the tile core lives in prefill_core.h.

// ---------------------------------------------------------------------------
// Dense bf16 GEMM with up to three weight segments along N (one dispatch for
// the fused QKV projection): columns [0, n1) use W0/B0, [n1, n2) W1/B1,
// [n2, N) W2/B2, each segment's rows indexed from its own start. Segment
// boundaries must be multiples of PG_BN; K a multiple of PG_BK.
//   X : fp32 [M][ldx]  (16-byte aligned rows)     Y : fp32 [M][ldy]
//   flags: 1 = add bias, 2 = accumulate into Y
// Grid: (N / PG_BN, ceil(M / BM)) threadgroups of PG_THREADS; BM = 16, 32
// or 64 (pf_gemm_bf16_m16 / _m32 / _m64).
// ---------------------------------------------------------------------------
struct PgGemmParams { uint M, N, K, ldx, ldy, n1, n2, flags; };

template <uint BM>
kernel void pf_gemm_bf16(device const float*   X  [[buffer(0)]],
                         device const bfloat*  W0 [[buffer(1)]],
                         device const bfloat*  W1 [[buffer(2)]],
                         device const bfloat*  W2 [[buffer(3)]],
                         device const bfloat*  B0 [[buffer(4)]],
                         device const bfloat*  B1 [[buffer(5)]],
                         device const bfloat*  B2 [[buffer(6)]],
                         device float*         Y  [[buffer(7)]],
                         constant PgGemmParams& p [[buffer(8)]],
                         uint2  tg   [[threadgroup_position_in_grid]],
                         uint   t    [[thread_index_in_threadgroup]],
                         uint   sg   [[simdgroup_index_in_threadgroup]],
                         uint   lane [[thread_index_in_simdgroup]]) {
    threadgroup float  As[BM * PG_LDA];
    threadgroup bfloat Bs[PG_BK * PG_LDB];

    const uint n0 = tg.x * PG_BN, m0 = tg.y * BM;
    device const bfloat* W; device const bfloat* B; uint nb;
    if (n0 < p.n1)      { W = W0; B = B0; nb = 0; }
    else if (n0 < p.n2) { W = W1; B = B1; nb = p.n1; }
    else                { W = W2; B = B2; nb = p.n2; }

    const uint sm = sg / 2, sn = sg % 2;
    const ushort2 fc = pg_frag_coord(ushort(lane));
    const uint ar_row = PgA<BM>::row(t);
    const bool a_ok = m0 + ar_row < p.M;
    device const float* arow = X + (ulong)min(m0 + ar_row, p.M - 1) * p.ldx;
    device const bfloat* brow = W + (ulong)(n0 - nb + t / 2) * p.K;

    pg_frag c[BM / 16][PG_FN];
    pg_zero<BM>(c);

    PgA<BM> ar; PgBRegs br;
    ar.load(arow, a_ok, 0, t);
    pg_load_b_bf16(br, brow, 0, t % 2);
    for (uint k0 = 0; k0 < p.K; k0 += PG_BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ar.store(As, t);
        pg_store_b_bf16(Bs, br, t);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k0 + PG_BK < p.K) {
            ar.load(arow, a_ok, k0 + PG_BK, t);
            pg_load_b_bf16(br, brow, k0 + PG_BK, t % 2);
        }
        pg_mma_step<BM>(As, Bs, c, sm, sn, fc);
    }

    // Epilogue straight from the accumulators: lane holds (row, col..col+1) of each 8x8.
    PG_UNROLL for (uint j = 0; j < PG_FN; ++j) {
        const uint n = n0 + sn * (PG_BN / 2) + j * 8 + fc.x;
        float2 bias = 0.0f;
        if (p.flags & 1u) bias = float2(float(B[n - nb]), float(B[n - nb + 1]));
        PG_UNROLL for (uint i = 0; i < BM / 16; ++i) {
            const uint m = m0 + sm * (BM / 2) + i * 8 + fc.y;
            if (m >= p.M) continue;
            device float2* yp = (device float2*)(Y + (ulong)m * p.ldy + n);
            float2 v = float2(c[i][j].thread_elements()[0], c[i][j].thread_elements()[1]) + bias;
            if (p.flags & 2u) v += *yp;
            *yp = v;
        }
    }
}

#define PF_GEMM_INST(NAME, BM) \
template [[host_name(NAME)]] kernel void pf_gemm_bf16<BM>( \
    device const float*, device const bfloat*, device const bfloat*, device const bfloat*, device const bfloat*, \
    device const bfloat*, device const bfloat*, device float*, constant PgGemmParams&, uint2, uint, uint, uint);
PF_GEMM_INST("pf_gemm_bf16_m16", 16)
PF_GEMM_INST("pf_gemm_bf16_m32", 32)
PF_GEMM_INST("pf_gemm_bf16_m64", 64)
