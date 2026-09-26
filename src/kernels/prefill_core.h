// Shared MSL core of the batched-prefill (M > 1) kernels: the simdgroup-MMA
// GEMM tile used by prefill_gemm.metal (dense bf16 weights) and
// prefill_moe.metal (grouped MXFP4 experts), and the 8x8 fragment helpers
// used by prefill_attn.metal. Emitted after common.h, ahead of every .metal
// file (tools/embed_shaders.cpp).
//
//   Y[M][N] = X[M][K] · W[N][K]^T (+ bias[N]) (+ Y)
//
// X is the fp32 activation matrix, W a row-major weight ([N][K], output
// feature n = row n, the layout the decode GEMVs read). The multiply runs on
// simdgroup_matrix<float,8,8> with fp32 accumulation. A is staged in
// threadgroup memory as fp32, B as bf16 (bf16 weights and fp4 * 2^(s-127)
// MXFP4 values are exact in bf16), and fragments are filled element-wise
// through thread_elements() with the conversion to fp32 — so the only
// rounding is the fp32 accumulation itself. (On M1/M2 fp32 and fp16 MMA have
// the same throughput; fp32 operands cost only threadgroup memory.)
//
// Tile: BM x 64 outputs per threadgroup of 4 simdgroups (2 x 2, each
// BM/2 x 32 = BM/16 x 4 accumulators), BM in {16, 32, 64} (small row counts —
// short prompts, sparsely routed experts — waste less on empty rows), K
// stepped by PG_BK = 32 (one MXFP4 block). The next K
// step's global loads are issued into registers before the MMAs on the
// current tile (register double buffering). All fixed-trip loops carry
// PG_UNROLL: without full unrolling the fragment arrays go to private memory
// and the kernel runs ~9x slower (measured: 0.8 vs 7.2 TFLOP/s).
// Measured on M2 Max: 7.2 TFLOP/s for M=512, N=4096, K=2880.

#define PG_BN 64u
#define PG_BK 32u
#define PG_LDA (PG_BK + 4u)           // A tile [BM][PG_LDA] fp32
#define PG_LDB (PG_BN + 8u)           // B^T tile [PG_BK][PG_LDB] bf16 (row = k, column = output n)
#define PG_THREADS 128u
#define PG_FN (PG_BN / 16u)           // 8x8 accumulator columns per simdgroup

typedef simdgroup_matrix<float, 8, 8> pg_frag;
#define PG_UNROLL _Pragma("clang loop unroll(full)")

// Lane -> (row, column pair) inside an 8x8 simdgroup matrix (thread_elements()
// holds elements (fr, fc) and (fr, fc + 1)).
inline ushort2 pg_frag_coord(ushort lane) {
    const ushort qid = lane / 4;
    return ushort2((qid & 2) * 2 + (lane % 2) * 2, (qid & 4) + ((lane / 2) % 4));   // (col, row)
}

// MMA over one staged K step; FM = BM / 16 accumulator rows per simdgroup.
template <uint BM>
inline void pg_mma_step(threadgroup const float* As, threadgroup const bfloat* Bs,
                        thread pg_frag (&c)[BM / 16][PG_FN], uint sm, uint sn, ushort2 fc) {
    constexpr uint FM = BM / 16;
    PG_UNROLL for (uint kk = 0; kk < PG_BK; kk += 8) {
        pg_frag a[FM], b[PG_FN];
        PG_UNROLL for (uint i = 0; i < FM; ++i) {
            threadgroup const float* ap = As + (sm * (BM / 2) + i * 8 + fc.y) * PG_LDA + kk + fc.x;
            a[i].thread_elements()[0] = ap[0];
            a[i].thread_elements()[1] = ap[1];
        }
        PG_UNROLL for (uint j = 0; j < PG_FN; ++j) {
            threadgroup const bfloat* bp = Bs + (kk + fc.y) * PG_LDB + sn * (PG_BN / 2) + j * 8 + fc.x;
            b[j].thread_elements()[0] = float(bp[0]);
            b[j].thread_elements()[1] = float(bp[1]);
        }
        PG_UNROLL for (uint i = 0; i < FM; ++i)
            PG_UNROLL for (uint j = 0; j < PG_FN; ++j)
                simdgroup_multiply_accumulate(c[i][j], a[i], b[j], c[i][j]);
    }
}

template <uint BM>
inline void pg_zero(thread pg_frag (&c)[BM / 16][PG_FN]) {
    PG_UNROLL for (uint i = 0; i < BM / 16; ++i)
        PG_UNROLL for (uint j = 0; j < PG_FN; ++j) c[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
}

// ---- operand loaders (register stage) -----------------------------------------
// A (fp32 activations): BM rows x PG_BK floats. Thread t owns row
// t / (128/BM) and BM/4 floats at column (t % (128/BM)) * BM/4.
template <uint BM> struct PgA {
    enum : uint { TPR = PG_THREADS / BM,   // threads per row
                  NV = BM / 16 };          // float4 per thread
    float4 v[NV];
    static uint row(uint t) { return t / TPR; }
    void load(device const float* rowp, bool ok, uint k0, uint t) {
        device const float4* p = (device const float4*)(rowp + k0 + (t % TPR) * (NV * 4));
        PG_UNROLL for (uint i = 0; i < NV; ++i) v[i] = ok ? p[i] : float4(0.0f);
    }
    void store(threadgroup float* As, uint t) const {
        threadgroup float* d = As + (t / TPR) * PG_LDA + (t % TPR) * (NV * 4);
        PG_UNROLL for (uint i = 0; i < NV; ++i) *(threadgroup float4*)(d + 4 * i) = v[i];
    }
};

// B (bf16 weights [N][K], 8-byte aligned rows): thread t owns W row t/2,
// PG_BK/2 bf16 at (t%2)*PG_BK/2; stored transposed into Bs[k][n].
#define PG_BV (PG_BK / 8u)   // uint2 per thread
struct PgBRegs { uint2 u[PG_BV]; };

inline void pg_load_b_bf16(thread PgBRegs& r, device const bfloat* row, uint k0, uint half_) {
    device const uint2* p = (device const uint2*)(row + k0 + half_ * (PG_BK / 2));
    PG_UNROLL for (uint i = 0; i < PG_BV; ++i) r.u[i] = p[i];
}
inline void pg_store_b_bf16(threadgroup bfloat* Bs, thread const PgBRegs& r, uint t) {
    threadgroup ushort* d = (threadgroup ushort*)Bs + (t % 2) * (PG_BK / 2) * PG_LDB + t / 2;
    PG_UNROLL for (uint i = 0; i < PG_BV; ++i) {
        d[(4 * i + 0) * PG_LDB] = ushort(r.u[i].x & 0xFFFFu);
        d[(4 * i + 1) * PG_LDB] = ushort(r.u[i].x >> 16);
        d[(4 * i + 2) * PG_LDB] = ushort(r.u[i].y & 0xFFFFu);
        d[(4 * i + 3) * PG_LDB] = ushort(r.u[i].y >> 16);
    }
}
