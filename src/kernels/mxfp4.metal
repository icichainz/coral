#include "common.h"

// MXFP4 dequantization to bf16: reference kernel for validating the format
// decode against the CPU implementation and for tooling. The production
// matmul kernels never materialize dequantized weights; they decode blocks in
// registers (see the roadmap in README.MD).
//
//   blocks : uint8 [rows][K/32][16]
//   scales : uint8 [rows][K/32]
//   out    : bf16  [rows][K]
// One thread per (row, block): decodes 32 elements.

struct Mxfp4DequantParams { uint rows; uint k; };

kernel void mxfp4_dequant_bf16(device const uchar*          blocks [[buffer(0)]],
                               device const uchar*          scales [[buffer(1)]],
                               device bfloat*               out    [[buffer(2)]],
                               constant Mxfp4DequantParams& p      [[buffer(3)]],
                               uint2 gid [[thread_position_in_grid]]) {
    const uint nblk = p.k / 32;
    const uint blk = gid.x, row = gid.y;
    if (blk >= nblk || row >= p.rows) return;

    const ulong bi = (ulong)row * nblk + blk;
    device const uchar* src = blocks + bi * 16;
    const float s = e8m0_to_float(scales[bi]);
    device bfloat* dst = out + (ulong)row * p.k + blk * 32;

    for (uint i = 0; i < 16; ++i) {
        float2 v = fp4x2(src[i]) * s;
        dst[2 * i]     = f2bf(v.x);
        dst[2 * i + 1] = f2bf(v.y);
    }
}
