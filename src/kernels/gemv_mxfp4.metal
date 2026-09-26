#include "common.h"

// MXFP4 expert GEMVs for the MoE decode path (M = 1), weights decoded in
// registers straight from the checkpoint layout (no repack):
//
//   blocks : uint8 [E][rows][K/32][16]   32 E2M1 nibbles per block, low nibble = even element
//   scales : uint8 [E][rows][K/32]       E8M0, value 2^(s-127)
//   bias   : bf16  [E][rows]
//
// Layout facts (verified against HF transformers, modeling_gpt_oss.py and
// integrations/mxfp4.py::convert_moe_packed_tensors): dequantizing gives
// W[e] of shape [rows, K] which HF then transposes to [K, rows] and uses as
// `x @ W`. So stored row r is output feature r and the GEMV is y = W·x, for
// both gate_up (rows = 2I, K = H) and down (rows = H, K = I). In gate_up the
// output features are interleaved: gate = gu[0::2], up = gu[1::2], i.e.
// stored rows 2i (gate) and 2i+1 (up) are adjacent in memory.
//
// Decode (kernels in moe.metal, 4 lanes per block, see mx_quarter_dot): a
// nibble n = s|e1|e0|m is turned into an fp16 bit pattern with
// E M placed at bits 11..9 and the sign at bit 15; that half equals
// fp4(n) * 2^-14 exactly (e == 0 maps to the fp16 subnormal range, which
// Apple GPUs handle natively). The 2^14 is re-applied once to the row sum.
// Two nibbles are decoded per 32-bit op (one half2).

constant constexpr float kMxUnscale = 16384.0f;   // 2^14

inline uint mx_half2_bits(uint w) {
    return ((w & 0x00070007u) << 9) | ((w & 0x00080008u) << 12);
}
