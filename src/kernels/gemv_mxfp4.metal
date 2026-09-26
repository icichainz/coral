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
// Decode: every lane owns whole 32-element blocks (one 16-byte load + one
// scale byte). A nibble n = s|e1|e0|m is turned into an fp16 bit pattern with
// E M placed at bits 11..9 and the sign at bit 15; that half equals
// fp4(n) * 2^-14 exactly (e == 0 maps to the fp16 subnormal range, which
// Apple GPUs handle natively). The 2^14 is re-applied once to the row sum.
// Two nibbles are decoded per 32-bit op (one half2).

constant constexpr float kMxUnscale = 16384.0f;   // 2^14

inline uint mx_half2_bits(uint w) {
    return ((w & 0x00070007u) << 9) | ((w & 0x00080008u) << 12);
}

// Dot of one block (16 bytes as uint4) with 32 fp32 activations in xr[0..7].
// Returns the unscaled sum (times 2^-14, before the E8M0 scale).
inline float mx_block_dot(uint4 q, thread const float4* xr) {
    float acc = 0.0f;
    for (uint c = 0; c < 4; ++c) {
        const uint w = q[c];  // elements 8c .. 8c+7, element k at bits 4k..4k+3
        const float2 v0 = float2(as_type<half2>(mx_half2_bits(w)));        // e0, e4
        const float2 v1 = float2(as_type<half2>(mx_half2_bits(w >> 4)));   // e1, e5
        const float2 v2 = float2(as_type<half2>(mx_half2_bits(w >> 8)));   // e2, e6
        const float2 v3 = float2(as_type<half2>(mx_half2_bits(w >> 12)));  // e3, e7
        acc += dot(float4(v0.x, v1.x, v2.x, v3.x), xr[2 * c]);
        acc += dot(float4(v0.y, v1.y, v2.y, v3.y), xr[2 * c + 1]);
    }
    return acc;
}

// 16-byte block load. A16: tensor base is 16-byte aligned (one uint4 load);
// otherwise the base is only 8-byte aligned and we use two uint2 loads.
template <bool A16>
inline uint4 mx_load_block(device const uchar* p) {
    if (A16) return *(device const uint4*)p;
    device const uint2* q = (device const uint2*)p;
    const uint2 a = q[0], b = q[1];
    return uint4(a.x, a.y, b.x, b.y);
}

inline void mx_load_x(device const float* x, thread float4* xr) {
    device const float4* x4 = (device const float4*)x;
    for (uint k = 0; k < 8; ++k) xr[k] = x4[k];
}

