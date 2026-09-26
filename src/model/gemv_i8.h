// Internal: int8 weight-only quantized GEMV (used for the lm_head).
//
// quantize_rows_i8 converts a bf16 [rows, K] tensor once, on the GPU, into a
// private int8 copy with one fp32 scale per row (symmetric absmax/127, round
// to nearest). encode_gemv_i8 then computes
//   y[r] = s[r] * sum_k q[r,k] * xn[k],   xn = norm ? rmsnorm(x) * norm_w : x
// (kernels in src/kernels/gemv_int8.metal).
#pragma once

#include <cstdint>

#include "coral/gpu.h"
#include "weights.h"

namespace coral {

struct QuantI8 {
    gpu::Buffer q;        // int8 [rows][K]
    gpu::Buffer s;        // fp32 [rows]
    uint32_t rows = 0, K = 0;
    bool valid() const { return q.valid(); }
    uint64_t bytes() const { return uint64_t(rows) * K + uint64_t(rows) * 4; }
};

// Synchronous (one submit on its own stream). W: bf16 [rows, K], K % 16 == 0.
QuantI8 quantize_rows_i8(gpu::Device& dev, const TensorRef& W);

// rows_per_simdgroup: 0 = default (tuned), else 1, 2, 4 or 8.
void encode_gemv_i8(gpu::Device& dev, gpu::CommandStream& cs, const QuantI8& W,
                    const gpu::Buffer& x, size_t x_offset, const gpu::Buffer& y, size_t y_offset,
                    const TensorRef* norm = nullptr, float eps = 1e-5f, uint32_t rows_per_simdgroup = 0);

} // namespace coral
