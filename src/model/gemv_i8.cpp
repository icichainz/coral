#include "gemv_i8.h"

#include <stdexcept>
#include <string>

namespace coral {

namespace {
constexpr uint32_t kSimdgroups = 8;     // GEMV_SG in gemv_bf16.metal
constexpr uint32_t kDefaultRows = 4;    // tuned on M2 Max (tests/test_gemv.cpp)
struct Params { uint32_t rows, K; float eps; uint32_t pad; };
struct QParams { uint32_t rows, K; };
void require(bool ok, const std::string& what) { if (!ok) throw std::invalid_argument("gemv_i8: " + what); }
} // namespace

QuantI8 quantize_rows_i8(gpu::Device& dev, const TensorRef& W) {
    require(W.buf.valid() && W.dtype == DType::BF16 && W.shape.size() == 2, "W must be a GPU-bound bf16 matrix");
    require((W.buf.gpu_address() + W.offset) % 8 == 0, "W must be 8-byte aligned");
    QuantI8 out;
    out.rows = uint32_t(W.dim(0));
    out.K = uint32_t(W.dim(1));
    require(out.K % 16 == 0, "K must be a multiple of 16");
    out.q = dev.alloc(size_t(out.rows) * out.K);
    out.s = dev.alloc(size_t(out.rows) * 4);
    auto cs = dev.stream();
    cs.begin();
    cs.dispatch(dev.kernel("quant_rows_i8"),
                gpu::Args().buffer(0, W.buf, W.offset).buffer(1, out.q).buffer(2, out.s).value(3, QParams{out.rows, out.K}),
                {(out.rows + kSimdgroups - 1) / kSimdgroups}, {kSimdgroups * 32});
    cs.submit_and_wait();
    return out;
}

void encode_gemv_i8(gpu::Device& dev, gpu::CommandStream& cs, const QuantI8& W,
                    const gpu::Buffer& x, size_t x_offset, const gpu::Buffer& y, size_t y_offset,
                    const TensorRef* norm, float eps, uint32_t R) {
    require(W.valid(), "weights not quantized");
    require(x.valid() && (x.gpu_address() + x_offset) % 16 == 0 && x_offset + size_t(W.K) * 4 <= x.size(), "bad x");
    require(y.valid() && (y.gpu_address() + y_offset) % 4 == 0 && y_offset + size_t(W.rows) * 4 <= y.size(), "bad y");
    if (norm) {
        require(norm->buf.valid() && norm->dtype == DType::BF16 && norm->nbytes >= size_t(W.K) * 2 &&
                (norm->buf.gpu_address() + norm->offset) % 8 == 0, "bad norm weight");
    }
    if (!R) R = kDefaultRows;
    require(R == 1 || R == 2 || R == 4 || R == 8, "rows_per_simdgroup must be 1, 2, 4 or 8");
    const std::string name = std::string(norm ? "gemv_i8_norm_r" : "gemv_i8_r") + std::to_string(R);
    const uint32_t per_tg = R * kSimdgroups;
    const Params p{W.rows, W.K, eps, 0};   // Args::value keeps a pointer until dispatch
    gpu::Args a;
    a.buffer(0, W.q).buffer(1, W.s).buffer(2, x, x_offset).buffer(3, y, y_offset);
    if (norm) a.buffer(4, norm->buf, norm->offset); else a.buffer(4, W.s);
    a.value(5, p);
    cs.dispatch(dev.kernel(name), a, {(W.rows + per_tg - 1) / per_tg}, {kSimdgroups * 32});
}

} // namespace coral
