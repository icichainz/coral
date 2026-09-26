#include "moe_ops.h"

#include <stdexcept>
#include <string>

namespace coral {

namespace {

// Must match src/kernels/moe.metal.
constexpr uint32_t kRouterThreads = 256;  // MOE_ROUTER_THREADS
constexpr uint32_t kRouterChunks = 4;     // MOE_ROUTER_CHUNKS
constexpr uint32_t kMaxE = 256;           // moe_select_topk limit
constexpr float kSwigluAlpha = 1.702f;

struct RouterParams { uint32_t H, E; float eps; uint32_t do_norm, K; };
struct GateUpParams { uint32_t H, I, E, K; float limit, alpha; uint32_t forced; };
struct DownParams { uint32_t H, I, K; };

void require_gpu(const TensorRef& t, const char* what) {
    if (!t.buf.valid()) throw std::logic_error(std::string("moe: tensor not bound to the GPU: ") + what + " (" + t.name + ")");
}

void check_config(const ModelConfig& c) {
    if (c.hidden_size % 32 || c.intermediate_size % 32 || c.hidden_size / 4 > kRouterThreads * kRouterChunks)
        throw std::invalid_argument("moe: unsupported hidden/intermediate size");
    if (c.num_experts == 0 || c.num_experts > kMaxE || c.experts_per_token == 0 || c.experts_per_token > 32 ||
        c.experts_per_token > c.num_experts)
        throw std::invalid_argument("moe: unsupported expert count / top-k");
}

void check_scratch(const ModelConfig& c, const MoeScratch& s) {
    const size_t H = c.hidden_size, I = c.intermediate_size, K = c.experts_per_token, E = c.num_experts;
    if (!s.normed.valid() || !s.logits.valid() || !s.expert_ids.valid() || !s.probs.valid() || !s.h.valid() ||
        s.normed.size() < H * 4 || s.logits.size() < E * 4 || s.expert_ids.size() < K * 4 ||
        s.probs.size() < K * 4 || s.h.size() < K * I * 4 || !s.counter.valid())
        throw std::invalid_argument("moe: scratch buffers missing or too small (use make_moe_scratch)");
}

} // namespace

MoeScratch make_moe_scratch(gpu::Device& dev, const ModelConfig& c) {
    MoeScratch s;
    s.normed = dev.alloc(size_t(c.hidden_size) * 4, true);
    s.logits = dev.alloc(size_t(c.num_experts) * 4, true);
    s.expert_ids = dev.alloc(size_t(c.experts_per_token) * 4, true);
    s.probs = dev.alloc(size_t(c.experts_per_token) * 4, true);
    s.h = dev.alloc(size_t(c.experts_per_token) * c.intermediate_size * 4, true);
    s.counter = dev.alloc(16, true);
    return s;
}

void encode_moe_router(gpu::Device& dev, gpu::CommandStream& cs, const ModelConfig& c,
                       const TensorRef& norm_w, const TensorRef& router_w, const TensorRef& router_b,
                       const gpu::Buffer& x, MoeScratch& s, bool apply_norm) {
    check_config(c);
    check_scratch(c, s);
    if (x.size() < size_t(c.hidden_size) * 4) throw std::invalid_argument("moe: input buffer too small");
    require_gpu(router_w, "router_w");
    require_gpu(router_b, "router_b");
    if (apply_norm) require_gpu(norm_w, "mlp_norm");
    const TensorRef& nw = apply_norm ? norm_w : router_w;   // unused slot when !apply_norm
    RouterParams p{c.hidden_size, c.num_experts, c.rms_norm_eps, apply_norm ? 1u : 0u, c.experts_per_token};
    cs.dispatch(dev.kernel("moe_router"),
                gpu::Args().buffer(0, x).buffer(1, nw.buf, nw.offset)
                           .buffer(2, router_w.buf, router_w.offset).buffer(3, router_b.buf, router_b.offset)
                           .buffer(4, s.normed).buffer(5, s.logits).value(6, p)
                           .buffer(7, s.expert_ids).buffer(8, s.probs).buffer(9, s.counter),
                {c.num_experts}, {kRouterThreads});
}

void encode_moe_gate_up(gpu::Device& dev, gpu::CommandStream& cs, const ModelConfig& c,
                        const LayerWeights& L, MoeScratch& s, bool forced_ids) {
    check_config(c);
    check_scratch(c, s);
    require_gpu(L.gate_up_blocks, "gate_up_blocks");
    require_gpu(L.gate_up_scales, "gate_up_scales");
    require_gpu(L.gate_up_bias, "gate_up_bias");
    GateUpParams p{c.hidden_size, c.intermediate_size, c.num_experts, c.experts_per_token,
                   c.swiglu_limit, kSwigluAlpha, forced_ids ? 1u : 0u};
    const MoeKernelConfig& kc = moe_kernel_config();
    const uint32_t per_tg = kc.gu_sg * kc.gu_pairs;
    const uint32_t ntg = (c.intermediate_size + per_tg - 1) / per_tg;
    cs.dispatch(dev.kernel("mx_gemv_gate_up_swiglu_p" + std::to_string(kc.gu_pairs)),
                gpu::Args().buffer(0, s.normed)
                           .buffer(1, L.gate_up_blocks.buf, L.gate_up_blocks.offset)
                           .buffer(2, L.gate_up_scales.buf, L.gate_up_scales.offset)
                           .buffer(3, L.gate_up_bias.buf, L.gate_up_bias.offset)
                           .buffer(4, s.logits).buffer(5, s.expert_ids).buffer(6, s.probs)
                           .buffer(7, s.h).value(8, p),
                {ntg, c.experts_per_token}, {kc.gu_sg * 32});
}

void encode_moe_down(gpu::Device& dev, gpu::CommandStream& cs, const ModelConfig& c,
                     const LayerWeights& L, const gpu::Buffer& residual, MoeScratch& s) {
    check_config(c);
    check_scratch(c, s);
    require_gpu(L.down_blocks, "down_blocks");
    require_gpu(L.down_scales, "down_scales");
    require_gpu(L.down_bias, "down_bias");
    if (residual.size() < size_t(c.hidden_size) * 4) throw std::invalid_argument("moe: residual buffer too small");
    DownParams p{c.hidden_size, c.intermediate_size, c.experts_per_token};
    const MoeKernelConfig& kc = moe_kernel_config();
    const gpu::Args args = gpu::Args().buffer(0, s.h)
                           .buffer(1, L.down_blocks.buf, L.down_blocks.offset)
                           .buffer(2, L.down_scales.buf, L.down_scales.offset)
                           .buffer(3, L.down_bias.buf, L.down_bias.offset)
                           .buffer(4, s.expert_ids).buffer(5, s.probs)
                           .buffer(6, residual).value(7, p);
    const uint32_t per_tg = kc.dn_sg * kc.dn_rows;
    const uint32_t ntg = (c.hidden_size + per_tg - 1) / per_tg;
    cs.dispatch(dev.kernel("mx_gemv_down_r" + std::to_string(kc.dn_rows)),
                args, {ntg}, {kc.dn_sg * 32});
}

void encode_moe_decode(gpu::Device& dev, gpu::CommandStream& cs, const ModelConfig& c,
                       const LayerWeights& L, const gpu::Buffer& residual, MoeScratch& s) {
    encode_moe_router(dev, cs, c, L.mlp_norm, L.router_w, L.router_b, residual, s, true);
    encode_moe_gate_up(dev, cs, c, L, s);
    encode_moe_down(dev, cs, c, L, residual, s);
}

MoeKernelConfig& moe_kernel_config() {
    static MoeKernelConfig k;
    return k;
}

uint64_t moe_decode_bytes_per_layer(const ModelConfig& c) {
    const uint64_t H = c.hidden_size, I = c.intermediate_size, K = c.experts_per_token, E = c.num_experts;
    const uint64_t gu = 2 * I * (H / 32) * 17 + 2 * I * 2;   // blocks + scales + bias
    const uint64_t dn = H * (I / 32) * 17 + H * 2;
    const uint64_t router = E * H * 2 + E * 2 + H * 2;     // router weight + bias, mlp_norm
    return K * (gu + dn) + router;
}

} // namespace coral
