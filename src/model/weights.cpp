#include "weights.h"

#include <sstream>
#include <stdexcept>

namespace coral {

namespace {

std::string shape_str(const std::vector<int64_t>& s) {
    std::ostringstream o;
    o << '[';
    for (size_t i = 0; i < s.size(); ++i) o << (i ? "," : "") << s[i];
    o << ']';
    return o.str();
}

// Fetch `name`, check dtype and exact shape, and return its binding.
TensorRef bind(const Safetensors& st, const std::string& name, DType dtype, std::vector<int64_t> shape) {
    if (!st.has(name)) throw std::runtime_error("weights: missing tensor '" + name + "'");
    TensorView v = st.get(name);
    const TensorInfo& t = *v.info;
    if (t.dtype != dtype)
        throw std::runtime_error("weights: tensor '" + name + "' has dtype " + dtype_name(t.dtype) +
                                 ", expected " + dtype_name(dtype));
    if (t.shape != shape)
        throw std::runtime_error("weights: tensor '" + name + "' has shape " + shape_str(t.shape) +
                                 ", expected " + shape_str(shape));
    if (v.buffer.valid() && v.buffer_offset + t.nbytes > v.buffer.size())
        throw std::runtime_error("weights: tensor '" + name + "' extends past its shard buffer");
    TensorRef r;
    r.name = name;
    r.buf = v.buffer;
    r.offset = v.buffer_offset;
    r.cpu = v.data;
    r.dtype = t.dtype;
    r.shape = std::move(shape);
    r.nbytes = t.nbytes;
    return r;
}

} // namespace

Weights bind_weights(const ModelConfig& c, const Safetensors& st) {
    if (c.expert_quant != "mxfp4")
        throw std::runtime_error("weights: unsupported expert quantization '" + c.expert_quant + "' (need mxfp4)");
    if (c.hidden_size % 32 || c.intermediate_size % 32)
        throw std::runtime_error("weights: hidden/intermediate size must be multiples of the MXFP4 block (32)");
    if (c.layer_is_sliding.size() != c.num_layers)
        throw std::runtime_error("weights: layer_is_sliding has wrong length");

    const int64_t H = c.hidden_size, I = c.intermediate_size, V = c.vocab_size, E = c.num_experts;
    const int64_t Q = c.q_dim(), KV = c.kv_dim(), NH = c.num_heads;
    constexpr DType BF = DType::BF16, U8 = DType::U8;

    Weights w;
    w.embed = bind(st, "model.embed_tokens.weight", BF, {V, H});
    if (st.has("lm_head.weight") || !c.tie_word_embeddings)
        w.lm_head = bind(st, "lm_head.weight", BF, {V, H});
    else
        w.lm_head = w.embed;
    w.final_norm = bind(st, "model.norm.weight", BF, {H});

    w.layers.resize(c.num_layers);
    for (uint32_t l = 0; l < c.num_layers; ++l) {
        LayerWeights& L = w.layers[l];
        const std::string p = "model.layers." + std::to_string(l) + ".";
        L.sliding = c.layer_is_sliding[l];
        L.attn_norm = bind(st, p + "input_layernorm.weight", BF, {H});
        L.wq = bind(st, p + "self_attn.q_proj.weight", BF, {Q, H});
        L.wk = bind(st, p + "self_attn.k_proj.weight", BF, {KV, H});
        L.wv = bind(st, p + "self_attn.v_proj.weight", BF, {KV, H});
        L.wo = bind(st, p + "self_attn.o_proj.weight", BF, {H, Q});
        if (c.attention_bias) {
            L.bq = bind(st, p + "self_attn.q_proj.bias", BF, {Q});
            L.bk = bind(st, p + "self_attn.k_proj.bias", BF, {KV});
            L.bv = bind(st, p + "self_attn.v_proj.bias", BF, {KV});
            L.bo = bind(st, p + "self_attn.o_proj.bias", BF, {H});
        }
        L.sinks = bind(st, p + "self_attn.sinks", BF, {NH});
        L.mlp_norm = bind(st, p + "post_attention_layernorm.weight", BF, {H});
        L.router_w = bind(st, p + "mlp.router.weight", BF, {E, H});
        L.router_b = bind(st, p + "mlp.router.bias", BF, {E});
        L.gate_up_blocks = bind(st, p + "mlp.experts.gate_up_proj_blocks", U8, {E, 2 * I, H / 32, 16});
        L.gate_up_scales = bind(st, p + "mlp.experts.gate_up_proj_scales", U8, {E, 2 * I, H / 32});
        L.gate_up_bias = bind(st, p + "mlp.experts.gate_up_proj_bias", BF, {E, 2 * I});
        L.down_blocks = bind(st, p + "mlp.experts.down_proj_blocks", U8, {E, H, I / 32, 16});
        L.down_scales = bind(st, p + "mlp.experts.down_proj_scales", U8, {E, H, I / 32});
        L.down_bias = bind(st, p + "mlp.experts.down_proj_bias", BF, {E, H});
    }
    return w;
}

void encode_embed_gather(gpu::Device& dev, gpu::CommandStream& cs, const TensorRef& table,
                         const gpu::Buffer& ids, size_t ids_offset,
                         const gpu::Buffer& out, size_t out_offset, uint32_t n_tokens) {
    if (!table.buf.valid()) throw std::logic_error("embed_gather: table is not bound to the GPU");
    if (table.dtype != DType::BF16 || table.shape.size() != 2)
        throw std::invalid_argument("embed_gather: table must be bf16 [vocab,dim]");
    const uint32_t dim = uint32_t(table.dim(1));
    if (ids.size() < ids_offset + size_t(n_tokens) * 4 || out.size() < out_offset + size_t(n_tokens) * dim * 2)
        throw std::out_of_range("embed_gather: ids/out buffer too small");
    struct { uint32_t n_tokens, dim; } p{n_tokens, dim};
    cs.dispatch(dev.kernel("embed_gather_bf16"),
                gpu::Args().buffer(0, table.buf, table.offset).buffer(1, ids, ids_offset)
                           .buffer(2, out, out_offset).value(3, p),
                {n_tokens}, {256});
}

} // namespace coral
