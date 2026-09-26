#include "test.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

#include "coral/gpu.h"
#include "coral/kernels.h"
#include "coral/model.h"
#include "../src/model/weights.h"

using namespace coral;

namespace {
// One device + loaded gpt-oss model shared by the tests in this file.
// Intentionally leaked: the no-copy shard buffers stay registered with the
// device's residency set, so the mappings must outlive every GPU object.
struct Fixture {
    std::unique_ptr<gpu::Device> dev;
    std::unique_ptr<Model> model;
    double load_seconds = 0;
};

Fixture& fixture() {
    std::string dir = test::model_dir_or_skip();
    static Fixture* f = [&] {
        auto* x = new Fixture;
        x->dev = gpu::Device::create();
        load_kernels(*x->dev);
        auto t0 = std::chrono::steady_clock::now();
        x->model = Model::load(*x->dev, dir);
        x->load_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("        Model::load: %.1f ms\n", x->load_seconds * 1e3);
        return x;
    }();
    return *f;
}
} // namespace

CORAL_TEST(model_bind_weights) {
    auto& f = fixture();
    const ModelConfig& c = f.model->config();
    const Weights& w = model_weights(*f.model);
    CHECK_EQ(w.layers.size(), size_t(c.num_layers));
    CHECK(w.embed.dtype == DType::BF16);
    CHECK_EQ(w.embed.dim(0), int64_t(c.vocab_size));
    CHECK_EQ(w.embed.dim(1), int64_t(c.hidden_size));
    CHECK_EQ(w.lm_head.nbytes, size_t(c.vocab_size) * c.hidden_size * 2);
    CHECK_EQ(w.final_norm.shape.size(), size_t(1));

    const LayerWeights& L = w.layers[23];
    CHECK(!L.sliding);
    CHECK(w.layers[0].sliding);
    CHECK_EQ(L.wo.dim(0), int64_t(c.hidden_size));
    CHECK_EQ(L.wo.dim(1), int64_t(c.q_dim()));
    CHECK_EQ(L.sinks.dim(0), int64_t(c.num_heads));
    CHECK(L.gate_up_blocks.dtype == DType::U8);
    CHECK_EQ(L.gate_up_blocks.nbytes, size_t(32) * 5760 * 90 * 16);
    CHECK_EQ(L.down_scales.nbytes, size_t(32) * 2880 * 90);
    CHECK(L.down_bias.dtype == DType::BF16);

    // Every binding is GPU-visible, in range, and its CPU pointer aliases the buffer.
    auto check_ref = [](const TensorRef& t) {
        CHECK(t.buf.valid());
        CHECK(t.offset + t.nbytes <= t.buf.size());
        CHECK(static_cast<const uint8_t*>(t.buf.data()) + t.offset == t.cpu);
    };
    check_ref(w.embed); check_ref(w.lm_head); check_ref(w.final_norm);
    for (const auto& l : w.layers)
        for (const TensorRef* t : {&l.attn_norm, &l.wq, &l.bq, &l.wk, &l.bk, &l.wv, &l.bv, &l.wo, &l.bo, &l.sinks,
                                   &l.mlp_norm, &l.router_w, &l.router_b, &l.gate_up_blocks, &l.gate_up_scales,
                                   &l.gate_up_bias, &l.down_blocks, &l.down_scales, &l.down_bias})
            check_ref(*t);
}

CORAL_TEST(model_bind_weights_rejects_wrong_config) {
    auto& f = fixture();
    const Safetensors& st = model_tensors(*f.model);

    ModelConfig bad = f.model->config();
    bad.hidden_size = 2944;
    try { bind_weights(bad, st); CHECK(false); }
    catch (const std::runtime_error& e) { CHECK(std::strstr(e.what(), "model.embed_tokens.weight") != nullptr); }

    ModelConfig bad2 = f.model->config();
    bad2.num_experts = 16;
    try { bind_weights(bad2, st); CHECK(false); }
    catch (const std::runtime_error& e) { CHECK(std::strstr(e.what(), "model.layers.0.mlp.router.weight") != nullptr); }

    ModelConfig bad3 = f.model->config();
    bad3.num_layers = 25;
    bad3.layer_is_sliding.push_back(true);
    try { bind_weights(bad3, st); CHECK(false); }
    catch (const std::runtime_error& e) { CHECK(std::strstr(e.what(), "model.layers.24.") != nullptr); }
}

CORAL_TEST(model_new_cache_sizes) {
    auto& f = fixture();
    KVCache kv = f.model->new_cache(1024);
    CHECK_EQ(kv.capacity, 1024u);
    CHECK_EQ(kv.length, 0u);
    const size_t full = size_t(12) * 1024 * 8 * 64 * 2;   // 12 full layers
    const size_t slide = size_t(12) * 128 * 8 * 64 * 2;   // 12 sliding layers, window 128
    CHECK_EQ(kv.k_full.size(), full);
    CHECK_EQ(kv.v_full.size(), full);
    CHECK_EQ(kv.k_slide.size(), slide);
    CHECK_EQ(kv.v_slide.size(), slide);
    const uint8_t* p = kv.v_full.as<uint8_t>();
    for (size_t i = 0; i < full; i += 4093) CHECK_EQ(int(p[i]), 0);
}

CORAL_TEST(model_embed_gather_gpu_matches_cpu) {
    auto& f = fixture();
    auto& dev = *f.dev;
    const Weights& w = model_weights(*f.model);
    const int32_t toks[] = {0, 1, 199999, 200005};
    const uint32_t n = 4, H = uint32_t(w.embed.dim(1));
    auto ids = dev.alloc(sizeof(toks));
    std::memcpy(ids.data(), toks, sizeof(toks));
    auto out = dev.alloc(size_t(n) * H * 2, true);

    auto cs = dev.stream();
    cs.begin();
    encode_embed_gather(dev, cs, w.embed, ids, 0, out, 0, n);
    cs.submit_and_wait();

    for (uint32_t t = 0; t < n; ++t) {
        const uint16_t* want = w.embed.as<uint16_t>() + size_t(toks[t]) * H;
        const uint16_t* got = out.as<uint16_t>() + size_t(t) * H;
        CHECK(std::memcmp(want, got, size_t(H) * 2) == 0);
        bool nonzero = false;
        for (uint32_t i = 0; i < H; ++i) nonzero |= (want[i] & 0x7FFF) != 0;
        CHECK(nonzero || toks[t] >= 200000);  // real rows are not all-zero
    }
}

CORAL_TEST(model_decode_bytes_per_token) {
    auto& f = fixture();
    const uint64_t b1 = f.model->decode_bytes_per_token(1);
    const uint64_t b4k = f.model->decode_bytes_per_token(4096);
    std::printf("        decode bytes/token: ctx=1 %llu, ctx=4096 %llu\n", (unsigned long long)b1, (unsigned long long)b4k);
    CHECK(b1 > 3.0e9 && b1 < 4.5e9);
    CHECK(f.model->decode_bytes_per_token(2) > b1);
    CHECK(f.model->decode_bytes_per_token(200) > f.model->decode_bytes_per_token(128));
    CHECK(b4k > f.model->decode_bytes_per_token(1024));
    // ctx=4096: 12 full layers * 4096 + 12 sliding * 128 positions, 2 KiB (K+V) each.
    CHECK_EQ(b4k - b1, uint64_t(2048) * (12 * 4095 + 12 * 127));

    // One decode step: fills the cache position and produces vocab-sized logits.
    auto cs = f.dev->stream();
    KVCache kv = f.model->new_cache(16);
    gpu::Buffer logits;
    f.model->decode(cs, kv, 0, logits);
    CHECK_EQ(kv.length, 1u);
    CHECK(logits.size() >= size_t(f.model->config().vocab_size) * 4);
}
