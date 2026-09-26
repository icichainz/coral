#include "test.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>

#include "coral/config.h"
#include "coral/json.h"
#include "coral/safetensors.h"

using namespace coral;

namespace {
// Write a tiny two-tensor safetensors file and return its path.
std::string write_fixture() {
    std::string path = (std::filesystem::temp_directory_path() / "coral_fixture.safetensors").string();
    Json header;
    header["a"] = Json::Object{{"dtype", "BF16"}, {"shape", Json::Array{2, 3}}, {"data_offsets", Json::Array{0, 12}}};
    header["b"] = Json::Object{{"dtype", "U8"},   {"shape", Json::Array{4}},    {"data_offsets", Json::Array{12, 16}}};
    header["__metadata__"] = Json::Object{{"format", "pt"}};
    std::string h = header.dump();
    while (h.size() % 8) h += ' ';
    uint64_t len = h.size();
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(&len), 8);
    f.write(h.data(), std::streamsize(h.size()));
    uint16_t a[6]; for (int i = 0; i < 6; ++i) a[i] = f32_to_bf16(float(i) * 0.5f);
    uint8_t b[4] = {1, 2, 3, 4};
    f.write(reinterpret_cast<const char*>(a), 12);
    f.write(reinterpret_cast<const char*>(b), 4);
    return path;
}
} // namespace

CORAL_TEST(bf16_roundtrip) {
    for (float v : {0.f, 1.f, -1.f, 0.5f, 3.0f, 65536.f, 0.001953125f, -25769803776.f}) CHECK_EQ(bf16_to_f32(f32_to_bf16(v)), v);
    CHECK_EQ(f32_to_bf16(1.00390625f), uint16_t(0x3F80));   // 1 + 2^-8 rounds to even -> 1.0
    CHECK_EQ(f32_to_bf16(1.01171875f), uint16_t(0x3F82));   // 1 + 3*2^-8 rounds up
    CHECK(std::isinf(bf16_to_f32(f32_to_bf16(INFINITY))));
}

CORAL_TEST(safetensors_fixture) {
    auto st = Safetensors::open(write_fixture());
    CHECK_EQ(st->tensors().size(), size_t(2));
    CHECK_EQ(st->total_bytes(), size_t(16));
    auto a = st->get("a");
    CHECK_EQ(a.info->dtype == DType::BF16, true);
    CHECK_EQ(a.info->numel(), int64_t(6));
    CHECK_EQ(bf16_to_f32(a.as<uint16_t>()[5]), 2.5f);
    auto b = st->get("b");
    CHECK_EQ(int(b.as<uint8_t>()[3]), 4);
    CHECK_THROWS(st->get("nope"));
}

CORAL_TEST(model_config_gptoss) {
    std::string dir = test::model_dir_or_skip();
    ModelConfig c = ModelConfig::load(dir);
    CHECK_EQ(c.model_type, std::string("gpt_oss"));
    CHECK_EQ(c.num_layers, 24u);
    CHECK_EQ(c.hidden_size, 2880u);
    CHECK_EQ(c.num_heads, 64u);
    CHECK_EQ(c.num_kv_heads, 8u);
    CHECK_EQ(c.head_dim, 64u);
    CHECK_EQ(c.num_experts, 32u);
    CHECK_EQ(c.experts_per_token, 4u);
    CHECK_EQ(c.sliding_window, 128u);
    CHECK(c.layer_is_sliding[0]);
    CHECK(!c.layer_is_sliding[1]);
    CHECK_EQ(c.expert_quant, std::string("mxfp4"));
    CHECK_NEAR(c.rope_factor, 32.0, 1e-6);
    CHECK_EQ(c.eos_token_id, 200002);
}

CORAL_TEST(model_weights_inventory) {
    std::string dir = test::model_dir_or_skip();
    ModelConfig c = ModelConfig::load(dir);
    auto st = Safetensors::open(dir);
    CHECK_EQ(st->tensors().size(), size_t(459));
    CHECK(st->has("model.embed_tokens.weight"));
    CHECK(st->has("lm_head.weight"));
    CHECK(st->has("model.norm.weight"));
    auto emb = st->get("model.embed_tokens.weight");
    CHECK_EQ(emb.info->dim(0), int64_t(c.vocab_size));
    CHECK_EQ(emb.info->dim(1), int64_t(c.hidden_size));

    // Layer 0 shapes, including the MXFP4 expert layout documented in model.h.
    auto gu = st->get("model.layers.0.mlp.experts.gate_up_proj_blocks");
    CHECK(gu.info->dtype == DType::U8);
    CHECK_EQ(gu.info->shape.size(), size_t(4));
    CHECK_EQ(gu.info->dim(0), int64_t(c.num_experts));
    CHECK_EQ(gu.info->dim(1), int64_t(2 * c.intermediate_size));
    CHECK_EQ(gu.info->dim(2), int64_t(c.hidden_size / 32));
    CHECK_EQ(gu.info->dim(3), int64_t(16));
    auto gs = st->get("model.layers.0.mlp.experts.gate_up_proj_scales");
    CHECK_EQ(gs.info->dim(2), int64_t(c.hidden_size / 32));
    auto dn = st->get("model.layers.0.mlp.experts.down_proj_blocks");
    CHECK_EQ(dn.info->dim(1), int64_t(c.hidden_size));
    CHECK_EQ(dn.info->dim(2), int64_t(c.intermediate_size / 32));
    auto q = st->get("model.layers.0.self_attn.q_proj.weight");
    CHECK_EQ(q.info->dim(0), int64_t(c.q_dim()));
    CHECK_EQ(q.info->dim(1), int64_t(c.hidden_size));
    auto sinks = st->get("model.layers.0.self_attn.sinks");
    CHECK_EQ(sinks.info->dim(0), int64_t(c.num_heads));

    // Every E8M0 scale in layer 0 should be a sane power of two for real weights.
    for (int64_t i = 0; i < gs.info->numel(); i += 4097) {
        int e = int(gs.as<uint8_t>()[i]) - 127;
        CHECK(e > -40 && e < 20);
    }
}
