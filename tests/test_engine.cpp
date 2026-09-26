// Engine: greedy generation through the full model, Harmony render -> parse,
// stop handling and stats.
#include "test.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "coral/engine.h"
#include "coral/gpu.h"
#include "coral/harmony.h"
#include "coral/json.h"
#include "coral/model.h"
#include "coral/tokenizer.h"

using namespace coral;

gpu::Device& attn_test_device();
Model& attn_test_model();
std::shared_ptr<Tokenizer> forward_test_tokenizer();

namespace {
Engine& engine() {
    static std::unique_ptr<Engine> e = [] {
        // The model is owned by the shared test fixture: non-owning shared_ptr.
        std::shared_ptr<Model> m(&attn_test_model(), [](Model*) {});
        auto x = Engine::create(attn_test_device(), m, forward_test_tokenizer());
        x->warmup();
        return x;
    }();
    return *e;
}
} // namespace

CORAL_TEST(engine_harmony_hello) {
    Engine& e = engine();
    const Tokenizer& tok = e.tokenizer();
    harmony::RenderOptions ro;
    ro.reasoning = harmony::ReasoningEffort::Low;
    GenerationRequest req;
    req.prompt = harmony::render(tok, {{harmony::Role::User, "Say hello in one word."}}, ro);
    req.max_new_tokens = 200;
    req.sampling.temperature = 0;
    req.stop_tokens = {tok.special("<|return|>"), tok.special("<|call|>")};
    CHECK(req.stop_tokens[0] == 200002 && req.stop_tokens[1] == 200012);

    harmony::Parser parser(tok);
    std::string final_text, analysis;
    std::vector<int32_t> out;
    bool saw_stop = false;
    GenerationStats st;
    const FinishReason why = e.generate(req, [&](int32_t t) {
        out.push_back(t);
        for (const auto& ev : parser.push(t)) {
            if (ev.kind == harmony::Event::Kind::Text) (ev.channel == "final" ? final_text : analysis) += ev.text;
            if (ev.kind == harmony::Event::Kind::Stop) saw_stop = true;
        }
        return true;
    }, &st);
    std::printf("        analysis: \"%s\"\n        final: \"%s\"\n", analysis.c_str(), final_text.c_str());
    std::printf("        prompt %u tok, prefill %.1f ms (%.1f ms/tok); %u generated, %u decode steps in %.1f ms "
                "= %.1f tok/s (gpu %.2f ms/tok, encode %.0f us/tok, %zu dispatches/tok)\n",
                st.prompt_tokens, st.prefill_seconds * 1e3, st.prefill_seconds * 1e3 / st.prompt_tokens,
                st.generated_tokens, st.decode_steps, st.decode_seconds * 1e3, st.tokens_per_second(),
                st.decode_forward.gpu_seconds * 1e3 / std::max(1u, st.decode_steps),
                st.decode_forward.encode_seconds * 1e6 / std::max(1u, st.decode_steps),
                st.decode_forward.dispatches / std::max<size_t>(1, st.decode_steps));
    CHECK(why == FinishReason::Stop);
    CHECK(saw_stop && parser.finished());
    CHECK_EQ(out.back(), 200002);
    CHECK(!final_text.empty());
    CHECK_EQ(st.generated_tokens, uint32_t(out.size()));
    CHECK_EQ(st.decode_steps + 1, st.generated_tokens);
    CHECK_EQ(st.prompt_tokens, uint32_t(req.prompt.size()));
    CHECK(st.decode_forward.dispatches == size_t(st.decode_steps) * 148);   // 146 + GPU argmax (2)
}

CORAL_TEST(engine_matches_reference_continuation) {
    // The engine's greedy path reproduces the mlx-lm greedy continuation of the
    // reference Harmony prompt (first 8 tokens), and max_new_tokens -> Length.
    const std::string path = "tests/data/logits_reference.json";
    if (!std::filesystem::exists(path)) SKIP("tests/data/logits_reference.json absent");
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    const Json ref = Json::parse(ss.str());
    const Json* hc = nullptr;
    for (const Json& c : ref["cases"].as_array()) if (c["name"].as_string() == "harmony_hello") hc = &c;
    if (!hc) SKIP("no harmony_hello case");
    GenerationRequest req;
    for (const Json& t : (*hc)["tokens"].as_array()) req.prompt.push_back(int32_t(t.as_int()));
    req.max_new_tokens = 8;
    req.sampling.temperature = 0;
    std::vector<int32_t> out;
    GenerationStats st;
    const FinishReason why = engine().generate(req, [&](int32_t t) { out.push_back(t); return true; }, &st);
    CHECK(why == FinishReason::Length);
    CHECK_EQ(out.size(), size_t(8));
    for (size_t i = 0; i < 8; ++i) CHECK_EQ(out[i], int32_t((*hc)["greedy_continuation"][i].as_int()));
    // Cancellation from the callback.
    out.clear();
    CHECK(engine().generate(req, [&](int32_t t) { out.push_back(t); return out.size() < 3; }) == FinishReason::Cancelled);
    CHECK_EQ(out.size(), size_t(3));
}

CORAL_TEST(engine_sampling_seeded) {
    // Temperature sampling (CPU sampler over the in-place logits) is
    // reproducible for a fixed seed.
    Engine& e = engine();
    GenerationRequest req;
    req.prompt = e.tokenizer().encode("Once upon a time");
    req.max_new_tokens = 12;
    req.sampling.temperature = 0.8f;
    req.sampling.top_p = 0.95f;
    req.sampling.top_k = 50;
    req.sampling.seed = 1234;
    std::vector<int32_t> a, b;
    e.generate(req, [&](int32_t t) { a.push_back(t); return true; });
    e.generate(req, [&](int32_t t) { b.push_back(t); return true; });
    std::printf("        sampled: \"%s\"\n", e.tokenizer().decode(a).c_str());
    CHECK(a == b);
    CHECK_EQ(a.size(), size_t(12));
}
