#include "test.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>

#include "coral/json.h"
#include "coral/tokenizer.h"

using namespace coral;

namespace coral::test {

// Loaded once and shared with test_harmony.cpp.
const Tokenizer& shared_tokenizer() {
    static std::unique_ptr<Tokenizer> tok;
    if (!tok) {
        std::string dir = model_dir_or_skip();
        if (!std::filesystem::exists(std::filesystem::path(dir) / "tokenizer.json")) SKIP("tokenizer.json not found in " + dir);
        tok = Tokenizer::load(dir);
    }
    return *tok;
}

Json load_test_data(const std::string& name) {
    auto path = std::filesystem::path(__FILE__).parent_path() / "data" / name;
    std::ifstream f(path, std::ios::binary);
    if (!f) throw Failure("cannot open " + path.string());
    std::ostringstream ss; ss << f.rdbuf();
    return Json::parse(ss.str());
}

} // namespace coral::test

using coral::test::shared_tokenizer;

namespace {
std::string ids_str(const std::vector<int32_t>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
    return s + "]";
}
} // namespace

CORAL_TEST(tokenizer_special_ids) {
    const Tokenizer& tok = shared_tokenizer();
    CHECK_EQ(tok.special("<|start|>"), 200006);
    CHECK_EQ(tok.special("<|end|>"), 200007);
    CHECK_EQ(tok.special("<|message|>"), 200008);
    CHECK_EQ(tok.special("<|channel|>"), 200005);
    CHECK_EQ(tok.special("<|constrain|>"), 200003);
    CHECK_EQ(tok.special("<|return|>"), 200002);
    CHECK_EQ(tok.special("<|call|>"), 200012);
    CHECK_EQ(tok.special("<|endoftext|>"), 199999);
    CHECK_EQ(tok.special("<|reserved_201087|>"), 201087);
    CHECK_EQ(tok.special("<|nope|>"), -1);
    CHECK_EQ(tok.special("hello"), -1);
    CHECK_EQ(tok.vocab_size(), size_t(201088));
    CHECK_EQ(std::string(tok.token_bytes(200006)), std::string("<|start|>"));
    CHECK_EQ(tok.token_bytes(-1).size(), size_t(0));
    CHECK_EQ(tok.token_bytes(201088).size(), size_t(0));
}

CORAL_TEST(tokenizer_reference_vectors) {
    const Tokenizer& tok = shared_tokenizer();
    Json doc = coral::test::load_test_data("tokenizer_vectors.json");
    size_t passed = 0, total = 0;
    std::string first_fail;
    for (const Json& c : doc["cases"].as_array()) {
        ++total;
        const std::string& text = c["text"].as_string();
        bool allow = c["allow_special"].as_bool();
        std::vector<int32_t> want;
        for (const Json& v : c["ids"].as_array()) want.push_back(int32_t(v.as_int()));
        std::vector<int32_t> got = tok.encode(text, allow);
        bool ok = got == want;
        if (ok && tok.decode(got) != text) ok = false;
        if (ok) ++passed;
        else if (first_fail.empty()) first_fail = "text=" + Json(text).dump() + "\n        want " + ids_str(want) + "\n        got  " + ids_str(got);
    }
    std::printf("        tokenizer vectors: %zu/%zu exact\n", passed, total);
    if (passed != total) throw coral::test::Failure("tokenizer mismatch; first: " + first_fail);
}

CORAL_TEST(tokenizer_specials_are_text_unless_allowed) {
    const Tokenizer& tok = shared_tokenizer();
    auto plain = tok.encode("<|start|>");
    for (int32_t id : plain) CHECK(id < 199998);
    auto sp = tok.encode("<|start|>", true);
    CHECK_EQ(sp.size(), size_t(1));
    CHECK_EQ(sp[0], 200006);
    CHECK_EQ(tok.decode(plain), std::string("<|start|>"));
    CHECK(tok.encode("").empty());
}

CORAL_TEST(tokenizer_decode_invalid_utf8) {
    const Tokenizer& tok = shared_tokenizer();
    // A single byte token that is a lone UTF-8 lead byte decodes to U+FFFD.
    std::string lead = "\xC3";
    auto ids = tok.encode("\xC3\xA9");  // "é"
    CHECK(!ids.empty());
    std::vector<int32_t> half;
    for (int32_t id = 0; id < 256 * 4 && half.empty(); ++id)
        if (tok.token_bytes(id) == lead) half.push_back(id);
    CHECK(!half.empty());
    CHECK_EQ(tok.decode(half), std::string("\xEF\xBF\xBD"));
    // Invalid input bytes round-trip through byte tokens.
    std::string bad = "ab\xFF\xFE" "cd\xC3";
    CHECK_EQ(tok.decode(tok.encode(bad)), std::string("ab\xEF\xBF\xBD\xEF\xBF\xBD" "cd\xEF\xBF\xBD"));
}

CORAL_TEST(tokenizer_throughput) {
    const Tokenizer& tok = shared_tokenizer();
    Json doc = coral::test::load_test_data("tokenizer_vectors.json");
    std::string corpus;
    const auto& cases = doc["cases"].as_array();
    for (size_t i = 0; corpus.size() < (1u << 20); ++i) {
        const Json& c = cases[i % cases.size()];
        if (!c["allow_special"].as_bool()) { corpus += c["text"].as_string(); corpus += ' '; }
    }
    // Add some prose so the mix is not dominated by fuzz strings.
    std::string prose = "The engine streams weights from unified memory; every decoded token touches about 3.8 GB. ";
    while (corpus.size() < (2u << 20)) corpus += prose;
    size_t ntok = tok.encode(corpus).size();  // warm the cache
    auto t0 = std::chrono::steady_clock::now();
    const int reps = 3;
    for (int r = 0; r < reps; ++r) ntok = tok.encode(corpus).size();
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
    std::printf("        encode: %.2f MB in %.1f ms -> %.1f MB/s, %.2f Mtok/s (%zu tokens)\n",
                corpus.size() / 1e6, s * 1e3, corpus.size() / 1e6 / s, ntok / 1e6 / s, ntok);
    CHECK(ntok > 0);
}
