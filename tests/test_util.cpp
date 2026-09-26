#include "test.h"

#include <map>
#include <vector>

#include "coral/engine.h"
#include "coral/tokenizer.h"

using namespace coral;

CORAL_TEST(utf8_streamer_splits_codepoints) {
    Utf8Streamer s;
    // "é" (C3 A9) arriving one byte at a time, then "😀" (F0 9F 98 80) split 3+1.
    CHECK_EQ(s.push("a\xC3"), std::string("a"));
    CHECK_EQ(s.push("\xA9"), std::string("\xC3\xA9"));
    CHECK_EQ(s.push("\xF0\x9F\x98"), std::string(""));
    CHECK_EQ(s.push("\x80!"), std::string("\xF0\x9F\x98\x80!"));
    CHECK_EQ(s.flush(), std::string(""));
    // Invalid: stray continuation byte and a truncated tail.
    CHECK_EQ(s.push("\x80x"), std::string("\xEF\xBF\xBDx"));
    CHECK_EQ(s.push("\xE2\x82"), std::string(""));
    CHECK_EQ(s.flush(), std::string("\xEF\xBF\xBD"));
}

CORAL_TEST(sampler_greedy_and_distribution) {
    std::vector<float> logits = {0.f, 1.f, 5.f, 2.f};
    uint64_t rng = 123;
    SamplingParams greedy; greedy.temperature = 0.f;
    CHECK_EQ(sample(logits, greedy, rng), 2);

    // top_k=1 must always pick the max regardless of temperature.
    SamplingParams k1; k1.temperature = 1.5f; k1.top_k = 1;
    for (int i = 0; i < 20; ++i) CHECK_EQ(sample(logits, k1, rng), 2);

    // With temperature 1 and no truncation, frequencies should follow softmax.
    SamplingParams t1;
    std::map<int32_t, int> counts;
    const int n = 20000;
    for (int i = 0; i < n; ++i) counts[sample(logits, t1, rng)]++;
    double z = 0; for (float l : logits) z += std::exp(l);
    for (size_t i = 0; i < logits.size(); ++i) CHECK_NEAR(double(counts[int32_t(i)]) / n, std::exp(logits[i]) / z, 0.02);

    // top_p small enough keeps only the top token.
    SamplingParams p; p.top_p = 0.5f;
    for (int i = 0; i < 20; ++i) CHECK_EQ(sample(logits, p, rng), 2);
}
