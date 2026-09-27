// Multi-sequence decode (continuous batching): the batched forward step
// against each sequence decoded alone, the BatchEngine (sequences joining and
// leaving mid-flight), and aggregate throughput.
#include "test.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <memory>
#include <string>
#include <vector>

#include "coral/engine.h"
#include "coral/gpu.h"
#include "coral/json.h"
#include "coral/model.h"
#include "coral/tokenizer.h"
#include "../src/model/attention_ops.h"
#include "../src/model/batch_ops.h"
#include "../src/model/weights.h"

using namespace coral;

gpu::Device& attn_test_device();
Model& attn_test_model();
std::shared_ptr<Tokenizer> forward_test_tokenizer();

namespace {

using clk = std::chrono::steady_clock;
double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }

double pearson(const float* a, const float* b, size_t n) {
    double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    for (size_t i = 0; i < n; ++i) {
        sx += a[i]; sy += b[i]; sxx += double(a[i]) * a[i]; syy += double(b[i]) * b[i]; sxy += double(a[i]) * b[i];
    }
    const double N = double(n);
    const double cov = sxy / N - sx / N * sy / N;
    return cov / std::sqrt((sxx / N - sx / N * sx / N) * (syy / N - sy / N * sy / N));
}

// Text-like prompt of exactly n tokens; `seed` picks the topic so rows differ.
std::vector<int32_t> prompt_of(size_t n, int seed) {
    static const char* topics[] = {
        "The history of the Roman Empire begins with",
        "Photosynthesis is the process by which plants",
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n",
        "In 1969, the Apollo 11 mission",
        "The recipe for a good sourdough bread starts with",
        "Quantum entanglement describes a situation where",
        "Once upon a time in a small village by the sea,",
        "The stock market crash of 1929 was caused by",
    };
    static const char* fill[] = {
        " Scholars have long debated the causes, the consequences and the people involved.",
        " Light energy is captured by chlorophyll and converted into chemical energy.",
        "\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n",
        " Neil Armstrong and Buzz Aldrin walked on the surface while Michael Collins orbited.",
        " Flour, water, salt and a lively starter are mixed and left to rise overnight.",
        " Measuring one particle instantly tells you something about the other one.",
        " There lived a fisherman who every morning pushed his boat into the grey waves.",
        " Speculation, margin buying and a fragile banking system all played a role.",
    };
    auto tok = forward_test_tokenizer();
    const int k = seed % 8;
    std::vector<int32_t> p = tok->encode(topics[k]);
    const auto f = tok->encode(fill[k]);
    while (p.size() < n) p.insert(p.end(), f.begin(), f.end());
    p.resize(n);
    return p;
}

struct AloneRun {
    std::vector<int32_t> tokens;          // [steps + 1]: prefill argmax, then one per decode step
    std::vector<std::vector<float>> logits;   // per decode step (fp32 [V])
};

// Prefill + `steps` greedy decode steps through the single-sequence path.
AloneRun run_alone(const std::vector<int32_t>& prompt, uint32_t steps, bool keep_logits) {
    auto& dev = attn_test_device();
    Model& m = attn_test_model();
    auto cs = dev.stream();
    KVCache cache = m.new_cache(uint32_t(prompt.size()) + steps + 8);
    gpu::Buffer logits;
    AloneRun r;
    int32_t tok = m.prefill_argmax(cs, cache, prompt, logits);
    r.tokens.push_back(tok);
    const size_t V = m.config().vocab_size;
    for (uint32_t i = 0; i < steps; ++i) {
        tok = m.decode_argmax(cs, cache, tok, logits);
        r.tokens.push_back(tok);
        if (keep_logits) r.logits.emplace_back(logits.as<float>(), logits.as<float>() + V);
    }
    return r;
}

} // namespace

// B sequences with different prompts and lengths (23-310 tokens; the
// 110-token one crosses the 128-position sliding window during the run), 32
// batched greedy steps (free-running: each row continues with its own
// token) vs each sequence decoded alone. The batched kernels keep the M = 1
// per-row arithmetic, so hidden states and KV caches are bitwise equal and
// so are the logits wherever the lm_head is the GEMV (B = 2, or forced);
// the B >= 3 MMA lm_head only reorders the final dot products. The gate is
// the task's tolerance (>= 95% of steps with equal argmax, logits corr >
// 0.999 where the tokens agree); divergences, if any, are printed.
CORAL_TEST(batch_decode_matches_alone) {
    test::model_dir_or_skip();
    auto& dev = attn_test_device();
    Model& m = attn_test_model();
    const size_t V = m.config().vocab_size;
    const uint32_t steps = 32;
    const std::vector<size_t> lens = {23, 70, 110, 310, 45, 180, 15, 96};
    std::vector<std::vector<int32_t>> prompts;
    for (size_t i = 0; i < lens.size(); ++i) prompts.push_back(prompt_of(lens[i], int(i)));
    std::vector<AloneRun> alone;
    for (auto& p : prompts) alone.push_back(run_alone(p, steps, true));

    struct Case { uint32_t B; int lm_mma; };
    for (Case cs_ : {Case{4, -1}, Case{2, -1}, Case{3, -1}, Case{8, -1}, Case{8, 0}}) {
        const uint32_t B = cs_.B;
        batch_kernel_config().lm_mma = cs_.lm_mma;
        std::vector<KVCache> caches;
        std::vector<int32_t> cur;
        auto cs = dev.stream();
        gpu::Buffer logits, am;
        for (uint32_t b = 0; b < B; ++b) {
            caches.push_back(m.new_cache(uint32_t(prompts[b].size()) + steps + 8));
            cur.push_back(m.prefill_argmax(cs, caches.back(), prompts[b], logits));
            CHECK_EQ(cur.back(), alone[b].tokens[0]);
        }
        std::vector<KVCache*> ptrs;
        for (auto& c : caches) ptrs.push_back(&c);
        uint32_t agree = 0, total = 0, bitwise = 0;
        std::vector<int> first_div(B, -1);
        double min_corr = 1.0, max_rel = 0;
        for (uint32_t s = 0; s < steps; ++s) {
            m.decode_batch(cs, ptrs, cur, &logits, &am);
            for (uint32_t b = 0; b < B; ++b) {
                const int32_t t = am.as<int32_t>()[b];
                const float* lr = logits.as<float>() + size_t(b) * V;
                if (first_div[b] < 0) {
                    ++total;
                    const std::vector<float>& ref = alone[b].logits[s];
                    if (t == alone[b].tokens[s + 1]) {
                        ++agree;
                        min_corr = std::min(min_corr, pearson(lr, ref.data(), V));
                        bitwise += std::memcmp(lr, ref.data(), V * 4) == 0;
                        double mx = 0, md = 0;
                        for (size_t i = 0; i < V; ++i) {
                            mx = std::max(mx, double(std::fabs(ref[i])));
                            md = std::max(md, double(std::fabs(lr[i] - ref[i])));
                        }
                        max_rel = std::max(max_rel, md / mx);
                    } else {
                        first_div[b] = int(s);
                        std::printf("        B=%u row %u (%zu tok): first divergence at step %u: %d vs alone %d, "
                                    "corr %.6f\n", B, b, prompts[b].size(), s, t, alone[b].tokens[s + 1],
                                    pearson(lr, ref.data(), V));
                    }
                }
                cur[b] = t;
            }
        }
        std::printf("        B=%u%s: argmax agree %u/%u steps (%.1f%%), no divergence in %u/%u rows, "
                    "bitwise-identical logits %u/%u, min corr %.9f, max |d|/max|logit| %.2g\n",
                    B, cs_.lm_mma == 0 ? " (GEMV lm_head)" : B >= 3 ? " (MMA lm_head)" : "", agree, total,
                    100.0 * agree / std::max(1u, total),
                    uint32_t(std::count(first_div.begin(), first_div.end(), -1)), B, bitwise, agree, min_corr, max_rel);
        CHECK(agree * 100 >= total * 95 && total >= B * steps * 95 / 100);
        CHECK(min_corr > 0.999);
        if (B == 2 || cs_.lm_mma == 0) CHECK_EQ(bitwise, agree);
        for (uint32_t b = 0; b < B; ++b) CHECK_EQ(caches[b].length, uint32_t(prompts[b].size()) + steps);
    }
    batch_kernel_config() = BatchKernelConfig{};
}

// The mlx-lm golden greedy continuations (tests/data/logits_reference.json,
// fp32 mlx-lm, 5 prompts of 5-73 tokens) decoded as ONE batch of 5 rows: each
// row agrees with mlx exactly as far as the single-sequence path does (whose
// agreement test_forward.cpp gates: 4 of 5 cases 16/16; "The capital of
// France is" leaves mlx after one token on a router near-tie), because the
// batched tokens equal the single path's.
CORAL_TEST(batch_decode_mlx_reference) {
    test::model_dir_or_skip();
    const std::string path = "tests/data/logits_reference.json";
    std::ifstream f(path);
    if (!f) SKIP("tests/data/logits_reference.json absent");
    std::stringstream ss;
    ss << f.rdbuf();
    const Json ref = Json::parse(ss.str());
    auto& dev = attn_test_device();
    Model& m = attn_test_model();
    std::vector<std::vector<int32_t>> prompts, gold;
    std::vector<std::string> names;
    for (const Json& c : ref["cases"].as_array()) {
        std::vector<int32_t> t, g;
        for (const Json& x : c["tokens"].as_array()) t.push_back(int32_t(x.as_int()));
        for (const Json& x : c["greedy_continuation"].as_array()) g.push_back(int32_t(x.as_int()));
        prompts.push_back(t);
        gold.push_back(g);
        std::string nm = c["name"].as_string();
        for (size_t i; (i = nm.find('\n')) != std::string::npos;) nm.replace(i, 1, "\\n");
        names.push_back(nm);
    }
    const uint32_t B = uint32_t(prompts.size()), n = uint32_t(gold[0].size());
    std::vector<KVCache> caches;
    std::vector<std::vector<int32_t>> out(B);
    std::vector<int32_t> cur;
    auto cs = dev.stream();
    gpu::Buffer logits, am;
    for (uint32_t b = 0; b < B; ++b) {
        caches.push_back(m.new_cache(uint32_t(prompts[b].size()) + n + 8));
        cur.push_back(m.prefill_argmax(cs, caches.back(), prompts[b], logits));
        out[b].push_back(cur[b]);
    }
    std::vector<KVCache*> ptrs;
    for (auto& c : caches) ptrs.push_back(&c);
    for (uint32_t s = 1; s < n; ++s) {
        m.decode_batch(cs, ptrs, cur, &logits, &am);
        for (uint32_t b = 0; b < B; ++b) out[b].push_back(cur[b] = am.as<int32_t>()[b]);
    }
    uint32_t full = 0;
    for (uint32_t b = 0; b < B; ++b) {
        const AloneRun single = run_alone(prompts[b], n - 1, false);
        uint32_t same = 0;
        while (same < n && out[b][same] == gold[b][same]) ++same;
        std::printf("        %-28s batched vs mlx: first %2u/%u tokens equal; batched == single path: %s\n",
                    ("\"" + names[b].substr(0, 24) + "\"").c_str(), same, n, out[b] == single.tokens ? "yes" : "NO");
        CHECK(out[b] == single.tokens);
        CHECK_EQ(out[b][0], gold[b][0]);   // same argmax after the prompt everywhere
        full += same == n;
    }
    CHECK(full >= 4);
}

namespace {

std::shared_ptr<Model> shared_test_model() {
    return std::shared_ptr<Model>(&attn_test_model(), [](Model*) {});   // owned by the shared fixture
}

Engine& ref_engine() {
    static std::unique_ptr<Engine> e = [] {
        auto x = Engine::create(attn_test_device(), shared_test_model(), forward_test_tokenizer());
        x->warmup();
        return x;
    }();
    return *e;
}

GenerationRequest greedy_req(std::vector<int32_t> prompt, uint32_t n) {
    GenerationRequest r;
    r.prompt = std::move(prompt);
    r.max_new_tokens = n;
    r.sampling.temperature = 0;
    r.kv_capacity = uint32_t(r.prompt.size()) + n + 16;
    return r;
}

std::vector<int32_t> generate_alone(const GenerationRequest& r) {
    std::vector<int32_t> out;
    ref_engine().generate(r, [&](int32_t t) { out.push_back(t); return true; });
    return out;
}

struct Tracked {
    std::vector<int32_t> tokens;
    bool finished = false;
    FinishReason why = FinishReason::Error;
    GenerationStats st;
};

uint64_t submit_tracked(BatchEngine& be, const GenerationRequest& r, Tracked& t, uint32_t stop_after = 0) {
    return be.submit(r,
                     [&t, stop_after](int32_t tok) { t.tokens.push_back(tok); return !stop_after || t.tokens.size() < stop_after; },
                     [&t](FinishReason w, const GenerationStats& st) { t.finished = true; t.why = w; t.st = st; });
}

} // namespace

// BatchEngine with sequences joining and leaving mid-flight: 3 submitted,
// one cancelled after a few steps (engine.cancel) and one stopping itself
// (callback false), 2 more admitted while the others run; every sequence's
// tokens equal the same request generated alone (greedy). Then 5 requests
// against max_batch = 2 (queueing, pool reuse).
CORAL_TEST(batch_engine_join_leave) {
    test::model_dir_or_skip();
    const uint32_t n = 24;
    std::vector<GenerationRequest> reqs;
    const size_t lens[] = {17, 45, 115, 8, 260};
    for (int i = 0; i < 5; ++i) reqs.push_back(greedy_req(prompt_of(lens[i], i + 3), n));
    std::vector<std::vector<int32_t>> ref;
    for (auto& r : reqs) ref.push_back(generate_alone(r));

    auto be = BatchEngine::create(attn_test_device(), shared_test_model(), BatchOptions{8});
    be->warmup();
    std::vector<Tracked> t(5);
    const uint64_t id0 = submit_tracked(*be, reqs[0], t[0]);
    const uint64_t id1 = submit_tracked(*be, reqs[1], t[1]);
    submit_tracked(*be, reqs[2], t[2], /*stop_after=*/10);
    (void)id0;
    for (int i = 0; i < 5; ++i) CHECK(be->step());
    CHECK_EQ(be->active(), size_t(3));
    be->cancel(id1);
    CHECK(be->step());
    CHECK(t[1].finished && t[1].why == FinishReason::Cancelled);
    for (int i = 0; i < 3; ++i) CHECK(be->step());
    submit_tracked(*be, reqs[3], t[3]);
    submit_tracked(*be, reqs[4], t[4]);
    be->run_until_idle();
    CHECK_EQ(be->active(), size_t(0));
    for (int i = 0; i < 5; ++i) {
        CHECK(t[i].finished);
        const size_t m = t[i].tokens.size();
        size_t same = 0;
        while (same < m && same < ref[i].size() && t[i].tokens[same] == ref[i][same]) ++same;
        std::printf("        seq %d (%zu tok prompt): %zu tokens, %zu match alone, finish %d, decode steps %u, "
                    "prefill %.0f ms\n", i, reqs[i].prompt.size(), m, same, int(t[i].why), t[i].st.decode_steps,
                    t[i].st.prefill_seconds * 1e3);
        CHECK_EQ(same, m);   // identical (a prefix for the ones that left early)
    }
    CHECK(t[0].why == FinishReason::Length && t[0].tokens.size() == n);
    CHECK(t[1].tokens.size() == 6);   // 1 at admission + 5 steps, then cancelled
    CHECK(t[2].why == FinishReason::Cancelled && t[2].tokens.size() == 10);
    CHECK(t[3].tokens == ref[3] && t[4].tokens == ref[4]);

    // Queueing: 5 requests, 2 slots.
    auto be2 = BatchEngine::create(attn_test_device(), shared_test_model(), BatchOptions{2});
    std::vector<Tracked> u(5);
    for (int i = 0; i < 5; ++i) submit_tracked(*be2, reqs[i], u[i]);
    CHECK_EQ(be2->queued(), size_t(5));
    CHECK(be2->step());
    CHECK_EQ(be2->active(), size_t(2));
    CHECK_EQ(be2->queued(), size_t(3));
    be2->run_until_idle();
    for (int i = 0; i < 5; ++i) CHECK(u[i].finished && u[i].tokens == ref[i]);
}

// Sampled (non-greedy) sequences in a batch reproduce their seeded
// single-sequence runs (the rows' logits equal the single path's).
CORAL_TEST(batch_engine_sampling_seeded) {
    test::model_dir_or_skip();
    std::vector<GenerationRequest> reqs;
    for (int i = 0; i < 3; ++i) {
        GenerationRequest r = greedy_req(prompt_of(12 + 9 * i, i), 16);
        if (i != 1) {   // row 1 stays greedy: mixed batch
            r.sampling.temperature = 0.8f;
            r.sampling.top_p = 0.95f;
            r.sampling.top_k = 50;
            r.sampling.seed = 1000 + i;
        }
        reqs.push_back(r);
    }
    std::vector<std::vector<int32_t>> ref;
    for (auto& r : reqs) ref.push_back(generate_alone(r));
    auto be = BatchEngine::create(attn_test_device(), shared_test_model(), BatchOptions{2});   // B = 2: bitwise logits
    std::vector<Tracked> t(3);
    for (int i = 0; i < 3; ++i) submit_tracked(*be, reqs[i], t[i]);
    be->run_until_idle();
    for (int i = 0; i < 3; ++i) {
        std::printf("        seq %d: \"%s\"\n", i, forward_test_tokenizer()->decode(t[i].tokens).c_str());
        CHECK(t[i].tokens == ref[i]);
    }
}

// Aggregate throughput of the batched step (informational): B sequences at
// ctx ~500-700, greedy, 48 steps.
CORAL_TEST(batch_throughput) {
    test::model_dir_or_skip();
    auto& dev = attn_test_device();
    Model& m = attn_test_model();
    const uint32_t steps = 48;
    for (uint32_t B : {1u, 2u, 4u, 8u}) {
        std::vector<KVCache> caches;
        std::vector<int32_t> cur;
        auto cs = dev.stream();
        gpu::Buffer logits, am;
        for (uint32_t b = 0; b < B; ++b) {
            const auto p = prompt_of(500 + 25 * b, int(b));
            caches.push_back(m.new_cache(uint32_t(p.size()) + steps + 8));
            cur.push_back(m.prefill_argmax(cs, caches.back(), p, logits));
        }
        std::vector<KVCache*> ptrs;
        for (auto& c : caches) ptrs.push_back(&c);
        ForwardStats st;
        // warm-up step
        if (B == 1) cur[0] = m.decode_argmax(cs, caches[0], cur[0], logits);
        else m.decode_batch(cs, ptrs, cur, &logits, &am);
        const auto t0 = clk::now();
        for (uint32_t s = 0; s < steps; ++s) {
            if (B == 1) {
                cur[0] = m.decode_argmax(cs, caches[0], cur[0], logits, &st);
            } else {
                m.decode_batch(cs, ptrs, cur, &logits, &am, &st);
                for (uint32_t b = 0; b < B; ++b) cur[b] = am.as<int32_t>()[b];
            }
        }
        const double wall = since(t0);
        std::printf("        B=%u: %.2f ms/step, %.1f tok/s aggregate (%.1f tok/s per sequence), gpu %.2f ms/step, "
                    "encode %.0f us/step\n", B, wall * 1e3 / steps, B * steps / wall, steps / wall,
                    st.gpu_seconds * 1e3 / steps, st.encode_seconds * 1e6 / steps);
    }
}

// Per-stage GPU time of the batched step (CORAL_BATCH_PROFILE=1; dev aid):
// full steps with one stage enabled at a time (BatchKernelConfig::stage_mask;
// the embed + lm_head + argmax base is timed with no layer stage and
// subtracted), on real sequences at ctx ~500-700; distinct experts per layer;
// kernel shape sweeps.
CORAL_TEST(batch_zz_profile) {
    if (!std::getenv("CORAL_BATCH_PROFILE")) SKIP("set CORAL_BATCH_PROFILE=1");
    auto& dev = attn_test_device();
    Model& m = attn_test_model();
    const char* names[] = {"qkv+rope", "attention", "o_proj", "router", "gate_up", "down"};
    for (uint32_t B : {2u, 4u, 8u}) {
        std::vector<KVCache> caches;
        std::vector<int32_t> cur;
        auto cs = dev.stream();
        gpu::Buffer logits, am;
        for (uint32_t b = 0; b < B; ++b) {
            const auto p = prompt_of(500 + 25 * b, int(b));
            caches.push_back(m.new_cache(uint32_t(p.size()) + 1400));
            cur.push_back(m.prefill_argmax(cs, caches.back(), p, logits));
        }
        std::vector<KVCache*> ptrs;
        for (auto& k : caches) ptrs.push_back(&k);
        auto time = [&](uint32_t mask) {
            batch_kernel_config().stage_mask = mask;
            double best = 1e9;
            for (int it = 0; it < 4; ++it) {
                ForwardStats st;
                m.decode_batch(cs, ptrs, cur, &logits, &am, &st);
                best = std::min(best, st.gpu_seconds);
            }
            return best;
        };
        {   // distinct experts per layer across the B rows (the MoE's DRAM floor)
            const ModelConfig& c = m.config();
            const auto& Wt = model_weights(m);
            BatchScratch bs = make_batch_scratch(dev, c);
            AttnScratch as = make_attn_scratch(dev, c, 2048);
            std::vector<BatchRow> rows;
            for (auto& k : caches)
                rows.push_back(BatchRow{k.k_full.gpu_address(), k.v_full.gpu_address(), k.k_slide.gpu_address(),
                                        k.v_slide.gpu_address(), k.length, k.capacity, 0, 0});
            cs.begin();
            encode_batch_embed(dev, cs, Wt.embed, cur, bs);
            cs.submit_and_wait();
            double sum = 0;
            uint32_t mn = 99, mx = 0;
            for (uint32_t l = 0; l < c.num_layers; ++l) {
                cs.begin();
                encode_batch_attention(dev, cs, c, Wt.layers[l], kv_slot_for_layer(c, l), rows, as, bs);
                encode_batch_moe(dev, cs, c, Wt.layers[l], B, bs);
                cs.submit_and_wait();
                std::vector<int32_t> ids(bs.expert_ids.as<int32_t>(), bs.expert_ids.as<int32_t>() + B * c.experts_per_token);
                std::sort(ids.begin(), ids.end());
                const uint32_t d = uint32_t(std::unique(ids.begin(), ids.end()) - ids.begin());
                sum += d; mn = std::min(mn, d); mx = std::max(mx, d);
            }
            const double uni = c.num_experts * (1 - std::pow(1 - double(c.experts_per_token) / c.num_experts, B));
            std::printf("        B=%u distinct experts/layer: mean %.1f (min %u max %u) of %u slots; uniform-random "
                        "routing would give %.1f\n", B, sum / c.num_layers, mn, mx, B * c.experts_per_token, uni);
        }
        const double full = time(~0u), base = time(0);
        std::printf("        B=%u full step %.2f ms; embed+lm_head+argmax %.2f ms\n", B, full * 1e3, base * 1e3);
        for (uint32_t i = 0; i < 6; ++i)
            std::printf("        B=%u   %-10s %6.2f ms / 24 layers\n", B, names[i], (time(1u << i) - base) * 1e3);
        // Expert GEMM variants: GEMV grouped (bitwise) vs MMA (tiles x simdgroups).
        auto sweep = [&](const char* what, auto&& set, uint32_t stages) {
            BatchKernelConfig& k = batch_kernel_config();
            k = BatchKernelConfig{};
            set(k);
            const double b0 = time(0);
            std::printf("        B=%u   %-28s %6.2f ms (+ embed/lm_head/argmax %.2f)\n", B, what,
                        stages ? (time(stages) - b0) * 1e3 : b0 * 1e3, b0 * 1e3);
            k = BatchKernelConfig{};
        };
        for (uint32_t ch : {1u, 2u, 4u, 8u})
            if (ch <= B) {
                char nm[64];
                std::snprintf(nm, sizeof nm, "qkv chunk %u", ch);
                sweep(nm, [&](BatchKernelConfig& k) { k.qkv_chunk = ch; }, 1u);
                std::snprintf(nm, sizeof nm, "o_proj rows 4 chunk %u", ch);
                sweep(nm, [&](BatchKernelConfig& k) { k.gemv_chunk = ch; }, 4u);
            }
        sweep("lm_head GEMV", [&](BatchKernelConfig& k) { k.lm_mma = 0; }, 0);
        sweep("lm_head MMA", [&](BatchKernelConfig& k) { k.lm_mma = 1; }, 0);
        for (uint32_t cap : {1u, 2u})
            for (uint32_t gr : {4u, 8u}) {
                char nm[64];
                std::snprintf(nm, sizeof nm, "gate_up cap %u rows %u", cap, gr);
                sweep(nm, [&](BatchKernelConfig& k) { k.moe_cap = cap; k.gu_rows = gr; }, 16u);
            }
        for (uint32_t dr : {2u, 4u})
            for (uint32_t sg : {2u, 4u}) {
                char nm[64];
                std::snprintf(nm, sizeof nm, "down rows %u sg %u", dr, sg);
                sweep(nm, [&](BatchKernelConfig& k) { k.dn_rows = dr; k.dn_sg = sg; }, 32u);
            }
        batch_kernel_config() = BatchKernelConfig{};
    }
}
