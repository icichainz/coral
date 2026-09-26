// Batched prefill (M > 1): bf16 GEMM, prefill attention, grouped MXFP4 MoE,
// and the end-to-end batched prefill against the token-by-token decode path.
#include "test.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "coral/gpu.h"
#include "coral/json.h"
#include "coral/model.h"
#include "coral/tokenizer.h"
#include "coral/safetensors.h"
#include "../src/model/attention_ops.h"
#include "../src/model/moe_ops.h"
#include "../src/model/prefill_ops.h"
#include "../src/model/weights.h"

using namespace coral;

// Shared with test_gemv.cpp / test_forward.cpp (one model load per binary).
gpu::Device& attn_test_device();
Model& attn_test_model();
std::shared_ptr<Tokenizer> forward_test_tokenizer();

namespace {

using clk = std::chrono::steady_clock;

float bfv(const TensorRef& t, size_t i) { return bf16_to_f32(t.as<uint16_t>()[i]); }

std::vector<float> random_vec(size_t n, uint32_t seed, float scale = 1.0f) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0.0f, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

} // namespace

// Y = X · W^T + b for the real wq (4096 x 2880) and wo (2880 x 4096) of layer 0,
// against an fp64 CPU reference. X is staged in fp32 and W as exact bf16, with
// fp32 MMA accumulation (prefill_core.h), so the error is fp32 summation only:
// observed max |err| ~3.5e-5 at |y| ~20-40 (~1e-6 relative), vs the ~1e-2
// relative a half/bf16-input MMA would give. The bound below is 1e-4 relative.
CORAL_TEST(prefill_gemm_bf16_matches_cpu) {
    auto& dev = attn_test_device();
    Model& m = attn_test_model();
    const LayerWeights& L = model_weights(m).layers[0];
    struct Shape { const TensorRef* W; const TensorRef* b; };
    for (const Shape& sh : {Shape{&L.wq, &L.bq}, Shape{&L.wo, &L.bo}}) {
        const uint32_t N = uint32_t(sh.W->dim(0)), K = uint32_t(sh.W->dim(1));
        for (uint32_t M : {1u, 7u, 64u, 256u}) {
            const std::vector<float> xs = random_vec(size_t(M) * K, 11 + M);
            gpu::Buffer X = dev.alloc(xs.size() * 4), Y = dev.alloc(size_t(M) * N * 4);
            std::memcpy(X.data(), xs.data(), xs.size() * 4);
            // accumulate path: Y starts at 1.0
            for (size_t i = 0; i < size_t(M) * N; ++i) Y.as<float>()[i] = 1.0f;
            auto cs = dev.stream();
            cs.begin();
            GemmSegment seg{sh.W, sh.b};
            encode_gemm_bf16(dev, cs, &seg, 1, X, 0, K, Y, 0, N, M, true);
            cs.submit_and_wait();
            double max_err = 0, max_ref = 0;
            const uint16_t* w = sh.W->as<uint16_t>();
            // Check a sample of rows (all of them for small M).
            for (uint32_t r = 0; r < M; r += (M > 16 ? 5 : 1)) {
                for (uint32_t n = 0; n < N; ++n) {
                    double acc = 0;
                    for (uint32_t k = 0; k < K; ++k) acc += double(bf16_to_f32(w[size_t(n) * K + k])) * xs[size_t(r) * K + k];
                    const double ref = 1.0 + acc + bfv(*sh.b, n);
                    max_err = std::max(max_err, std::fabs(ref - Y.as<float>()[size_t(r) * N + n]));
                    max_ref = std::max(max_ref, std::fabs(ref));
                }
            }
            std::printf("        %ux%u M=%u: max|err| %.3g (max|ref| %.3g)\n", N, K, M, max_err, max_ref);
            CHECK(max_err < 1e-4 * max_ref + 1e-4);
        }
    }
}

// Throughput of the dense GEMM at prefill shapes (informational).
CORAL_TEST(prefill_gemm_bf16_throughput) {
    auto& dev = attn_test_device();
    Model& m = attn_test_model();
    const auto& W = model_weights(m);
    for (uint32_t M : {256u, 512u}) {
        const uint32_t N = 4096, K = 2880;
        gpu::Buffer X = dev.alloc(size_t(M) * K * 4, true), Y = dev.alloc(size_t(M) * N * 4, true);
        auto cs = dev.stream();
        const int reps = 24;
        double best = 1e9;
        for (int it = 0; it < 3; ++it) {
            cs.begin();
            for (int r = 0; r < reps; ++r) {
                GemmSegment seg{&W.layers[r].wq, &W.layers[r].bq};
                encode_gemm_bf16(dev, cs, &seg, 1, X, 0, K, Y, 0, N, M, false);
            }
            best = std::min(best, cs.submit_and_wait() / reps);
        }
        const double flops = 2.0 * M * N * K;
        std::printf("        M=%u %ux%u: %.1f us/GEMM = %.2f TFLOP/s\n", M, N, K, best * 1e6, flops / best / 1e12);
    }
}

namespace {

// fp32 cos/sin exactly as make_rope_table (validated against HF in test_attention.cpp).
struct Rope {
    YarnParams y;
    explicit Rope(const ModelConfig& c) : y(yarn_parameters(c)) {}
    void apply(float* v, uint32_t pos) const {   // one 64-wide head, in place
        for (int j = 0; j < 32; ++j) {
            const float f = y.inv_freq[j] * float(pos);
            const float c = float(std::cos(double(f))) * y.attention_factor;
            const float s = float(std::sin(double(f))) * y.attention_factor;
            const float a = v[j], b = v[j + 32];
            v[j] = a * c - b * s;
            v[j + 32] = b * c + a * s;
        }
    }
};

float round_bf16(float x) { return bf16_to_f32(f32_to_bf16(x)); }

} // namespace

// Prefill attention (RoPE + KV staging, flash attention, ring update, o_proj)
// for a sliding and a full layer, M = 200 positions after 50 positions
// decoded into the cache (the chunk spans more than the 128-slot window):
//   (a) the attention core against an fp64 CPU attention over the GPU's own
//       roped q/k/v (cache rows < 50 + CPU-roped chunk rows),
//   (b) the updated residual against the decode path run token by token,
//   (c) the cache afterwards (ring = last 128 positions / full rows) against
//       the decode path's cache.
CORAL_TEST(prefill_attention_matches_cpu_and_decode) {
    auto& dev = attn_test_device();
    Model& m = attn_test_model();
    const ModelConfig& c = m.config();
    const auto& Wt = model_weights(m);
    const uint32_t H = c.hidden_size, Q = c.q_dim(), KV = c.kv_dim(), W = c.sliding_window;
    const uint32_t P0 = 50, M = 200, N = P0 + M;
    const Rope rope(c);
    AttnScratch as = make_attn_scratch(dev, c, 512);
    PrefillScratch ps = make_prefill_scratch(dev, c, M);
    // Residual rows = embeddings of pseudo-random tokens (layer 0's real
    // input): random-normal rows give attention scores ~2500, where fp32
    // rounding of the score alone is ~1e-4 relative.
    std::vector<float> xs(size_t(N) * H);
    {
        std::mt19937 rng(7);
        const uint32_t V = uint32_t(Wt.embed.dim(0));
        for (uint32_t p = 0; p < N; ++p) {
            const uint32_t t = rng() % V;
            for (uint32_t i = 0; i < H; ++i) xs[size_t(p) * H + i] = bfv(Wt.embed, size_t(t) * H + i);
        }
    }

    for (uint32_t layer : {0u, 1u}) {
        const LayerWeights& L = Wt.layers[layer];
        const bool sliding = L.sliding;
        const uint32_t slot = kv_slot_for_layer(c, layer);
        auto cs = dev.stream();
        KVCache dc = m.new_cache(N + 8), pc = m.new_cache(N + 8);
        // Decode path over all N positions (reference), and over the first P0 on pc.
        std::vector<float> dec_out(size_t(N) * H);
        gpu::Buffer r = dev.alloc(size_t(H) * 4);
        for (uint32_t p = 0; p < N; ++p) {
            for (KVCache* kc : {&dc, &pc}) {
                if (kc == &pc && p >= P0) continue;
                std::memcpy(r.data(), xs.data() + size_t(p) * H, size_t(H) * 4);
                cs.begin();
                encode_attention_decode(dev, cs, c, L, layer, sliding, slot, *kc, p, r, 0, as);
                cs.submit_and_wait();
                if (kc == &dc) std::memcpy(dec_out.data() + size_t(p) * H, r.data(), size_t(H) * 4);
            }
        }
        // Snapshot of pc's rows for positions < P0 (ring slot or full row).
        const gpu::Buffer& kbuf = sliding ? pc.k_slide : pc.k_full;
        const gpu::Buffer& vbuf = sliding ? pc.v_slide : pc.v_full;
        const size_t base = size_t(slot) * (sliding ? W : pc.capacity) * KV;
        auto cache_row = [&](const gpu::Buffer& b, uint32_t pos) {
            return b.as<uint16_t>() + base + size_t(sliding ? pos % W : pos) * KV;
        };
        std::vector<std::vector<float>> Kp(N, std::vector<float>(KV)), Vp(N, std::vector<float>(KV));
        for (uint32_t p = 0; p < P0; ++p)
            for (uint32_t i = 0; i < KV; ++i) {
                Kp[p][i] = bf16_to_f32(cache_row(kbuf, p)[i]);
                Vp[p][i] = bf16_to_f32(cache_row(vbuf, p)[i]);
            }

        // Batched: norm + QKV GEMM, read the un-roped projections back, then the rest.
        std::memcpy(ps.x.data(), xs.data() + size_t(P0) * H, size_t(M) * H * 4);
        cs.begin();
        encode_rmsnorm_f32(dev, cs, L.attn_norm, c.rms_norm_eps, ps.x, 0, ps.normed, 0, M);
        const GemmSegment qkv[3] = {{&L.wq, &L.bq}, {&L.wk, &L.bk}, {&L.wv, &L.bv}};
        encode_gemm_bf16(dev, cs, qkv, 3, ps.normed, 0, H, ps.qkv, 0, Q + 2 * KV, M, false);
        cs.submit_and_wait();
        const uint32_t ldq = Q + 2 * KV;
        std::vector<float> raw(ps.qkv.as<float>(), ps.qkv.as<float>() + size_t(M) * ldq);
        cs.begin();
        encode_prefill_rope_kv(dev, cs, c, sliding, slot, pc, P0, M, as, ps);
        encode_prefill_attn_core(dev, cs, c, L, sliding, slot, pc, P0, M, ps);
        if (sliding) encode_prefill_ring_write(dev, cs, c, slot, pc, P0, M, ps);
        const GemmSegment o{&L.wo, &L.bo};
        encode_gemm_bf16(dev, cs, &o, 1, ps.attn, 0, Q, ps.x, 0, H, M, true);
        cs.submit_and_wait();

        // (a) CPU attention over the GPU projections.
        std::vector<std::vector<float>> q(M);
        double kst_err = 0;
        for (uint32_t r0 = 0; r0 < M; ++r0) {
            const float* row = raw.data() + size_t(r0) * ldq;
            q[r0].assign(row, row + Q);
            for (uint32_t h = 0; h < c.num_heads; ++h) rope.apply(q[r0].data() + h * 64, P0 + r0);
            std::vector<float> k(row + Q, row + Q + KV);
            for (uint32_t h = 0; h < c.num_kv_heads; ++h) rope.apply(k.data() + h * 64, P0 + r0);
            for (uint32_t i = 0; i < KV; ++i) {
                Kp[P0 + r0][i] = round_bf16(k[i]);
                Vp[P0 + r0][i] = round_bf16(row[Q + KV + i]);
                kst_err = std::max(kst_err, double(std::fabs(Kp[P0 + r0][i] -
                                                   bf16_to_f32(ps.kst.as<uint16_t>()[size_t(r0) * KV + i]))));
            }
        }
        double attn_err = 0, attn_max = 0;
        for (uint32_t r0 = 0; r0 < M; ++r0) {
            const uint32_t p = P0 + r0, start = sliding ? (p + 1 > W ? p + 1 - W : 0) : 0;
            for (uint32_t h = 0; h < c.num_heads; ++h) {
                const uint32_t kvh = h / 8;
                std::vector<double> s;
                double mx = bfv(L.sinks, h);
                for (uint32_t j = start; j <= p; ++j) {
                    double d = 0;
                    for (int e = 0; e < 64; ++e) d += double(q[r0][h * 64 + e]) * Kp[j][kvh * 64 + e];
                    s.push_back(d / 8.0);
                    mx = std::max(mx, s.back());
                }
                double sum = std::exp(bfv(L.sinks, h) - mx);
                for (double& x : s) { x = std::exp(x - mx); sum += x; }
                for (int e = 0; e < 64; ++e) {
                    double acc = 0;
                    for (uint32_t j = start; j <= p; ++j) acc += s[j - start] * Vp[j][kvh * 64 + e];
                    const double ref = acc / sum, got = ps.attn.as<float>()[size_t(r0) * Q + h * 64 + e];
                    attn_err = std::max(attn_err, std::fabs(ref - got));
                    attn_max = std::max(attn_max, std::fabs(ref));
                }
            }
        }
        // (b) residual vs the decode path.
        double res_err = 0, res_max = 0;
        for (uint32_t r0 = 0; r0 < M; ++r0)
            for (uint32_t i = 0; i < H; ++i) {
                const double a = dec_out[size_t(P0 + r0) * H + i], b = ps.x.as<float>()[size_t(r0) * H + i];
                res_err = std::max(res_err, std::fabs(a - b));
                res_max = std::max(res_max, std::fabs(a));
            }
        // (c) cache contents vs the decode path's cache.
        const gpu::Buffer& dk = sliding ? dc.k_slide : dc.k_full;
        const gpu::Buffer& dv = sliding ? dc.v_slide : dc.v_full;
        double cache_err = 0;
        uint32_t checked = 0;
        for (uint32_t p = sliding ? N - W : 0; p < N; ++p, ++checked) {
            const size_t off = base + size_t(sliding ? p % W : p) * KV;
            for (uint32_t i = 0; i < KV; ++i) {
                cache_err = std::max(cache_err, double(std::fabs(bf16_to_f32(kbuf.as<uint16_t>()[off + i]) -
                                                                 bf16_to_f32(dk.as<uint16_t>()[off + i]))));
                cache_err = std::max(cache_err, double(std::fabs(bf16_to_f32(vbuf.as<uint16_t>()[off + i]) -
                                                                 bf16_to_f32(dv.as<uint16_t>()[off + i]))));
            }
        }
        std::printf("        layer %u (%s): attn max|err| %.2g (max %.3g), roped k vs CPU %.2g, residual vs decode "
                    "%.2g (max %.3g), cache (%u rows) vs decode %.3g\n",
                    layer, sliding ? "sliding" : "full", attn_err, attn_max, kst_err, res_err, res_max, checked, cache_err);
        CHECK(attn_err < 1e-4 * std::max(1.0, attn_max));
        CHECK(kst_err < 0.02);             // <= ~1 bf16 ulp of |k| < 4 (fp32 rope order vs CPU)
        CHECK(res_err < 2e-3 * std::max(1.0, res_max));
        CHECK(cache_err < 0.06);           // K/V from GEMM vs GEMV projections: bf16 rounding flips only
    }
}

// MoE prefill for one layer (M = 200 rows) against the decode-path MoE
// (encode_moe_decode, validated vs CPU in test_moe.cpp) applied row by row.
CORAL_TEST(prefill_moe_matches_decode_rows) {
    auto& dev = attn_test_device();
    Model& m = attn_test_model();
    const ModelConfig& c = m.config();
    const auto& Wt = model_weights(m);
    const uint32_t H = c.hidden_size, K = c.experts_per_token, M = 200;
    PrefillScratch ps = make_prefill_scratch(dev, c, 256);
    MoeScratch ms = make_moe_scratch(dev, c);
    for (uint32_t layer : {0u, 13u}) {
        const LayerWeights& L = Wt.layers[layer];
        const std::vector<float> xs = random_vec(size_t(M) * H, 100 + layer, 2.0f);
        std::memcpy(ps.x.data(), xs.data(), xs.size() * 4);
        auto cs = dev.stream();
        cs.begin();
        encode_prefill_moe(dev, cs, c, L, M, ps);
        const double gpu = cs.submit_and_wait();
        // Decode path per row.
        gpu::Buffer r = dev.alloc(size_t(H) * 4);
        double max_err = 0, max_abs = 0;
        uint32_t id_mismatch = 0, rows_checked = 0;
        std::vector<uint32_t> hist(c.num_experts, 0);
        for (uint32_t row = 0; row < M; ++row) {
            std::memcpy(r.data(), xs.data() + size_t(row) * H, size_t(H) * 4);
            cs.begin();
            encode_moe_decode(dev, cs, c, L, r, ms);
            cs.submit_and_wait();
            bool same = true;
            for (uint32_t k = 0; k < K; ++k) {
                same = same && ms.expert_ids.as<int32_t>()[k] == ps.top_ids.as<int32_t>()[row * K + k];
                hist[ps.top_ids.as<int32_t>()[row * K + k]]++;
                CHECK_NEAR(ms.probs.as<float>()[k], ps.top_probs.as<float>()[row * K + k], 1e-5);
            }
            if (!same) { ++id_mismatch; continue; }
            ++rows_checked;
            for (uint32_t i = 0; i < H; ++i) {
                const double a = r.as<float>()[i], b = ps.x.as<float>()[size_t(row) * H + i];
                max_err = std::max(max_err, std::fabs(a - b));
                max_abs = std::max(max_abs, std::fabs(a));
            }
        }
        const auto [mn, mx] = std::minmax_element(hist.begin(), hist.end());
        std::printf("        layer %u: %u rows, expert rows min %u max %u, routing mismatches %u, max|err| %.3g "
                    "(max|x| %.3g), batched MoE %.2f ms\n",
                    layer, rows_checked, *mn, *mx, id_mismatch, max_err, max_abs, gpu * 1e3);
        CHECK(id_mismatch == 0);
        CHECK(max_err < 1e-3 * std::max(1.0, max_abs / 10));
    }
}

namespace {

struct PrefillRun {
    std::vector<float> logits;
    std::vector<int32_t> greedy;   // first = argmax of logits
    double seconds = 0;
    ForwardStats st;
};

// Prefill `tokens` with the batched (or token-by-token) path, then decode `n` greedy tokens.
PrefillRun prefill_run(const std::vector<int32_t>& tokens, bool batched, uint32_t n, uint32_t chunk = 0) {
    auto& dev = attn_test_device();
    Model& m = attn_test_model();
    model_set_batched_prefill(m, batched);
    model_set_prefill_min_tokens(m, 2);   // batched for every prompt of >= 2 tokens
    if (chunk) model_set_prefill_chunk(m, chunk);
    auto cs = dev.stream();
    KVCache cache = m.new_cache(uint32_t(tokens.size()) + n + 8);
    gpu::Buffer logits;
    PrefillRun r;
    const auto t0 = clk::now();
    int32_t tok = m.prefill_argmax(cs, cache, tokens, logits, &r.st);
    r.seconds = std::chrono::duration<double>(clk::now() - t0).count();
    model_set_batched_prefill(m, true);
    model_set_prefill_chunk(m, 1024);     // default
    model_set_prefill_min_tokens(m, 0);   // default
    r.logits.assign(logits.as<float>(), logits.as<float>() + m.config().vocab_size);
    CHECK_EQ(cache.length, uint32_t(tokens.size()));
    for (uint32_t i = 0; i < n; ++i) {
        r.greedy.push_back(tok);
        if (i + 1 < n) tok = m.decode_argmax(cs, cache, tok, logits);
    }
    return r;
}

double pearson(const std::vector<float>& a, const std::vector<float>& b) {
    double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    const double n = double(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        sx += a[i]; sy += b[i]; sxx += double(a[i]) * a[i]; syy += double(b[i]) * b[i]; sxy += double(a[i]) * b[i];
    }
    const double cov = sxy / n - sx / n * sy / n;
    return cov / std::sqrt((sxx / n - sx / n * sx / n) * (syy / n - sy / n * sy / n));
}

int32_t argmax_of(const std::vector<float>& v) { return int32_t(std::max_element(v.begin(), v.end()) - v.begin()); }

std::vector<std::vector<int32_t>> reference_prompts() {
    std::vector<std::vector<int32_t>> out;
    const std::string path = "tests/data/logits_reference.json";
    std::ifstream f(path);
    if (!f) return out;
    std::stringstream ss;
    ss << f.rdbuf();
    const Json ref = Json::parse(ss.str());
    for (const Json& c : ref["cases"].as_array()) {
        std::vector<int32_t> t;
        for (const Json& x : c["tokens"].as_array()) t.push_back(int32_t(x.as_int()));
        out.push_back(std::move(t));
    }
    return out;
}

// Deterministic long synthetic prompt (text-like token ids from the tokenizer).
std::vector<int32_t> long_prompt(size_t n) {
    auto tok = forward_test_tokenizer();
    const auto chunk = tok->encode(" In the beginning the city was small, and its people farmed the hills along the "
                                   "river. Over the centuries, merchants arrived from distant ports, bringing spices, "
                                   "silk and new ideas about law, art and the stars.");
    std::vector<int32_t> p = tok->encode("The history of the Roman Empire begins with");
    while (p.size() < n) p.insert(p.end(), chunk.begin(), chunk.end());
    p.resize(n);
    return p;
}

} // namespace

// Batched prefill vs the token-by-token decode path, end to end (24 layers,
// int8 lm_head).
//  * Every reference prompt (tests/data/logits_reference.json, 5-73 tokens):
//    last-position logits corr > 0.9999 and equal argmax; for the three
//    longest, the 16-token greedy continuation after the prefill agrees on
//    >= 15/16 tokens.
//  * Long synthetic prompts (300 / 700 tokens: one chunk spanning > 2 sliding
//    windows, several chunks): corr > 0.997, and the batched result is
//    bitwise independent of the chunking (300 tokens as 1 x 300, 3 x 100,
//    4 x 75 — the ring written by one chunk and read by the next, and a ring
//    wrapped inside one chunk, give the same bits).
//    Why not 0.9999 here: the two paths round K/V to bf16 from differently
//    ordered fp32 sums, and on long repetitive text that noise is amplified
//    by near-ties (attention and expert routing). Measured against fp32
//    mlx-lm (CORAL_PF_DUMP + an mlx script, 60-700 tokens), token and batched
//    paths are equally close to the reference — each is closer on about half
//    of the lengths, both drop to ~0.67 corr(top-64) at 280 tokens and both
//    miss mlx's argmax at 130/250/280/400 — while token vs batched stays
//    >= 0.998 (all logits). Greedy continuations on these prompts are
//    printed, not gated (they sit on such near-ties).
CORAL_TEST(prefill_batched_matches_token_path) {
    auto prompts = reference_prompts();
    if (prompts.empty()) SKIP("tests/data/logits_reference.json absent");
    std::vector<size_t> order(prompts.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t x, size_t y) { return prompts[x].size() > prompts[y].size(); });

    auto compare = [&](const std::vector<int32_t>& tokens, uint32_t n, uint32_t chunk, const PrefillRun& a,
                       const PrefillRun& b) {
        const double corr = pearson(a.logits, b.logits);
        double maxd = 0;
        for (size_t i = 0; i < a.logits.size(); ++i) maxd = std::max(maxd, double(std::fabs(a.logits[i] - b.logits[i])));
        uint32_t same = 0;
        for (uint32_t i = 0; i < n; ++i) same += a.greedy[i] == b.greedy[i];
        if (const char* d = std::getenv("CORAL_PF_DUMP")) {   // dev aid: compare both paths to an external reference
            const std::string base = std::string(d) + "/" + std::to_string(tokens.size());
            std::ofstream t(base + ".tokens.json");
            t << "[";
            for (size_t i = 0; i < tokens.size(); ++i) t << (i ? "," : "") << tokens[i];
            t << "]";
            std::ofstream(base + ".token.f32", std::ios::binary)
                .write(reinterpret_cast<const char*>(a.logits.data()), std::streamsize(a.logits.size() * 4));
            std::ofstream(base + ".batched.f32", std::ios::binary)
                .write(reinterpret_cast<const char*>(b.logits.data()), std::streamsize(b.logits.size() * 4));
        }
        std::printf("        %4zu tok%s: corr %.7f max|d| %.4f argmax %d/%d greedy %u/%u | token path %.1f tok/s, "
                    "batched %.1f tok/s (%u submits, %zu dispatches)\n",
                    tokens.size(), chunk ? (" chunk " + std::to_string(chunk)).c_str() : "", corr, maxd,
                    argmax_of(a.logits), argmax_of(b.logits), same, n, tokens.size() / a.seconds,
                    tokens.size() / b.seconds, b.st.submits, b.st.dispatches);
        return std::make_pair(corr, same);
    };

    for (size_t k = 0; k < order.size(); ++k) {
        const auto& t = prompts[order[k]];
        const uint32_t n = k < 3 ? 16 : 1;
        const PrefillRun a = prefill_run(t, false, n), b = prefill_run(t, true, n);
        const auto [corr, same] = compare(t, n, 0, a, b);
        CHECK(corr > 0.9999);
        CHECK_EQ(argmax_of(a.logits), argmax_of(b.logits));
        CHECK_EQ(a.greedy[0], b.greedy[0]);
        CHECK(same + 1 >= n);
    }

    const auto p300 = long_prompt(300), p700 = long_prompt(700);
    const PrefillRun a300 = prefill_run(p300, false, 16), b300 = prefill_run(p300, true, 16);
    CHECK(compare(p300, 16, 0, a300, b300).first > 0.997);
    for (uint32_t chunk : {100u, 75u}) {
        const PrefillRun c = prefill_run(p300, true, 16, chunk);
        compare(p300, 16, chunk, a300, c);
        CHECK(c.logits == b300.logits);   // bitwise: chunking does not change the result
        CHECK(c.greedy == b300.greedy);
    }
    const PrefillRun a700 = prefill_run(p700, false, 16), b700 = prefill_run(p700, true, 16);
    CHECK(compare(p700, 16, 0, a700, b700).first > 0.997);
}

// Prompt throughput of the batched prefill (informational; bench also
// reports it: coral bench <model> --ctx N).
CORAL_TEST(prefill_throughput) {
    for (size_t n : {512u, 2048u}) {
        const auto p = long_prompt(n);
        (void)prefill_run(p, true, 1);   // warm-up (pipelines, scratch)
        const PrefillRun r = prefill_run(p, true, 1);
        std::printf("        %zu tokens: %.1f ms = %.1f tok/s (gpu %.1f ms, encode %.2f ms, %u submits, %zu dispatches)\n",
                    n, r.seconds * 1e3, n / r.seconds, r.st.gpu_seconds * 1e3, r.st.encode_seconds * 1e3,
                    r.st.submits, r.st.dispatches);
    }
}

// Per-stage GPU time of one batched chunk (CORAL_PF_PROFILE=1; dev aid):
// each stage is recorded for all 24 layers in its own submit, on realistic
// activations (a real prompt's chunk state is replayed per stage).
CORAL_TEST(prefill_zz_profile) {
    if (!std::getenv("CORAL_PF_PROFILE")) SKIP("set CORAL_PF_PROFILE=1");
    auto& dev = attn_test_device();
    Model& m = attn_test_model();
    const ModelConfig& c = m.config();
    const auto& Wt = model_weights(m);
    const uint32_t H = c.hidden_size, Q = c.q_dim(), KV = c.kv_dim();
    for (uint32_t M : {64u, 128u, 256u, 512u, 1024u}) {
        PrefillScratch ps = make_prefill_scratch(dev, c, M);
        AttnScratch as = make_attn_scratch(dev, c, 4096);
        KVCache cache = m.new_cache(4096);
        const auto p = long_prompt(M);
        std::memcpy(ps.ids.data(), p.data(), size_t(M) * 4);
        auto cs = dev.stream();
        cs.begin();
        encode_prefill_embed(dev, cs, Wt.embed, M, ps);
        cs.submit_and_wait();
        const uint32_t pos0 = 1024;   // attention over 1024 cached positions + the chunk (full layers)
        auto time = [&](const char* name, auto&& fn) {
            double best = 1e9;
            for (int it = 0; it < 2; ++it) {
                cs.begin();
                for (uint32_t l = 0; l < c.num_layers; ++l) fn(l);
                best = std::min(best, cs.submit_and_wait());
            }
            std::printf("        M=%4u %-22s %8.2f ms / 24 layers = %6.1f us/token\n", M, name, best * 1e3, best * 1e6 / M);
            return best;
        };
        double total = 0;
        total += time("rmsnorm x2", [&](uint32_t l) {
            encode_rmsnorm_f32(dev, cs, Wt.layers[l].attn_norm, c.rms_norm_eps, ps.x, 0, ps.normed, 0, M);
            encode_rmsnorm_f32(dev, cs, Wt.layers[l].mlp_norm, c.rms_norm_eps, ps.x, 0, ps.normed, 0, M);
        });
        total += time("qkv gemm", [&](uint32_t l) {
            const LayerWeights& L = Wt.layers[l];
            const GemmSegment s3[3] = {{&L.wq, &L.bq}, {&L.wk, &L.bk}, {&L.wv, &L.bv}};
            encode_gemm_bf16(dev, cs, s3, 3, ps.normed, 0, H, ps.qkv, 0, Q + 2 * KV, M, false);
        });
        total += time("rope + kv", [&](uint32_t l) {
            encode_prefill_rope_kv(dev, cs, c, Wt.layers[l].sliding, kv_slot_for_layer(c, l), cache, pos0, M, as, ps);
        });
        total += time("attention (ctx 1024+)", [&](uint32_t l) {
            const LayerWeights& L = Wt.layers[l];
            encode_prefill_attn_core(dev, cs, c, L, L.sliding, kv_slot_for_layer(c, l), cache, pos0, M, ps);
            if (L.sliding) encode_prefill_ring_write(dev, cs, c, kv_slot_for_layer(c, l), cache, pos0, M, ps);
        });
        total += time("o_proj gemm", [&](uint32_t l) {
            const GemmSegment o{&Wt.layers[l].wo, &Wt.layers[l].bo};
            encode_gemm_bf16(dev, cs, &o, 1, ps.attn, 0, Q, ps.normed, 0, H, M, false);
        });
        // MoE on the normed embeddings (realistic routing spread).
        cs.begin();
        encode_rmsnorm_f32(dev, cs, Wt.layers[0].mlp_norm, c.rms_norm_eps, ps.x, 0, ps.normed, 0, M);
        cs.submit_and_wait();
        total += time("moe route (3)", [&](uint32_t l) { encode_prefill_moe_route(dev, cs, c, Wt.layers[l], M, ps); });
        // Experts on layer 0's routing (the tile table of the last route call), all 24 layers' weights.
        cs.begin();
        encode_prefill_moe_route(dev, cs, c, Wt.layers[0], M, ps);
        cs.submit_and_wait();
        total += time("moe experts (3)", [&](uint32_t l) { encode_prefill_moe_experts(dev, cs, c, Wt.layers[l], M, ps); });
        std::printf("        M=%4u total %.1f ms = %.1f us/token = %.0f tok/s (layers only)\n", M, total * 1e3,
                    total * 1e6 / M, M / total);
    }
}

// Short prompts: token-by-token vs batched prefill wall time by length
// (CORAL_PF_PROFILE=1; picks gptoss.cpp kMinBatched).
CORAL_TEST(prefill_zz_crossover) {
    if (!std::getenv("CORAL_PF_PROFILE")) SKIP("set CORAL_PF_PROFILE=1");
    for (size_t n : {2u, 3u, 4u, 6u, 8u, 12u, 16u, 24u, 32u, 48u, 64u}) {
        const auto p = long_prompt(n);
        double best[2] = {1e9, 1e9};
        for (int it = 0; it < 3; ++it)
            for (int b = 0; b < 2; ++b) best[b] = std::min(best[b], prefill_run(p, b == 1, 1).seconds);
        std::printf("        %3zu tokens: token path %6.1f ms, batched %6.1f ms  (%s)\n", n, best[0] * 1e3,
                    best[1] * 1e3, best[1] < best[0] ? "batched" : "token");
    }
}
