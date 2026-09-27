#include "coral/engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>

#include "coral/kernels.h"

namespace coral {

// ---------------------------------------------------------------------------
// CPU sampler (non-greedy path; a GPU top-k pre-pass is roadmap step 7)
// ---------------------------------------------------------------------------
namespace {
inline uint64_t splitmix64(uint64_t& s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
inline float uniform01(uint64_t& s) { return float((splitmix64(s) >> 40) * (1.0 / 16777216.0)); }
} // namespace

int32_t sample(std::span<const float> logits, const SamplingParams& p, uint64_t& rng) {
    const size_t n = logits.size();
    if (n == 0) throw std::invalid_argument("sample: empty logits");
    if (p.temperature <= 0.0f) {
        return int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
    }

    // Candidate set: top-k by logit (or all).
    std::vector<int32_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    size_t k = (p.top_k > 0 && size_t(p.top_k) < n) ? size_t(p.top_k) : n;
    if (k < n) {
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
        idx.resize(k);
    } else {
        std::sort(idx.begin(), idx.end(), [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
    }

    // Softmax with temperature over candidates.
    const float inv_t = 1.0f / p.temperature;
    const float mx = logits[idx[0]] * inv_t;
    std::vector<float> probs(idx.size());
    double sum = 0.0;
    for (size_t i = 0; i < idx.size(); ++i) { probs[i] = std::exp(logits[idx[i]] * inv_t - mx); sum += probs[i]; }
    for (auto& v : probs) v = float(v / sum);

    // Nucleus truncation.
    size_t keep = idx.size();
    if (p.top_p < 1.0f) {
        double c = 0.0;
        for (size_t i = 0; i < idx.size(); ++i) { c += probs[i]; if (c >= p.top_p) { keep = i + 1; break; } }
    }
    double total = 0.0;
    for (size_t i = 0; i < keep; ++i) total += probs[i];
    float r = uniform01(rng) * float(total);
    for (size_t i = 0; i < keep; ++i) { r -= probs[i]; if (r <= 0.0f) return idx[i]; }
    return idx[keep - 1];
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------
namespace {

class SingleSequenceEngine final : public Engine {
public:
    SingleSequenceEngine(gpu::Device& dev, std::shared_ptr<Model> model, std::shared_ptr<Tokenizer> tok)
        : dev_(dev), model_(std::move(model)), tok_(std::move(tok)), cs_(dev_.stream()) {
        if (!model_) throw std::invalid_argument("Engine::create: null model");
        // Compile the embedded kernels unless the caller already did.
        try { (void)dev_.kernel("embed_gather_f32"); } catch (const std::exception&) { load_kernels(dev_); }
    }

    const Model& model() const override { return *model_; }
    const Tokenizer& tokenizer() const override {
        if (!tok_) throw std::logic_error("Engine: no tokenizer");
        return *tok_;
    }

    void warmup() override {
        // One throwaway step builds every pipeline state, allocates the
        // model's scratch and touches all weight pages the decode path reads.
        KVCache tmp = model_->new_cache(8);
        (void)model_->decode_argmax(cs_, tmp, 0, logits_, nullptr);
    }

    std::unique_ptr<BatchEngine> make_batch_engine(const BatchOptions& opt) override {
        return BatchEngine::create(dev_, model_, opt);
    }

    FinishReason generate(const GenerationRequest& req, const TokenCallback& on_token,
                          GenerationStats* stats_out) override {
        using clk = std::chrono::steady_clock;
        auto secs = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); };
        GenerationStats st;
        st.prompt_tokens = uint32_t(req.prompt.size());
        if (req.prompt.empty()) throw std::invalid_argument("generate: empty prompt");
        if (req.max_new_tokens == 0) { if (stats_out) *stats_out = st; return FinishReason::Length; }

        const ModelConfig& cfg = model_->config();
        uint32_t cap = req.kv_capacity;
        if (cap == 0) {
            const uint64_t need = uint64_t(req.prompt.size()) + req.max_new_tokens;
            cap = uint32_t(std::min<uint64_t>(std::max<uint64_t>(kDefaultKvCapacity, need), cfg.max_position_embeddings));
        }
        if (req.prompt.size() >= cap)
            throw std::out_of_range("generate: prompt (" + std::to_string(req.prompt.size()) +
                                    " tokens) does not fit the KV cache (" + std::to_string(cap) + ")");
        if (cache_.capacity != cap) cache_ = model_->new_cache(cap);   // reused across calls otherwise
        cache_.length = 0;   // stale rows beyond `length` are never attended

        const SamplingParams& sp = req.sampling;
        const bool greedy = sp.temperature <= 0.0f && sp.repetition_penalty == 1.0f;
        uint64_t rng = sp.seed ? sp.seed : (uint64_t(std::random_device{}()) << 32) ^ std::random_device{}();
        const std::unordered_set<int32_t> stops(req.stop_tokens.begin(), req.stop_tokens.end());
        std::vector<int32_t> seen;   // for the repetition penalty
        if (sp.repetition_penalty != 1.0f) seen.assign(req.prompt.begin(), req.prompt.end());

        auto pick = [&]() -> int32_t {
            float* l = logits_.as<float>();
            const size_t V = cfg.vocab_size;
            if (sp.repetition_penalty != 1.0f) {
                // HF convention: shrink positive logits, push negative ones further down.
                std::vector<int32_t> u(seen);
                std::sort(u.begin(), u.end());
                u.erase(std::unique(u.begin(), u.end()), u.end());
                for (int32_t t : u)
                    if (t >= 0 && size_t(t) < V) l[t] = l[t] > 0 ? l[t] / sp.repetition_penalty : l[t] * sp.repetition_penalty;
            }
            return sample(std::span<const float>(l, V), sp, rng);
        };

        // Prefill.
        const auto t0 = clk::now();
        int32_t tok = greedy ? model_->prefill_argmax(cs_, cache_, req.prompt, logits_, &st.prefill_forward)
                             : (model_->prefill(cs_, cache_, req.prompt, logits_, &st.prefill_forward), pick());
        const auto t1 = clk::now();
        st.prefill_seconds = secs(t0, t1);

        // Decode loop: deliver `tok`, then compute the next one.
        FinishReason why = FinishReason::Length;
        for (;;) {
            st.generated_tokens++;
            if (sp.repetition_penalty != 1.0f) seen.push_back(tok);
            const bool keep = on_token ? on_token(tok) : true;
            if (stops.count(tok)) { why = FinishReason::Stop; break; }
            if (!keep) { why = FinishReason::Cancelled; break; }
            if (st.generated_tokens >= req.max_new_tokens || cache_.length >= cache_.capacity) {
                why = FinishReason::Length;
                break;
            }
            if (greedy) {
                tok = model_->decode_argmax(cs_, cache_, tok, logits_, &st.decode_forward);
            } else {
                model_->decode(cs_, cache_, tok, logits_, &st.decode_forward);
                tok = pick();
            }
            st.decode_steps++;
        }
        st.decode_seconds = secs(t1, clk::now());
        if (stats_out) *stats_out = st;
        return why;
    }

private:
    gpu::Device& dev_;
    std::shared_ptr<Model> model_;
    std::shared_ptr<Tokenizer> tok_;
    gpu::CommandStream cs_;
    KVCache cache_;
    gpu::Buffer logits_;
};

} // namespace

std::unique_ptr<Engine> Engine::create(gpu::Device& dev, std::shared_ptr<Model> model,
                                       std::shared_ptr<Tokenizer> tokenizer) {
    return std::make_unique<SingleSequenceEngine>(dev, std::move(model), std::move(tokenizer));
}

} // namespace coral
