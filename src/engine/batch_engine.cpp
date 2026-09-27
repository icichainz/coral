// Continuous batching: many sequences, one GPU stream (see engine.h).
//
// Scheduling policy (one step()):
//   1. drop queued requests that were cancelled or whose client is gone;
//   2. admit queued requests, oldest first, while fewer than max_batch are
//      active: prefill each in full on its own (pooled) KV cache and deliver
//      its first token (from the prefill logits);
//   3. retire active sequences that were cancelled / whose client is gone;
//   4. one decode step for every active sequence: Model::decode_batch for 2+
//      rows, the single-sequence path (pipelined greedy decode_argmax, or
//      decode + CPU sampler) for a lone sequence;
//   5. deliver each row's token, retire finished sequences (stop token,
//      length, callback false), return their caches to the pool.
// A prefill therefore delays the running sequences by its duration (~0.5 s
// for a 512-token prompt on M2 Max); prompts are not chunked across steps.
#include "coral/engine.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <random>
#include <stdexcept>
#include <unordered_set>

#include "coral/kernels.h"

namespace coral {

namespace {

using clk = std::chrono::steady_clock;
double secs(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }

struct Seq {
    uint64_t id = 0;
    GenerationRequest req;
    TokenCallback on_token;
    FinishCallback on_finish;
    std::function<bool()> alive;
    std::unordered_set<int32_t> stops;
    bool greedy = true;
    uint64_t rng = 0;
    std::vector<int32_t> seen;   // repetition penalty history
    KVCache cache;
    int32_t next = 0;            // token the next decode step consumes (the last one delivered)
    GenerationStats st;
    clk::time_point t_decode0;
    bool cancelled = false;
};

class BatchEngineImpl final : public BatchEngine {
public:
    BatchEngineImpl(gpu::Device& dev, std::shared_ptr<Model> model, const BatchOptions& opt)
        : dev_(dev), model_(std::move(model)), cs_(dev_.stream()) {
        if (!model_) throw std::invalid_argument("BatchEngine::create: null model");
        if (opt.max_batch < 1 || opt.max_batch > Model::kMaxDecodeBatch)
            throw std::invalid_argument("BatchEngine: max_batch must be 1.." + std::to_string(Model::kMaxDecodeBatch));
        max_batch_ = opt.max_batch;
        try { (void)dev_.kernel("embed_gather_f32"); } catch (const std::exception&) { load_kernels(dev_); }
    }

    ~BatchEngineImpl() override {
        // Sequences still queued or active are dropped without callbacks
        // (their owners' state is going away with the engine).
    }

    const Model& model() const override { return *model_; }
    uint32_t max_batch() const override { return max_batch_; }

    uint64_t submit(GenerationRequest req, TokenCallback on_token, FinishCallback on_finish,
                    std::function<bool()> alive) override {
        if (req.prompt.empty()) throw std::invalid_argument("submit: empty prompt");
        auto s = std::make_unique<Seq>();
        s->req = std::move(req);
        s->on_token = std::move(on_token);
        s->on_finish = std::move(on_finish);
        s->alive = std::move(alive);
        uint64_t id;
        {
            std::lock_guard<std::mutex> lk(mu_);
            id = s->id = ++next_id_;
            queue_.push_back(std::move(s));
        }
        cv_.notify_all();
        return id;
    }

    void cancel(uint64_t id) override {
        std::lock_guard<std::mutex> lk(mu_);
        cancel_.insert(id);
    }

    bool wait_for_work(double timeout_seconds) override {
        std::unique_lock<std::mutex> lk(mu_);
        if (!queue_.empty() || n_active_ > 0) return true;
        cv_.wait_for(lk, std::chrono::duration<double>(timeout_seconds),
                     [&] { return !queue_.empty() || woken_; });
        woken_ = false;
        return !queue_.empty() || n_active_ > 0;
    }
    void wake() override {
        {
            std::lock_guard<std::mutex> lk(mu_);
            woken_ = true;
        }
        cv_.notify_all();
    }

    size_t active() const override { std::lock_guard<std::mutex> lk(mu_); return n_active_; }
    size_t queued() const override { std::lock_guard<std::mutex> lk(mu_); return queue_.size(); }

    void warmup() override {
        // Builds the batched pipelines and scratch with a throwaway 2-row step.
        if (max_batch_ < 2) return;
        KVCache a = model_->new_cache(8), b = model_->new_cache(8);
        KVCache* caches[2] = {&a, &b};
        const int32_t toks[2] = {0, 1};
        model_->decode_batch(cs_, caches, toks, &blogits_, &bargmax_);
    }

    bool step() override {
        // 1. Admission (and queued cancellations).
        std::vector<std::unique_ptr<Seq>> admit, dropped;
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (auto it = queue_.begin(); it != queue_.end();) {
                if (cancel_.erase((*it)->id)) { (*it)->cancelled = true; dropped.push_back(std::move(*it)); it = queue_.erase(it); }
                else ++it;
            }
            while (!queue_.empty() && active_.size() + admit.size() < max_batch_) {
                admit.push_back(std::move(queue_.front()));
                queue_.pop_front();
            }
            for (auto& s : active_) if (cancel_.erase(s->id)) s->cancelled = true;
        }
        for (auto& s : dropped) finish(*s, FinishReason::Cancelled, false);
        for (auto& s : admit) {
            if (s->alive && !s->alive()) { finish(*s, FinishReason::Cancelled, false); continue; }
            if (start(*s)) active_.push_back(std::move(s));
        }
        set_active_count();

        // 2. Retire cancelled / abandoned sequences.
        for (auto& s : active_)
            if (s->cancelled || (s->alive && !s->alive())) { finish(*s, FinishReason::Cancelled, true); s.reset(); }
        compact();

        // 3. One decode step for everyone.
        if (!active_.empty()) decode_step();
        set_active_count();
        std::lock_guard<std::mutex> lk(mu_);
        return !queue_.empty() || !active_.empty();
    }

private:
    uint32_t capacity_for(const GenerationRequest& req) const {
        const ModelConfig& cfg = model_->config();
        uint32_t cap = req.kv_capacity;
        if (cap == 0) {
            const uint64_t need = uint64_t(req.prompt.size()) + req.max_new_tokens;
            cap = uint32_t(std::min<uint64_t>(std::max<uint64_t>(Engine::kDefaultKvCapacity, need),
                                              cfg.max_position_embeddings));
        }
        return cap;
    }

    // KV caches are pooled by capacity: a request takes the smallest pooled
    // cache that is large enough (stale rows beyond `length` are never read).
    KVCache acquire(uint32_t cap) {
        size_t best = pool_.size();
        for (size_t i = 0; i < pool_.size(); ++i)
            if (pool_[i].capacity >= cap && (best == pool_.size() || pool_[i].capacity < pool_[best].capacity)) best = i;
        KVCache c;
        if (best < pool_.size()) {
            c = pool_[best];
            pool_.erase(pool_.begin() + ptrdiff_t(best));
        } else {
            c = model_->new_cache(cap);
        }
        c.length = 0;
        return c;
    }
    void release(KVCache& c) {
        if (!c.k_full.valid()) return;
        pool_.push_back(c);
        c = KVCache{};
        // Keep at most max_batch idle caches (the largest ones win: they serve any request).
        if (pool_.size() > max_batch_) {
            auto smallest = std::min_element(pool_.begin(), pool_.end(),
                                             [](const KVCache& a, const KVCache& b) { return a.capacity < b.capacity; });
            pool_.erase(smallest);
        }
    }

    int32_t sample_row(Seq& s, float* l) {
        const size_t V = model_->config().vocab_size;
        const SamplingParams& sp = s.req.sampling;
        if (sp.repetition_penalty != 1.0f) {
            std::vector<int32_t> u(s.seen);
            std::sort(u.begin(), u.end());
            u.erase(std::unique(u.begin(), u.end()), u.end());
            for (int32_t t : u)
                if (t >= 0 && size_t(t) < V) l[t] = l[t] > 0 ? l[t] / sp.repetition_penalty : l[t] * sp.repetition_penalty;
        }
        return sample(std::span<const float>(l, V), sp, s.rng);
    }

    // Prefill + first token. Returns true if the sequence continues decoding.
    bool start(Seq& s) {
        const GenerationRequest& req = s.req;
        s.st.prompt_tokens = uint32_t(req.prompt.size());
        if (req.max_new_tokens == 0) { finish(s, FinishReason::Length, false); return false; }
        const SamplingParams& sp = req.sampling;
        s.greedy = sp.temperature <= 0.0f && sp.repetition_penalty == 1.0f;
        s.rng = sp.seed ? sp.seed : (uint64_t(std::random_device{}()) << 32) ^ std::random_device{}();
        s.stops.insert(req.stop_tokens.begin(), req.stop_tokens.end());
        if (sp.repetition_penalty != 1.0f) s.seen.assign(req.prompt.begin(), req.prompt.end());
        int32_t tok = 0;
        try {
            const uint32_t cap = capacity_for(req);
            if (req.prompt.size() >= cap)
                throw std::out_of_range("prompt (" + std::to_string(req.prompt.size()) +
                                        " tokens) does not fit the KV cache (" + std::to_string(cap) + ")");
            s.cache = acquire(cap);
            const auto t0 = clk::now();
            if (s.greedy) tok = model_->prefill_argmax(cs_, s.cache, req.prompt, plogits_, &s.st.prefill_forward);
            else {
                model_->prefill(cs_, s.cache, req.prompt, plogits_, &s.st.prefill_forward);
                tok = sample_row(s, plogits_.as<float>());
            }
            s.t_decode0 = clk::now();
            s.st.prefill_seconds = secs(t0, s.t_decode0);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[coral] batch: prefill failed: %s\n", e.what());
            finish(s, FinishReason::Error, false);
            return false;
        }
        return !deliver(s, tok);
    }

    // Hands `tok` to the sequence; returns true (and finishes it) when it ends.
    bool deliver(Seq& s, int32_t tok) {
        s.st.generated_tokens++;
        if (s.req.sampling.repetition_penalty != 1.0f) s.seen.push_back(tok);
        bool keep = true;
        try { keep = s.on_token ? s.on_token(tok) : true; }
        catch (const std::exception& e) {
            std::fprintf(stderr, "[coral] batch: token callback threw: %s\n", e.what());
            finish(s, FinishReason::Error, true);
            return true;
        }
        FinishReason why;
        if (s.stops.count(tok)) why = FinishReason::Stop;
        else if (!keep) why = FinishReason::Cancelled;
        else if (s.st.generated_tokens >= s.req.max_new_tokens || s.cache.length >= s.cache.capacity) why = FinishReason::Length;
        else { s.next = tok; return false; }
        finish(s, why, true);
        return true;
    }

    void finish(Seq& s, FinishReason why, bool decoded) {
        if (decoded) s.st.decode_seconds = secs(s.t_decode0, clk::now());
        release(s.cache);
        if (s.on_finish) {
            try { s.on_finish(why, s.st); }
            catch (const std::exception& e) { std::fprintf(stderr, "[coral] batch: finish callback threw: %s\n", e.what()); }
        }
    }

    void decode_step() {
        const size_t B = active_.size();
        ForwardStats fs;
        std::vector<int32_t> toks(B);
        try {
            if (B == 1) {
                Seq& s = *active_[0];
                if (s.greedy) toks[0] = model_->decode_argmax(cs_, s.cache, s.next, logits1_, &fs);
                else {
                    model_->decode(cs_, s.cache, s.next, logits1_, &fs);
                    toks[0] = sample_row(s, logits1_.as<float>());
                }
            } else {
                std::vector<KVCache*> caches(B);
                std::vector<int32_t> in(B);
                for (size_t i = 0; i < B; ++i) { caches[i] = &active_[i]->cache; in[i] = active_[i]->next; }
                model_->decode_batch(cs_, caches, in, &blogits_, &bargmax_, &fs);
                const size_t V = model_->config().vocab_size;
                for (size_t i = 0; i < B; ++i)
                    toks[i] = active_[i]->greedy ? bargmax_.as<int32_t>()[i]
                                                 : sample_row(*active_[i], blogits_.as<float>() + i * V);
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[coral] batch: decode step failed: %s\n", e.what());
            for (auto& s : active_) { finish(*s, FinishReason::Error, true); s.reset(); }
            compact();
            return;
        }
        for (size_t i = 0; i < B; ++i) {
            Seq& s = *active_[i];
            s.st.decode_steps++;
            s.st.decode_forward.gpu_seconds += fs.gpu_seconds;
            s.st.decode_forward.encode_seconds += fs.encode_seconds;
            s.st.decode_forward.dispatches += fs.dispatches;
            s.st.decode_forward.submits += fs.submits;
            s.st.decode_forward.tokens += 1;
            if (deliver(s, toks[i])) active_[i].reset();
        }
        compact();
        ++steps_;
    }

    void compact() {
        active_.erase(std::remove(active_.begin(), active_.end(), nullptr), active_.end());
    }
    void set_active_count() {
        std::lock_guard<std::mutex> lk(mu_);
        n_active_ = active_.size();
    }

    gpu::Device& dev_;
    std::shared_ptr<Model> model_;
    gpu::CommandStream cs_;
    uint32_t max_batch_ = 8;

    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::unique_ptr<Seq>> queue_;     // guarded by mu_
    std::unordered_set<uint64_t> cancel_;        // guarded by mu_
    uint64_t next_id_ = 0;                       // guarded by mu_
    size_t n_active_ = 0;                        // guarded by mu_ (mirror of active_.size())
    bool woken_ = false;                         // guarded by mu_

    // Driver-thread state.
    std::vector<std::unique_ptr<Seq>> active_;
    std::vector<KVCache> pool_;
    gpu::Buffer plogits_, logits1_, blogits_, bargmax_;
    uint64_t steps_ = 0;
};

} // namespace

std::unique_ptr<BatchEngine> BatchEngine::create(gpu::Device& dev, std::shared_ptr<Model> model,
                                                 const BatchOptions& opt) {
    return std::make_unique<BatchEngineImpl>(dev, std::move(model), opt);
}

} // namespace coral
