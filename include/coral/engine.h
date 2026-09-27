// Generation engine: sampling, stop handling, streaming callbacks, and the
// continuous-batching scheduler (BatchEngine) that interleaves many
// sequences on one GPU stream.
//
// Single-sequence generation (src/engine/engine.cpp): prefill the prompt,
// then one command buffer per generated token. Greedy decoding
// (temperature 0, no repetition penalty) uses the GPU argmax and reads back
// 4 bytes per token, pipelined by Model::decode_argmax (step t+1 is encoded
// and submitted while step t runs); otherwise the sampler reads the fp32
// logits in place (unified memory, no copy).
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "coral/gpu.h"
#include "coral/model.h"
#include "coral/tokenizer.h"

namespace coral {

struct SamplingParams {
    float temperature = 1.0f;     // 0 = greedy
    float top_p = 1.0f;
    int32_t top_k = 0;            // 0 = disabled
    float repetition_penalty = 1.0f;
    uint64_t seed = 0;            // 0 = random
};

struct GenerationRequest {
    std::vector<int32_t> prompt;
    uint32_t max_new_tokens = 1024;
    SamplingParams sampling;
    std::vector<int32_t> stop_tokens;     // e.g. <|return|>, <|call|>
    uint32_t kv_capacity = 0;             // KV cache positions; 0 = max(engine default 8192,
                                          // prompt + max_new_tokens), capped at max_position_embeddings
};

enum class FinishReason { Stop, Length, Cancelled, Error };

struct GenerationStats {
    uint32_t prompt_tokens = 0;
    uint32_t generated_tokens = 0;
    double prefill_seconds = 0;
    double decode_seconds = 0;
    uint32_t decode_steps = 0;            // forward passes after prefill (generated_tokens - 1 usually:
                                          // the first token comes from the prefill logits)
    ForwardStats prefill_forward;         // GPU/encode breakdown
    ForwardStats decode_forward;
    double tokens_per_second() const {
        const uint32_t n = decode_steps ? decode_steps : generated_tokens;
        return decode_seconds > 0 ? n / decode_seconds : 0;
    }
};

// Called for every generated token. Return false to cancel.
using TokenCallback = std::function<bool(int32_t token)>;
// Called once when a sequence ends (stop token, length, cancellation, error).
using FinishCallback = std::function<void(FinishReason, const GenerationStats&)>;

class BatchEngine;
struct BatchOptions {
    uint32_t max_batch = 8;     // sequences decoding together, 1..Model::kMaxDecodeBatch
};

class Engine {
public:
    static std::unique_ptr<Engine> create(gpu::Device& dev, std::shared_ptr<Model> model,
                                          std::shared_ptr<Tokenizer> tokenizer);
    virtual ~Engine() = default;

    virtual const Model& model() const = 0;
    virtual const Tokenizer& tokenizer() const = 0;

    // Synchronous single-sequence generation.
    virtual FinishReason generate(const GenerationRequest& req, const TokenCallback& on_token,
                                  GenerationStats* stats = nullptr) = 0;

    // Warm up: compile kernels (pipeline states), run one throwaway decode
    // step so every weight page and scratch buffer is touched.
    virtual void warmup() = 0;

    // A continuous-batching engine over the same device and model (and
    // tokenizer). Not for concurrent use with generate(): the model is not
    // thread-safe, so drive one engine at a time.
    virtual std::unique_ptr<BatchEngine> make_batch_engine(const BatchOptions& opt = {}) = 0;

    static constexpr uint32_t kDefaultKvCapacity = 8192;
};

// Continuous batching (src/engine/batch_engine.cpp). Requests are queued from
// any thread; one driver thread calls step() in a loop. A step admits queued
// requests while fewer than max_batch sequences are active — each is
// prefilled in full (its first token comes from the prefill logits) — then
// runs ONE decode step for all active sequences (Model::decode_batch: one
// command buffer, dense weights read once for all of them; a lone sequence
// takes the single-sequence path, pipelined when greedy), samples each row
// (GPU argmax when greedy, the CPU sampler on that row's logits otherwise),
// delivers the tokens and retires finished sequences, whose KV caches go
// back to a pool. Greedy results match Engine::generate token for token
// (tests/test_batch.cpp).
class BatchEngine {
public:
    static std::unique_ptr<BatchEngine> create(gpu::Device& dev, std::shared_ptr<Model> model,
                                               const BatchOptions& opt = {});
    virtual ~BatchEngine() = default;

    virtual const Model& model() const = 0;
    virtual uint32_t max_batch() const = 0;

    // Thread-safe. Queues `req`; returns its id. on_token runs on the step()
    // thread for every generated token (return false to stop); on_finish runs
    // exactly once on that thread, also for cancelled and failed requests.
    // `alive`, if given, is polled on the step() thread before admission and
    // at every step: false cancels the sequence without waiting for a token
    // (e.g. the HTTP client went away).
    virtual uint64_t submit(GenerationRequest req, TokenCallback on_token, FinishCallback on_finish,
                            std::function<bool()> alive = {}) = 0;
    // Thread-safe: the sequence finishes with FinishReason::Cancelled at the next step.
    virtual void cancel(uint64_t id) = 0;

    // One scheduling iteration (single driver thread). Returns true while
    // sequences are queued or active.
    virtual bool step() = 0;
    void run_until_idle() { while (step()) {} }
    // Blocks until work is queued, wake() is called, or the timeout passes;
    // returns true if there is work.
    virtual bool wait_for_work(double timeout_seconds) = 0;
    virtual void wake() = 0;                  // thread-safe

    virtual size_t active() const = 0;        // thread-safe: sequences decoding
    virtual size_t queued() const = 0;        // thread-safe: waiting for admission
    virtual void warmup() = 0;                // one throwaway batched step (driver thread)
};

// Sampler: fp32 logits -> token id. CPU implementation for the vocabulary
// tail; the GPU performs the argmax/top-k pre-pass in the roadmap.
int32_t sample(std::span<const float> logits, const SamplingParams& p, uint64_t& rng_state);

} // namespace coral
