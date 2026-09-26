// Generation engine: sampling, stop handling, streaming callbacks, and the
// scheduler that will later interleave multiple sequences (continuous
// batching) on one GPU stream.
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

    static constexpr uint32_t kDefaultKvCapacity = 8192;
};

// Sampler: fp32 logits -> token id. CPU implementation for the vocabulary
// tail; the GPU performs the argmax/top-k pre-pass in the roadmap.
int32_t sample(std::span<const float> logits, const SamplingParams& p, uint64_t& rng_state);

} // namespace coral
