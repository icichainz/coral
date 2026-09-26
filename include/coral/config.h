// Model hyper-parameters, loaded from a Hugging Face config.json.
// Currently targets the gpt_oss architecture (gpt-oss-20b / gpt-oss-120b).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace coral {

struct ModelConfig {
    std::string model_type;               // "gpt_oss"
    std::string path;                     // model directory

    uint32_t vocab_size = 201088;
    uint32_t hidden_size = 2880;
    uint32_t intermediate_size = 2880;    // per-expert MLP width
    uint32_t num_layers = 24;
    uint32_t num_heads = 64;
    uint32_t num_kv_heads = 8;
    uint32_t head_dim = 64;
    uint32_t num_experts = 32;
    uint32_t experts_per_token = 4;
    uint32_t sliding_window = 128;
    std::vector<bool> layer_is_sliding;   // per layer: true = sliding window attention
    float rms_norm_eps = 1e-5f;
    float swiglu_limit = 7.0f;
    bool attention_bias = true;
    bool tie_word_embeddings = false;

    // RoPE (YaRN)
    float rope_theta = 150000.0f;
    float rope_factor = 32.0f;            // scaling factor
    uint32_t rope_original_max_pos = 4096;
    float rope_beta_fast = 32.0f;
    float rope_beta_slow = 1.0f;
    uint32_t max_position_embeddings = 131072;

    // Tokens
    int32_t eos_token_id = 200002;
    int32_t pad_token_id = 199999;

    // Quantization of expert weights ("mxfp4" or "" for dense bf16)
    std::string expert_quant = "mxfp4";

    uint32_t q_dim() const { return num_heads * head_dim; }
    uint32_t kv_dim() const { return num_kv_heads * head_dim; }

    static ModelConfig load(const std::string& model_dir);  // throws
    std::string summary() const;
};

} // namespace coral
