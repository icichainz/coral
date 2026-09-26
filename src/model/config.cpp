#include "coral/config.h"
#include "coral/json.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace coral {

ModelConfig ModelConfig::load(const std::string& model_dir) {
    std::filesystem::path p = std::filesystem::path(model_dir) / "config.json";
    std::ifstream in(p);
    if (!in) throw std::runtime_error("config: cannot open " + p.string());
    std::stringstream ss; ss << in.rdbuf();
    Json j = Json::parse(ss.str());

    ModelConfig c;
    c.path = model_dir;
    c.model_type = j.get_string("model_type", "");
    if (c.model_type != "gpt_oss")
        throw std::runtime_error("config: unsupported model_type '" + c.model_type + "' (only gpt_oss)");

    c.vocab_size        = uint32_t(j.get_int("vocab_size", c.vocab_size));
    c.hidden_size       = uint32_t(j.get_int("hidden_size", c.hidden_size));
    c.intermediate_size = uint32_t(j.get_int("intermediate_size", c.intermediate_size));
    c.num_layers        = uint32_t(j.get_int("num_hidden_layers", c.num_layers));
    c.num_heads         = uint32_t(j.get_int("num_attention_heads", c.num_heads));
    c.num_kv_heads      = uint32_t(j.get_int("num_key_value_heads", c.num_kv_heads));
    c.head_dim          = uint32_t(j.get_int("head_dim", c.head_dim));
    c.num_experts       = uint32_t(j.get_int("num_local_experts", c.num_experts));
    c.experts_per_token = uint32_t(j.get_int("num_experts_per_tok", j.get_int("experts_per_token", c.experts_per_token)));
    c.sliding_window    = uint32_t(j.get_int("sliding_window", c.sliding_window));
    c.rms_norm_eps      = float(j.get_double("rms_norm_eps", c.rms_norm_eps));
    c.swiglu_limit      = float(j.get_double("swiglu_limit", c.swiglu_limit));
    c.attention_bias    = j.get_bool("attention_bias", c.attention_bias);
    c.tie_word_embeddings = j.get_bool("tie_word_embeddings", c.tie_word_embeddings);
    c.rope_theta        = float(j.get_double("rope_theta", c.rope_theta));
    c.max_position_embeddings = uint32_t(j.get_int("max_position_embeddings", c.max_position_embeddings));
    c.eos_token_id      = int32_t(j.get_int("eos_token_id", c.eos_token_id));
    c.pad_token_id      = int32_t(j.get_int("pad_token_id", c.pad_token_id));

    const Json& rs = j["rope_scaling"];
    if (rs.is_object()) {
        c.rope_factor = float(rs.get_double("factor", c.rope_factor));
        c.rope_original_max_pos = uint32_t(rs.get_int("original_max_position_embeddings", c.rope_original_max_pos));
        c.rope_beta_fast = float(rs.get_double("beta_fast", c.rope_beta_fast));
        c.rope_beta_slow = float(rs.get_double("beta_slow", c.rope_beta_slow));
    }

    c.layer_is_sliding.assign(c.num_layers, false);
    const Json& lt = j["layer_types"];
    if (lt.is_array()) {
        for (size_t i = 0; i < lt.size() && i < c.num_layers; ++i)
            c.layer_is_sliding[i] = lt[i].as_string() == "sliding_attention";
    } else {
        for (size_t i = 0; i < c.num_layers; ++i) c.layer_is_sliding[i] = (i % 2 == 0);  // gpt-oss default
    }

    const Json& qc = j["quantization_config"];
    c.expert_quant = qc.is_object() ? qc.get_string("quant_method", "") : "";

    if (c.head_dim * c.num_heads == 0 || c.hidden_size == 0 || c.num_layers == 0)
        throw std::runtime_error("config: invalid dimensions");
    return c;
}

std::string ModelConfig::summary() const {
    std::ostringstream s;
    size_t sliding = 0; for (bool b : layer_is_sliding) sliding += b;
    s << model_type << ": " << num_layers << " layers (" << sliding << " sliding/" << (num_layers - sliding) << " full)"
      << ", hidden " << hidden_size << ", heads " << num_heads << "q/" << num_kv_heads << "kv x " << head_dim
      << ", experts " << num_experts << " (top-" << experts_per_token << ", " << intermediate_size << " wide, "
      << (expert_quant.empty() ? "bf16" : expert_quant) << ")"
      << ", vocab " << vocab_size << ", ctx " << max_position_embeddings
      << ", rope theta " << rope_theta << " yarn x" << rope_factor;
    return s.str();
}

} // namespace coral
