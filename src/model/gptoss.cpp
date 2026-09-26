// gpt-oss model: weight binding, KV cache allocation, the decode forward
// pass, token-by-token prefill, and bandwidth accounting.
//
// One decode step (encode_decode) records, into a single command buffer:
//   embed_gather_f32                         1  token row -> fp32 residual
//   24 x attention (attention_ops.h)         3
//   24 x MoE       (moe_ops.h)               3
//   gemv_i8_norm   (final RMSNorm + lm_head)  1  fused, fp32 logits [vocab]
//                  (gemv_bf16_norm with ModelOptions::LmHead::Bf16)
//   argmax_f32_partial/final (optional)      2
// = 146 dispatches (148 with the GPU argmax). Consecutive dispatches are
// barrier-ordered by the CommandStream, so the scratch buffers below are
// reused by every layer and by every token of a multi-token submit.
#include "coral/model.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "attention_ops.h"
#include "gemv_i8.h"
#include "moe_ops.h"
#include "weights.h"

namespace coral {

namespace {

class GptOssModel final : public Model {
public:
    GptOssModel(gpu::Device& dev, const std::string& dir, const ModelOptions& opt) : dev_(dev) {
        cfg_ = ModelConfig::load(dir);
        st_ = Safetensors::open(dir);
        st_->bind_gpu(dev_);
        w_ = bind_weights(cfg_, *st_);
        for (bool s : cfg_.layer_is_sliding) (s ? layers_slide_ : layers_full_)++;
        // Page in and wire the shard buffers now (one blocking submit), so the
        // first decode step does not pay for it. No madvise prefetch: the
        // residency wiring reads every page anyway.
        dev_.make_resident();
        bool int8 = opt.lm_head == ModelOptions::LmHead::Int8;
        if (const char* e = std::getenv("CORAL_LM_HEAD")) {
            if (!std::strcmp(e, "bf16")) int8 = false;
            else if (!std::strcmp(e, "int8")) int8 = true;
            else throw std::invalid_argument(std::string("CORAL_LM_HEAD must be bf16 or int8, got '") + e + "'");
        }
        set_lm_head_int8(int8);
        pipeline_ = opt.pipeline_greedy;
        if (const char* e = std::getenv("CORAL_PIPELINE")) pipeline_ = std::strcmp(e, "0") != 0;
    }

    // Switch the unembedding precision (quantizes on first use of int8).
    void set_lm_head_int8(bool on) {
        if (on && !lm_i8_.valid()) {
            lm_i8_ = quantize_rows_i8(dev_, w_.lm_head);
            dev_.make_resident();
        }
        lm_int8_ = on;
    }
    bool lm_head_int8() const { return lm_int8_; }
    void set_pipeline(bool on) { retire_spec(); pipeline_ = on; }

    const ModelConfig& config() const override { return cfg_; }
    const Weights& weights() const { return w_; }
    const Safetensors& tensors() const { return *st_; }

    // Layout (model.h): bf16 [layers_full][capacity][kv_heads][head_dim] and
    // [layers_slide][window][kv_heads][head_dim], zero-initialized.
    KVCache new_cache(uint32_t capacity) const override {
        const size_t row = size_t(cfg_.kv_dim()) * 2;  // bytes per position per layer
        KVCache c;
        c.capacity = capacity;
        c.length = 0;
        const size_t full = size_t(layers_full_) * capacity * row;
        const size_t slide = size_t(layers_slide_) * cfg_.sliding_window * row;
        c.k_full = dev_.alloc(full, true);
        c.v_full = dev_.alloc(full, true);
        c.k_slide = dev_.alloc(slide, true);
        c.v_slide = dev_.alloc(slide, true);
        return c;
    }

    // ---- forward pass ---------------------------------------------------------
    void encode_decode(gpu::CommandStream& cs, KVCache& cache, int32_t token,
                       gpu::Buffer* logits_out, gpu::Buffer* argmax_out) override {
        retire_spec();
        if (token < 0 || uint32_t(token) >= uint32_t(w_.embed.dim(0)))
            throw std::out_of_range("decode: token id " + std::to_string(token) + " out of range");
        encode_step(cs, cache, token, nullptr, logits_out, argmax_out);
    }

    // One decode step. The token comes by value, or (token_src != nullptr)
    // from int32 token_src[0] on the GPU (a previous step's argmax).
    void encode_step(gpu::CommandStream& cs, KVCache& cache, int32_t token, const gpu::Buffer* token_src,
                     gpu::Buffer* logits_out, gpu::Buffer* argmax_out) {
        if (cache.length >= cache.capacity)
            throw std::out_of_range("decode: KV cache full (capacity " + std::to_string(cache.capacity) + ")");
        ensure_scratch(cache.capacity);
        const uint32_t pos = cache.length;
        const uint32_t H = cfg_.hidden_size;

        // 1. embedding row -> fp32 residual.
        struct { int32_t token; uint32_t dim, from_buffer, ids_index; } ep{token, H, token_src ? 1u : 0u, 0};
        cs.dispatch(dev_.kernel("embed_gather_f32"),
                    gpu::Args().buffer(0, w_.embed.buf, w_.embed.offset).buffer(1, token_src ? *token_src : argmax_buf_)
                               .buffer(2, residual_).value(3, ep),
                    {(H / 4 + 255) / 256}, {256});

        // 2. decoder layers.
        for (uint32_t l = 0; l < cfg_.num_layers; ++l) {
            const LayerWeights& L = w_.layers[l];
            encode_attention_decode(dev_, cs, cfg_, L, l, L.sliding, kv_slot_[l], cache, pos, residual_, 0, attn_);
            encode_moe_decode(dev_, cs, cfg_, L, residual_, moe_);
        }

        // 3. final RMSNorm folded into the lm_head GEMV, 4. optional argmax.
        if (logits_out || argmax_out) {
            gpu::Buffer& logits = logits_out ? *logits_out : logits_scratch_;
            const size_t need = size_t(cfg_.vocab_size) * 4;
            if (!logits.valid()) logits = dev_.alloc(need, true);
            if (logits.size() < need) throw std::invalid_argument("decode: logits buffer smaller than vocab*4");
            if (lm_int8_) {
                encode_gemv_i8(dev_, cs, lm_i8_, residual_, 0, logits, 0, &w_.final_norm, cfg_.rms_norm_eps);
            } else {
                GemvOptions o;
                o.norm = &w_.final_norm;
                o.eps = cfg_.rms_norm_eps;
                encode_gemv_bf16(dev_, cs, w_.lm_head, residual_, 0, logits, 0, o);
            }
            if (argmax_out) {
                if (!argmax_out->valid()) *argmax_out = dev_.alloc(16, true);
                encode_argmax_f32(dev_, cs, logits, 0, cfg_.vocab_size, *argmax_out, 0, argmax_scratch_);
            }
        }
        cache.length = pos + 1;
    }

    void prefill(gpu::CommandStream& cs, KVCache& cache, std::span<const int32_t> tokens,
                 gpu::Buffer& logits_out, ForwardStats* stats) override {
        retire_spec();
        run_tokens(cs, cache, tokens, logits_out, nullptr, stats);
    }
    int32_t prefill_argmax(gpu::CommandStream& cs, KVCache& cache, std::span<const int32_t> tokens,
                           gpu::Buffer& logits_out, ForwardStats* stats) override {
        retire_spec();
        run_tokens(cs, cache, tokens, logits_out, &argmax_buf_, stats);
        return argmax_buf_.as<int32_t>()[0];
    }
    void decode(gpu::CommandStream& cs, KVCache& cache, int32_t token,
                gpu::Buffer& logits_out, ForwardStats* stats) override {
        retire_spec();
        run_tokens(cs, cache, std::span<const int32_t>(&token, 1), logits_out, nullptr, stats);
    }
    int32_t decode_argmax(gpu::CommandStream& cs, KVCache& cache, int32_t token,
                          gpu::Buffer& logits_out, ForwardStats* stats) override {
        if (pipeline_) return decode_argmax_pipelined(cache, token, logits_out, stats);
        retire_spec();
        run_tokens(cs, cache, std::span<const int32_t>(&token, 1), logits_out, &argmax_buf_, stats);
        return argmax_buf_.as<int32_t>()[0];
    }

    // ---- pipelined greedy decode -------------------------------------------
    // decode_argmax(token t at position p) submits step p, then — before
    // waiting for it — encodes and submits step p+1 with its token read on
    // the GPU from step p's argmax (a queue barrier orders the two command
    // buffers). The next call, if it continues the same sequence with that
    // argmax (same cache, length p+1), finds its step already running and
    // only encodes p+2: host encode and submit latency are hidden behind GPU
    // work. Any other call retires (waits for) the speculative step first;
    // its KV row p+1 lies beyond cache.length and is simply rewritten later.
    // Two internal streams, argmax slots and logits buffers alternate; the
    // caller's `logits_out` handle is pointed at the buffer holding this
    // step's logits. The caller's CommandStream is not used on this path.
    struct Spec {
        bool active = false;
        const KVCache* owner = nullptr;
        KVCache cache;              // holds the cache buffers alive while in flight
        uint32_t pos = 0;           // position of the in-flight step
        int32_t token = 0;          // the token it consumes (argmax of the step before)
        int slot = 0;
        size_t dispatches = 0;
        double encode = 0;
        std::chrono::steady_clock::time_point submitted;
    };

    void retire_spec() {
        if (!spec_.active) return;
        pstream_[spec_.slot]->wait();
        spec_ = Spec{};
    }

    // Encode + submit one step on pipeline slot `slot` for position cache.length.
    Spec launch(KVCache& cache, int32_t token, const gpu::Buffer* token_src, int slot) {
        using clk = std::chrono::steady_clock;
        if (!pstream_[slot]) pstream_[slot] = std::make_unique<gpu::CommandStream>(dev_.stream());
        const size_t need = size_t(cfg_.vocab_size) * 4;
        if (!plogits_[slot].valid()) plogits_[slot] = dev_.alloc(need, true);
        if (!pargmax_[slot].valid()) pargmax_[slot] = dev_.alloc(16, true);
        gpu::CommandStream& cs = *pstream_[slot];
        Spec s;
        s.owner = &cache;
        s.cache = cache;
        s.pos = cache.length;
        s.token = token;
        s.slot = slot;
        const auto t0 = clk::now();
        cs.begin();
        try {
            encode_step(cs, cache, token, token_src, &plogits_[slot], &pargmax_[slot]);
        } catch (...) {
            cs.submit();
            cs.wait();
            throw;
        }
        s.dispatches = cs.dispatch_count();
        cs.submit();
        s.submitted = clk::now();
        s.encode = std::chrono::duration<double>(s.submitted - t0).count();
        s.active = true;
        return s;
    }

    int32_t decode_argmax_pipelined(KVCache& cache, int32_t token, gpu::Buffer& logits_out, ForwardStats* stats) {
        using clk = std::chrono::steady_clock;
        const bool hit = spec_.active && spec_.owner == &cache && spec_.pos == cache.length &&
                         spec_.token == token && spec_.cache.k_full.gpu_address() == cache.k_full.gpu_address() &&
                         cache.capacity == spec_.cache.capacity;
        Spec cur;
        if (hit) {
            cur = spec_;
            spec_ = Spec{};
            cache.length = cur.pos + 1;
        } else {
            retire_spec();
            if (token < 0 || uint32_t(token) >= uint32_t(w_.embed.dim(0)))
                throw std::out_of_range("decode: token id " + std::to_string(token) + " out of range");
            cur = launch(cache, token, nullptr, next_slot_);
            next_slot_ ^= 1;
        }
        // Speculate the next step (its token = this step's argmax, on the GPU).
        if (cache.length < cache.capacity) {
            const uint32_t len = cache.length;
            spec_ = launch(cache, -1, &pargmax_[cur.slot], cur.slot ^ 1);
            cache.length = len;   // not committed until the caller asks for it
        }
        pstream_[cur.slot]->wait();
        const auto done = clk::now();
        const int32_t out = pargmax_[cur.slot].as<int32_t>()[0];
        if (spec_.active) spec_.token = out;
        logits_out = plogits_[cur.slot];
        if (stats) {
            // GPU time of this step: from when it could start (its submit, or
            // the previous step's completion) to its completion.
            const auto start = std::max(cur.submitted, last_done_);
            stats->gpu_seconds += std::chrono::duration<double>(done - start).count();
            // Encode time and dispatches of the step being returned (a
            // speculative step is counted when it is committed).
            stats->encode_seconds += cur.encode;
            stats->dispatches += cur.dispatches;
            stats->submits += 1;
            stats->tokens += 1;
        }
        last_done_ = done;
        next_slot_ = cur.slot ^ 1;
        return out;
    }

    // Bytes read from memory for one decode step at context length `ctx`
    // (the new token attends to `ctx` positions, itself included):
    //
    //   embed row                       H*2
    // + per layer:
    //     attn_norm + mlp_norm          2*H*2
    //     wq,wk,wv,wo (bf16)            (Q*H + 2*KV*H + H*Q)*2
    //     bq,bk,bv,bo                   (Q + 2*KV + H)*2
    //     sinks                         heads*2
    //     router w + b                  (E*H + E)*2
    //     top-k experts (MXFP4)         k * [2I*H/32*(16+1) + 2I*2      gate_up blocks+scales+bias
    //                                        + H*I/32*(16+1) + H*2]     down    blocks+scales+bias
    //     KV cache read (K and V)       2 * n_pos * KV*2,  n_pos = ctx (full) or min(ctx, window) (sliding)
    // + final_norm + lm_head            H*2 + V*H*2   (bf16)  or  H*2 + V*H + V*4  (int8 + row scales)
    //
    // Activations and the KV write of the new token are ignored (a few KB).
    uint64_t decode_bytes_per_token(uint32_t ctx) const override {
        const uint64_t H = cfg_.hidden_size, I = cfg_.intermediate_size, V = cfg_.vocab_size;
        const uint64_t Q = cfg_.q_dim(), KV = cfg_.kv_dim(), E = cfg_.num_experts, K = cfg_.experts_per_token;
        const uint64_t attn = (Q * H + 2 * KV * H + H * Q) * 2 + (Q + 2 * KV + H) * 2 + uint64_t(cfg_.num_heads) * 2;
        const uint64_t norms = 2 * H * 2;
        const uint64_t router = (E * H + E) * 2;
        const uint64_t expert = 2 * I * (H / 32) * 17 + 2 * I * 2 + H * (I / 32) * 17 + H * 2;
        const uint64_t per_layer = attn + norms + router + K * expert;
        const uint64_t kv_row = 2 * KV * 2;  // K + V, bf16, one position one layer
        const uint64_t kv = kv_row * (uint64_t(layers_full_) * ctx +
                                      uint64_t(layers_slide_) * std::min<uint64_t>(ctx, cfg_.sliding_window));
        const uint64_t lm = lm_int8_ ? V * H + V * 4 : V * H * 2;
        return H * 2 + uint64_t(cfg_.num_layers) * per_layer + kv + H * 2 + lm;
    }

private:
    // Tokens per command buffer in prefill: 16 x 146 dispatches stays well
    // inside the backend's 4096-dispatch argument-table ring.
    static constexpr uint32_t kPrefillTokensPerSubmit = 16;

    // Run `tokens` through the decode path, kPrefillTokensPerSubmit per
    // submit; only the last one gets the unembedding (+ argmax).
    void run_tokens(gpu::CommandStream& cs, KVCache& cache, std::span<const int32_t> tokens,
                    gpu::Buffer& logits_out, gpu::Buffer* argmax_out, ForwardStats* stats) {
        if (tokens.empty()) throw std::invalid_argument("prefill: no tokens");
        if (size_t(cache.length) + tokens.size() > cache.capacity)
            throw std::out_of_range("prefill: " + std::to_string(tokens.size()) + " tokens at position " +
                                    std::to_string(cache.length) + " exceed KV cache capacity " +
                                    std::to_string(cache.capacity));
        using clk = std::chrono::steady_clock;
        size_t i = 0;
        while (i < tokens.size()) {
            const size_t end = std::min(tokens.size(), i + kPrefillTokensPerSubmit);
            const auto t0 = clk::now();
            cs.begin();
            try {
                for (; i < end; ++i) {
                    const bool last = i + 1 == tokens.size();
                    encode_decode(cs, cache, tokens[i], last ? &logits_out : nullptr, last ? argmax_out : nullptr);
                }
            } catch (...) {
                cs.submit();   // keep the stream balanced; the partial work is harmless
                cs.wait();
                throw;
            }
            const size_t n_dispatch = cs.dispatch_count();
            cs.submit();
            const double enc = std::chrono::duration<double>(clk::now() - t0).count();
            const double gpu = cs.wait();
            if (stats) {
                stats->encode_seconds += enc;
                stats->gpu_seconds += gpu;
                stats->dispatches += n_dispatch;
                stats->submits += 1;
            }
        }
        if (stats) stats->tokens += uint32_t(tokens.size());
    }

    // Scratch shared by all layers/tokens; the rope table covers
    // [0, max_positions) and is rebuilt when a larger cache shows up.
    void ensure_scratch(uint32_t max_positions) {
        if (!residual_.valid()) {
            residual_ = dev_.alloc(size_t(cfg_.hidden_size) * 4, true);
            moe_ = make_moe_scratch(dev_, cfg_);
            argmax_scratch_ = dev_.alloc(argmax_scratch_bytes(), true);
            argmax_buf_ = dev_.alloc(16, true);
            kv_slot_.resize(cfg_.num_layers);
            for (uint32_t l = 0; l < cfg_.num_layers; ++l) kv_slot_[l] = kv_slot_for_layer(cfg_, l);
        }
        if (attn_.max_positions < max_positions) attn_ = make_attn_scratch(dev_, cfg_, max_positions);
    }

    gpu::Device& dev_;
    ModelConfig cfg_;
    std::unique_ptr<Safetensors> st_;
    Weights w_;
    QuantI8 lm_i8_;               // int8 lm_head (ModelOptions::LmHead::Int8)
    bool lm_int8_ = false;
    uint32_t layers_full_ = 0, layers_slide_ = 0;

    // Forward-pass state (lazily created by ensure_scratch).
    AttnScratch attn_;
    MoeScratch moe_;
    gpu::Buffer residual_;        // fp32 [H]
    gpu::Buffer logits_scratch_;  // fp32 [vocab] when the caller wants only the argmax
    gpu::Buffer argmax_scratch_;
    gpu::Buffer argmax_buf_;      // int32 greedy id of the last step
    // Pipelined greedy decode (decode_argmax_pipelined).
    bool pipeline_ = true;
    Spec spec_;
    std::unique_ptr<gpu::CommandStream> pstream_[2];
    gpu::Buffer plogits_[2], pargmax_[2];
    int next_slot_ = 0;
    std::chrono::steady_clock::time_point last_done_{};
    std::vector<uint32_t> kv_slot_;
};

const GptOssModel& as_gptoss(const Model& m) {
    auto* g = dynamic_cast<const GptOssModel*>(&m);
    if (!g) throw std::invalid_argument("not a gpt-oss model");
    return *g;
}

} // namespace

std::unique_ptr<Model> Model::load(gpu::Device& dev, const std::string& model_dir) {
    return load(dev, model_dir, ModelOptions{});
}
std::unique_ptr<Model> Model::load(gpu::Device& dev, const std::string& model_dir, const ModelOptions& opt) {
    return std::make_unique<GptOssModel>(dev, model_dir, opt);
}

void model_set_lm_head_int8(Model& m, bool on) {
    auto* g = dynamic_cast<GptOssModel*>(&m);
    if (!g) throw std::invalid_argument("not a gpt-oss model");
    g->set_lm_head_int8(on);
}
bool model_lm_head_int8(const Model& m) { return as_gptoss(m).lm_head_int8(); }
void model_set_pipeline(Model& m, bool on) {
    auto* g = dynamic_cast<GptOssModel*>(&m);
    if (!g) throw std::invalid_argument("not a gpt-oss model");
    g->set_pipeline(on);
}

const Weights& model_weights(const Model& m) { return as_gptoss(m).weights(); }
const Safetensors& model_tensors(const Model& m) { return as_gptoss(m).tensors(); }

} // namespace coral
