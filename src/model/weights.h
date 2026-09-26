// Internal: gpt-oss weights bound zero-copy to the GPU.
//
// Every tensor is a (shard buffer, byte offset) pair into the mmap'd
// safetensors shards wrapped with gpu::Device::wrap_no_copy, plus a CPU
// pointer to the same bytes. Nothing is copied or repacked here; MXFP4 expert
// weights keep the checkpoint layout (blocks [E][rows][K/32][16], scales
// [E][rows][K/32]). See the layout table in include/coral/model.h.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "coral/config.h"
#include "coral/gpu.h"
#include "coral/model.h"
#include "coral/safetensors.h"

namespace coral {

struct TensorRef {
    std::string          name;          // HF checkpoint name (diagnostics)
    gpu::Buffer          buf;           // whole-shard buffer (invalid if bind_gpu was not called)
    size_t               offset = 0;    // byte offset of the tensor within `buf`
    const uint8_t*       cpu = nullptr; // CPU pointer to the same bytes
    DType                dtype = DType::BF16;
    std::vector<int64_t> shape;
    size_t               nbytes = 0;

    bool valid() const { return cpu != nullptr; }
    int64_t dim(size_t i) const { return shape.at(i); }
    template <class T> const T* as() const { return reinterpret_cast<const T*>(cpu); }
};

struct LayerWeights {
    bool sliding = false;
    TensorRef attn_norm;                         // bf16 [H]
    TensorRef wq, bq;                            // bf16 [Q,H] / [Q]
    TensorRef wk, bk;                            // bf16 [KV,H] / [KV]
    TensorRef wv, bv;                            // bf16 [KV,H] / [KV]
    TensorRef wo, bo;                            // bf16 [H,Q] / [H]
    TensorRef sinks;                             // bf16 [heads]
    TensorRef mlp_norm;                          // bf16 [H]
    TensorRef router_w, router_b;                // bf16 [E,H] / [E]
    TensorRef gate_up_blocks, gate_up_scales;    // u8 [E,2I,H/32,16] / [E,2I,H/32]
    TensorRef gate_up_bias;                      // bf16 [E,2I]
    TensorRef down_blocks, down_scales;          // u8 [E,H,I/32,16] / [E,H,I/32]
    TensorRef down_bias;                         // bf16 [E,H]
};

struct Weights {
    TensorRef embed;        // bf16 [vocab,H]
    TensorRef lm_head;      // bf16 [vocab,H]
    TensorRef final_norm;   // bf16 [H]
    std::vector<LayerWeights> layers;
};

// Look up and validate (dtype + full shape) every tensor the forward pass
// needs. Throws std::runtime_error naming the tensor on any mismatch.
Weights bind_weights(const ModelConfig& cfg, const Safetensors& st);

// Accessors for the concrete gpt-oss model behind a Model (tests/tools).
// Throw std::invalid_argument if `m` is not a gpt-oss model.
const Weights&     model_weights(const Model& m);
const Safetensors& model_tensors(const Model& m);
// Unembedding precision of a loaded gpt-oss model (see ModelOptions::lm_head);
// switching to int8 quantizes on first use (~0.1 s, 580 MB).
void model_set_lm_head_int8(Model& m, bool on);
bool model_lm_head_int8(const Model& m);
// Pipelined greedy decode on/off (see ModelOptions::pipeline_greedy).
void model_set_pipeline(Model& m, bool on);
// Batched prefill (prefill_ops.h) on/off (default on; CORAL_PREFILL=token|batched
// sets the load-time default); off = token-by-token through the decode path.
void model_set_batched_prefill(Model& m, bool on);
bool model_batched_prefill(const Model& m);
// Maximum rows per batched-prefill chunk (default 1024; CORAL_PREFILL_CHUNK).
void model_set_prefill_chunk(Model& m, uint32_t rows);
// Shortest prompt that takes the batched path (default: see gptoss.cpp
// kMinBatched; 0 restores it; clamped to >= 2 — a single token always uses
// the decode path).
void model_set_prefill_min_tokens(Model& m, uint32_t n);

// Record an embedding gather: out[t] = embed[ids[t]] for t < n_tokens.
// `ids` holds int32 token ids; `out` receives bf16 [n_tokens][H].
void encode_embed_gather(gpu::Device& dev, gpu::CommandStream& cs, const TensorRef& table,
                         const gpu::Buffer& ids, size_t ids_offset,
                         const gpu::Buffer& out, size_t out_offset, uint32_t n_tokens);

} // namespace coral
