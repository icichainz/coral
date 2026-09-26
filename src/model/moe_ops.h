// MoE half of a gpt-oss decoder layer, decode path (one token, M = 1).
//
// Everything here only ENCODES dispatches onto a CommandStream; the caller
// submits. Expert selection stays on the GPU: the router writes logits and its
// last threadgroup selects the top-k (ids/probs) for the gate_up and down
// kernels (no CPU round trip).
//
// Per layer, encode_moe_decode issues 3 dispatches (kernels in src/kernels/moe.metal):
//   1. moe_router              normed = rmsnorm(residual) * mlp_norm; logits = W_r·normed + b_r;
//                              top-k + softmax -> ids, probs
//   2. mx_gemv_gate_up_swiglu  h[k] = swiglu(W_gu[ids[k]]·normed + b_gu)
//   3. mx_gemv_down            residual += sum_k probs[k] * (W_d[ids[k]]·h[k] + b_d)
//
// The input is the raw fp32 residual stream (the post-attention RMSNorm is
// fused into dispatch 1); the residual is updated in place.
#pragma once

#include <cstddef>
#include <cstdint>

#include "coral/config.h"
#include "coral/gpu.h"
#include "weights.h"

namespace coral {

struct MoeScratch {
    gpu::Buffer normed;       // fp32 [H]      rmsnorm(residual) * mlp_norm
    gpu::Buffer logits;       // fp32 [E]      router logits
    gpu::Buffer expert_ids;   // int32 [K]     selected experts, by descending logit (ties: lowest index)
    gpu::Buffer probs;        // fp32 [K]      softmax over the K selected logits
    gpu::Buffer h;            // fp32 [K][I]   SwiGLU activations per selected expert
    gpu::Buffer counter;      // uint          router threadgroup election (zero between dispatches)
};

MoeScratch make_moe_scratch(gpu::Device& dev, const ModelConfig& cfg);

// Full MoE block: residual += moe(rmsnorm(residual)). 3 dispatches.
void encode_moe_decode(gpu::Device& dev, gpu::CommandStream& cs, const ModelConfig& cfg,
                       const LayerWeights& L, const gpu::Buffer& residual, MoeScratch& s);

// ---- individual stages (tests / tooling) ------------------------------------

// Dispatch 1: s.normed, s.logits, s.expert_ids, s.probs. If `apply_norm` is
// false, s.normed receives a copy of `x` and the router runs on it
// unnormalized (norm_w is unused).
void encode_moe_router(gpu::Device& dev, gpu::CommandStream& cs, const ModelConfig& cfg,
                       const TensorRef& norm_w, const TensorRef& router_w, const TensorRef& router_b,
                       const gpu::Buffer& x, MoeScratch& s, bool apply_norm = true);

// Dispatch 2: reads s.normed and s.expert_ids, writes s.h. (`forced_ids` is
// accepted for compatibility; the experts always come from s.expert_ids.)
void encode_moe_gate_up(gpu::Device& dev, gpu::CommandStream& cs, const ModelConfig& cfg,
                        const LayerWeights& L, MoeScratch& s, bool forced_ids = false);

// Dispatch 3: reads s.h, s.expert_ids, s.probs; residual += weighted expert outputs.
void encode_moe_down(gpu::Device& dev, gpu::CommandStream& cs, const ModelConfig& cfg,
                     const LayerWeights& L, const gpu::Buffer& residual, MoeScratch& s);

// Bytes the MoE half reads per token for one layer (K experts' gate_up and
// down blocks + scales + bias, router weight + bias, mlp_norm) — for GB/s.
uint64_t moe_decode_bytes_per_layer(const ModelConfig& cfg);

// Kernel shape knobs (defaults tuned on M2 Max with tests/test_moe.cpp
// moe_lab_sweep; tests / tooling may change them).
struct MoeKernelConfig {
    uint32_t gu_pairs = 2;   // (gate, up) row pairs per simdgroup: 1, 2 or 4
    uint32_t gu_sg = 2;      // simdgroups per threadgroup
    uint32_t dn_rows = 4;    // down rows per simdgroup: 1, 2 or 4
    uint32_t dn_sg = 2;
};
MoeKernelConfig& moe_kernel_config();

} // namespace coral
