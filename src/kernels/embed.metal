#include "common.h"

// Token embedding lookup: out[t] = table[ids[t]].
//
//   table : bf16 [vocab][dim]   (bound zero-copy from the safetensors shard)
//   ids   : int32 [n_tokens]
//   out   : bf16 [n_tokens][dim]
// One threadgroup per token; threads stride across the row.

struct EmbedParams { uint n_tokens; uint dim; };

kernel void embed_gather_bf16(device const bfloat*  table [[buffer(0)]],
                              device const int*     ids   [[buffer(1)]],
                              device bfloat*        out   [[buffer(2)]],
                              constant EmbedParams& p     [[buffer(3)]],
                              uint  t       [[threadgroup_position_in_grid]],
                              uint  tid     [[thread_position_in_threadgroup]],
                              uint  tg_size [[threads_per_threadgroup]]) {
    if (t >= p.n_tokens) return;
    device const bfloat* src = table + (ulong)uint(ids[t]) * p.dim;
    device bfloat*       dst = out + (ulong)t * p.dim;
    for (uint i = tid; i < p.dim; i += tg_size) dst[i] = src[i];
}
