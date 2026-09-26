#include "common.h"

// Decode-path token embedding straight into the fp32 residual stream:
//   out[i] = float(table[id][i]),  i < dim
//
//   table : bf16 [vocab][dim]  (zero-copy safetensors shard)
//   ids   : int32 token id buffer, read only when p.from_buffer != 0
//           (lets a previous step's GPU argmax feed the next step directly)
//   out   : fp32 [dim]
// The token id otherwise comes by value in the params, so the host never has
// to stage it in a buffer. Few threadgroups, 4 elements per thread.

struct EmbedF32Params { int token; uint dim; uint from_buffer; uint ids_index; };

kernel void embed_gather_f32(device const bfloat*     table [[buffer(0)]],
                             device const int*        ids   [[buffer(1)]],
                             device float*            out   [[buffer(2)]],
                             constant EmbedF32Params& p     [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    const uint i = gid * 4;
    if (i >= p.dim) return;
    const int id = p.from_buffer ? ids[p.ids_index] : p.token;
    device const bfloat* src = table + (ulong)uint(id) * p.dim;
    if (i + 4 <= p.dim) {
        const float4 v = float4(src[i], src[i + 1], src[i + 2], src[i + 3]);
        *((device float4*)(out + i)) = v;
    } else {
        for (uint j = i; j < p.dim; ++j) out[j] = float(src[j]);
    }
}
