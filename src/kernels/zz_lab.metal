#include "common.h"

// Development aid (tests/test_moe.cpp moe_lab_roofline, CORAL_MOE_LAB=1):
// raw streaming-read bandwidth, the roofline for the GEMV kernels, and the
// fixed cost of an empty dispatch + barrier. Not used by the engine.
// lab_stream_read: grid-stride loop, one uint4 per thread per iteration.
struct LabReadParams { uint n16; uint unroll; };

kernel void lab_stream_read(device const uint4* src [[buffer(0)]],
                            device uint*        out [[buffer(1)]],
                            constant LabReadParams& p [[buffer(2)]],
                            uint gid [[thread_position_in_grid]],
                            uint gsz [[threads_per_grid]]) {
    uint4 a = 0;
    for (uint i = gid; i < p.n16; i += gsz) a ^= src[i];
    if (a.x == 0x12345678u && a.y == 0x9abcdef0u) out[0] = a.z;
}

// Contiguous chunk per simdgroup (like a GEMV row run): simdgroup g reads
// chunk g of `chunk16` uint4s.
kernel void lab_chunk_read(device const uint4* src [[buffer(0)]],
                           device uint*        out [[buffer(1)]],
                           constant LabReadParams& p [[buffer(2)]],
                           uint tg   [[threadgroup_position_in_grid]],
                           uint sg   [[simdgroup_index_in_threadgroup]],
                           uint nsg  [[simdgroups_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    const uint chunk = p.unroll;   // uint4 per simdgroup chunk
    const ulong base = ulong(tg * nsg + sg) * chunk;
    uint4 a = 0;
    for (uint i = lane; i < chunk; i += 32) if (base + i < p.n16) a ^= src[base + i];
    if (a.x == 0x12345678u && a.y == 0x9abcdef0u) out[0] = a.z;
}

kernel void lab_empty(device uint* out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    if (gid == 0xFFFFFFFFu) out[0] = 1;
}
