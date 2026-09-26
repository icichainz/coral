// Zero-copy safetensors loader.
//
// Each shard is mmap'd read-only and page-aligned, so the whole mapping can be
// handed to gpu::Device::wrap_no_copy and every tensor becomes a (buffer,
// offset) pair. Nothing is ever copied into a separate GPU allocation.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "coral/gpu.h"

namespace coral {

enum class DType : uint8_t { F32, F16, BF16, I64, I32, I16, I8, U8, BOOL, F64, U16, U32, U64, F8_E4M3, F8_E5M2 };
const char* dtype_name(DType t);
size_t dtype_size(DType t);
DType dtype_from_name(const std::string& s);  // throws on unknown

struct TensorInfo {
    std::string name;
    DType dtype = DType::BF16;
    std::vector<int64_t> shape;
    size_t shard = 0;          // index into Safetensors::shards()
    size_t offset = 0;         // byte offset of first element from shard data start
    size_t nbytes = 0;

    int64_t numel() const;
    int64_t dim(size_t i) const { return shape.at(i); }
};

struct TensorView {
    const TensorInfo* info = nullptr;
    const uint8_t*    data = nullptr;   // CPU pointer into the mapping
    gpu::Buffer       buffer;           // whole-shard buffer (valid after bind_gpu)
    size_t            buffer_offset = 0;// offset of the tensor within `buffer`

    template <class T> const T* as() const { return reinterpret_cast<const T*>(data); }
};

class Safetensors {
public:
    struct Shard {
        std::string path;
        void*   map = nullptr;        // page-aligned mmap base
        size_t  map_bytes = 0;
        size_t  data_start = 0;       // 8 + header_len
        gpu::Buffer buffer;           // set by bind_gpu()
    };

    // Open a single .safetensors file or a directory containing
    // model.safetensors.index.json (sharded) / model.safetensors.
    static std::unique_ptr<Safetensors> open(const std::string& path);
    ~Safetensors();

    const std::vector<Shard>& shards() const { return shards_; }
    const std::map<std::string, TensorInfo>& tensors() const { return tensors_; }
    size_t total_bytes() const { return total_bytes_; }

    bool has(const std::string& name) const { return tensors_.count(name) > 0; }
    TensorView get(const std::string& name) const;   // throws if missing

    // Wrap every shard mapping as a GPU buffer (no copy). Idempotent.
    void bind_gpu(gpu::Device& dev);

    // Hint the kernel to page the shards in sequentially (speeds first use).
    void prefetch() const;

private:
    Safetensors() = default;
    void open_shard(const std::string& path);

    std::vector<Shard> shards_;
    std::map<std::string, TensorInfo> tensors_;
    size_t total_bytes_ = 0;
    bool bound_ = false;
};

// bf16 <-> f32 helpers for CPU-side reference math and tests.
inline float bf16_to_f32(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; memcpy(&f, &u, 4); return f; }
inline uint16_t f32_to_bf16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    if ((u & 0x7F800000u) == 0x7F800000u) return uint16_t(u >> 16);  // inf/nan: truncate
    uint32_t lsb = (u >> 16) & 1u;
    return uint16_t((u + 0x7FFFu + lsb) >> 16);                        // round to nearest even
}

} // namespace coral
