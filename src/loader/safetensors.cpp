#include "coral/safetensors.h"
#include "coral/json.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace coral {

// ---------------------------------------------------------------------------
// dtype table
// ---------------------------------------------------------------------------
namespace {
struct DTypeEntry { DType t; const char* name; size_t size; };
constexpr DTypeEntry kDTypes[] = {
    {DType::F32, "F32", 4}, {DType::F16, "F16", 2}, {DType::BF16, "BF16", 2}, {DType::I64, "I64", 8},
    {DType::I32, "I32", 4}, {DType::I16, "I16", 2}, {DType::I8, "I8", 1},     {DType::U8, "U8", 1},
    {DType::BOOL, "BOOL", 1}, {DType::F64, "F64", 8}, {DType::U16, "U16", 2}, {DType::U32, "U32", 4},
    {DType::U64, "U64", 8}, {DType::F8_E4M3, "F8_E4M3", 1}, {DType::F8_E5M2, "F8_E5M2", 1},
};
} // namespace

const char* dtype_name(DType t) { for (auto& e : kDTypes) if (e.t == t) return e.name; return "?"; }
size_t dtype_size(DType t) { for (auto& e : kDTypes) if (e.t == t) return e.size; return 0; }
DType dtype_from_name(const std::string& s) {
    for (auto& e : kDTypes) if (s == e.name) return e.t;
    throw std::runtime_error("safetensors: unknown dtype '" + s + "'");
}

int64_t TensorInfo::numel() const { int64_t n = 1; for (auto d : shape) n *= d; return n; }

// ---------------------------------------------------------------------------
// Safetensors
// ---------------------------------------------------------------------------
Safetensors::~Safetensors() {
    for (auto& s : shards_) if (s.map) munmap(s.map, s.map_bytes);
}

std::unique_ptr<Safetensors> Safetensors::open(const std::string& path) {
    auto st = std::unique_ptr<Safetensors>(new Safetensors());
    fs::path p(path);
    if (fs::is_directory(p)) {
        fs::path index = p / "model.safetensors.index.json";
        if (fs::exists(index)) {
            std::ifstream in(index);
            std::stringstream ss; ss << in.rdbuf();
            Json j = Json::parse(ss.str());
            // Distinct shard files in a stable order.
            std::map<std::string, int> files;
            for (const auto& [tensor, file] : j["weight_map"].as_object()) files.emplace(file.as_string(), 0);
            for (auto& [file, _] : files) st->open_shard((p / file).string());
        } else if (fs::exists(p / "model.safetensors")) {
            st->open_shard((p / "model.safetensors").string());
        } else {
            throw std::runtime_error("safetensors: no model.safetensors(.index.json) in " + path);
        }
    } else {
        st->open_shard(path);
    }
    return st;
}

void Safetensors::open_shard(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("safetensors: cannot open " + path);
    struct stat sb{};
    if (fstat(fd, &sb) != 0) { ::close(fd); throw std::runtime_error("safetensors: fstat failed on " + path); }
    size_t bytes = size_t(sb.st_size);
    void* map = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (map == MAP_FAILED) throw std::runtime_error("safetensors: mmap failed on " + path);

    Shard sh;
    sh.path = path;
    sh.map = map;
    sh.map_bytes = bytes;

    if (bytes < 8) { munmap(map, bytes); throw std::runtime_error("safetensors: file too small: " + path); }
    uint64_t header_len = 0;
    memcpy(&header_len, map, 8);
    if (8 + header_len > bytes) { munmap(map, bytes); throw std::runtime_error("safetensors: bad header length in " + path); }
    sh.data_start = 8 + size_t(header_len);

    std::string_view header(static_cast<const char*>(map) + 8, size_t(header_len));
    Json j = Json::parse(header);
    size_t shard_idx = shards_.size();
    for (const auto& [name, meta] : j.as_object()) {
        if (name == "__metadata__") continue;
        TensorInfo t;
        t.name = name;
        t.dtype = dtype_from_name(meta["dtype"].as_string());
        for (const auto& d : meta["shape"].as_array()) t.shape.push_back(d.as_int());
        const auto& off = meta["data_offsets"].as_array();
        size_t b = size_t(off[0].as_int()), e = size_t(off[1].as_int());
        if (e < b || sh.data_start + e > bytes)
            throw std::runtime_error("safetensors: tensor '" + name + "' out of bounds in " + path);
        if (e - b != size_t(t.numel()) * dtype_size(t.dtype))
            throw std::runtime_error("safetensors: tensor '" + name + "' size mismatch in " + path);
        t.shard = shard_idx;
        t.offset = b;
        t.nbytes = e - b;
        total_bytes_ += t.nbytes;
        if (tensors_.count(name)) throw std::runtime_error("safetensors: duplicate tensor '" + name + "'");
        tensors_.emplace(name, std::move(t));
    }
    shards_.push_back(std::move(sh));
}

TensorView Safetensors::get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("safetensors: missing tensor '" + name + "'");
    const TensorInfo& t = it->second;
    const Shard& sh = shards_[t.shard];
    TensorView v;
    v.info = &t;
    v.data = static_cast<const uint8_t*>(sh.map) + sh.data_start + t.offset;
    v.buffer = sh.buffer;
    v.buffer_offset = sh.data_start + t.offset;
    return v;
}

void Safetensors::bind_gpu(gpu::Device& dev) {
    if (bound_) return;
    for (auto& sh : shards_) sh.buffer = dev.wrap_no_copy(sh.map, sh.map_bytes);
    bound_ = true;
}

void Safetensors::prefetch() const {
    for (const auto& sh : shards_) madvise(sh.map, sh.map_bytes, MADV_WILLNEED);
}

} // namespace coral
