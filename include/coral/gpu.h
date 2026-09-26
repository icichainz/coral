// coral::gpu — the plain C++ face of the GPU.
//
// This header is the only thing the rest of the engine sees. Its single
// implementation, src/gpu/metal_backend.mm, is the one Objective-C++ file in
// the tree and owns every Metal call (Metal 4 command model: command
// allocators, argument tables, residency sets, shared-event completion).
//
// Design rules that shape this interface:
//   * Unified memory. Every Buffer is CPU-visible; there is no upload/download.
//   * Weights are mmap'd and wrapped without copying (wrap_no_copy).
//   * Kernel parameters are small structs passed by value; the backend places
//     them in a ring buffer because Metal 4 has no "set bytes".
//   * A CommandStream records many dispatches and submits them as one command
//     buffer. Decode issues one submit per token (or per micro-batch).
//   * Consecutive dispatches are ordered by an explicit barrier unless the
//     caller marks a dispatch as independent of the previous one.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace coral::gpu {

struct DeviceInfo {
    std::string name;
    uint64_t max_buffer_bytes = 0;
    uint64_t recommended_working_set_bytes = 0;
    bool unified_memory = false;
    bool metal4 = false;
    uint32_t simd_width = 32;
    uint32_t max_threads_per_threadgroup = 1024;
    uint32_t max_threadgroup_memory_bytes = 32768;
};

struct Grid {
    uint32_t x = 1, y = 1, z = 1;
    constexpr Grid() = default;
    constexpr Grid(uint32_t x_, uint32_t y_ = 1, uint32_t z_ = 1) : x(x_), y(y_), z(z_) {}
};

class Device;

// A GPU-visible allocation. Cheap to copy (shared handle).
class Buffer {
public:
    Buffer() = default;
    void*    data() const;                 // CPU pointer (shared storage)
    size_t   size() const;
    uint64_t gpu_address() const;          // 64-bit device address (Metal 4 binding)
    bool     valid() const { return impl_ != nullptr; }

    template <class T> T* as() const { return static_cast<T*>(data()); }

    struct Impl;
private:
    friend class Device;
    friend class CommandStream;
    std::shared_ptr<Impl> impl_;
};

// A compiled compute pipeline.
class Kernel {
public:
    Kernel() = default;
    const std::string& name() const;
    uint32_t thread_execution_width() const;
    uint32_t max_threads_per_threadgroup() const;
    bool valid() const { return impl_ != nullptr; }

    struct Impl;
private:
    friend class Device;
    friend class CommandStream;
    std::shared_ptr<Impl> impl_;
};

// Bindings for one dispatch. Index = [[buffer(N)]] slot in the kernel.
class Args {
public:
    Args& buffer(uint32_t index, const Buffer& b, size_t offset = 0);
    // Copies `bytes` into a per-stream ring buffer and binds it at `index`.
    Args& bytes(uint32_t index, const void* p, size_t bytes);
    template <class T> Args& value(uint32_t index, const T& v) { return bytes(index, &v, sizeof(T)); }

    struct Binding { uint32_t index; uint64_t address; const void* bytes; size_t size; };
    const std::vector<Binding>& bindings() const { return bindings_; }
    uint32_t max_index() const { return max_index_; }
private:
    std::vector<Binding> bindings_;
    uint32_t max_index_ = 0;
};

// Records dispatches, submits them as one command buffer, and waits.
// All streams of a Device commit to one queue, and each submit's work starts
// only after all work submitted before it (from any stream) has completed:
// a submit may consume, on the GPU, results of an earlier submit that the CPU
// has not waited for yet.
class CommandStream {
public:
    ~CommandStream();
    CommandStream(CommandStream&&) noexcept;
    CommandStream& operator=(CommandStream&&) noexcept;

    // Begin recording. Must be balanced by submit().
    void begin();

    // Enqueue `kernel` over `grid` threadgroups of `threadgroup` threads.
    // A barrier is inserted after the previous dispatch unless `independent`.
    void dispatch(const Kernel& kernel, const Args& args, Grid threadgroups, Grid threadgroup,
                  bool independent = false);

    // Same, but sized in threads; the backend rounds up to whole threadgroups
    // and the kernel must guard its tail.
    void dispatch_threads(const Kernel& kernel, const Args& args, Grid threads, Grid threadgroup,
                          bool independent = false);

    // Explicit compute->compute barrier (rarely needed by callers).
    void barrier();

    // GPU fill of a buffer range with a byte value.
    void fill(const Buffer& b, size_t offset, size_t size, uint8_t value);

    // Submit the recorded work. Returns immediately.
    void submit();
    // Block until the last submit completed; throws after `timeout_seconds`
    // (the submit stays pending and wait() may be called again). Returns the
    // wall time from submit to completion in seconds.
    double wait(double timeout_seconds = 600.0);
    // submit() + wait().
    double submit_and_wait(double timeout_seconds = 600.0) { submit(); return wait(timeout_seconds); }

    size_t dispatch_count() const;

    struct Impl;
private:
    friend class Device;
    explicit CommandStream(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class Device {
public:
    // Create the system default GPU device. Throws std::runtime_error on failure.
    static std::unique_ptr<Device> create();
    ~Device();

    const DeviceInfo& info() const;

    // Allocate `bytes` of shared (CPU+GPU) memory. Zero-initialized if `zero`.
    Buffer alloc(size_t bytes, bool zero = false);

    // Wrap page-aligned memory (e.g. an mmap'd weight shard) as a GPU buffer
    // with no copy. `ptr` must be page-aligned; `bytes` is rounded up to a
    // page. The caller keeps the mapping alive for the buffer's lifetime.
    Buffer wrap_no_copy(void* ptr, size_t bytes);

    // Every Buffer joins the device residency set on creation and leaves it
    // when its last handle is released. The first command buffer after new
    // buffers join pages them in and wires them, which for 13 GB of weights on
    // a cold page cache can take a minute. make_resident() pays that cost now
    // (submits an empty command buffer and waits) so no later dispatch does.
    void make_resident(double timeout_seconds = 900.0);

    // Compile an MSL translation unit. `name` is used in diagnostics.
    // Throws std::runtime_error with the compiler log on failure.
    void compile_library(std::string_view msl_source, const std::string& name = "coral");

    // Look up a kernel function in the compiled library and build its pipeline.
    // Pipelines are cached by name.
    Kernel kernel(const std::string& function_name);

    CommandStream stream();

    struct Impl;
private:
    Device();
    std::unique_ptr<Impl> impl_;
};

// Page size used for wrap_no_copy alignment.
size_t page_size();

} // namespace coral::gpu
