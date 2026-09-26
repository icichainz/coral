// The one Objective-C++ file in coral. Everything Metal lives here, behind the
// plain C++ interface declared in include/coral/gpu.h.
//
// Metal 4 command model (macOS 26+):
//   MTL4CommandQueue   — accepts MTL4CommandBuffers, tracks residency sets
//   MTL4CommandBuffer  — recorded against a MTL4CommandAllocator
//   MTL4ComputeCommandEncoder — dispatches; NO implicit barriers between them
//   MTL4ArgumentTable  — bindings are raw GPU addresses (bindless style)
//   MTLResidencySet    — every buffer we create is made resident once, up front
//   MTLSharedEvent     — CPU waits for completion
//
// Compared with the Metal 3 encoder model this removes per-dispatch resource
// tracking and hazard analysis from the driver's hot path, which is where the
// host time goes when a decode step issues a few hundred dispatches.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "coral/gpu.h"

#include <mach/mach_time.h>
#include <unistd.h>

#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace coral::gpu {

// ---------------------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------------------
static constexpr uint32_t kMaxBufferBindings   = 32;          // per kernel
static constexpr uint32_t kArgTableRing        = 4096;        // dispatches per submit
static constexpr size_t   kParamRingBytes      = 8u << 20;    // 8 MiB of inline params per submit
static constexpr size_t   kParamAlign          = 256;

static std::string ns_to_std(NSString* s) { return s ? std::string([s UTF8String]) : std::string(); }

[[noreturn]] static void fail(const std::string& what, NSError* err = nil) {
    std::string msg = "coral::gpu: " + what;
    if (err) msg += ": " + ns_to_std(err.localizedDescription);
    throw std::runtime_error(msg);
}

size_t page_size() { return static_cast<size_t>(getpagesize()); }

// ---------------------------------------------------------------------------
// Buffer
// ---------------------------------------------------------------------------
// Residency set shared between the device and every buffer, so a buffer can
// leave the set when its last handle dies even if the device is already gone.
struct Residency {
    id<MTLResidencySet> set = nil;
    std::mutex mutex;
    void add(id<MTLBuffer> b)    { std::lock_guard<std::mutex> l(mutex); [set addAllocation:b];    [set commit]; }
    void remove(id<MTLBuffer> b) { std::lock_guard<std::mutex> l(mutex); [set removeAllocation:b]; [set commit]; }
};

struct Buffer::Impl {
    id<MTLBuffer> mtl = nil;
    size_t size = 0;
    std::weak_ptr<Residency> residency;
    ~Impl() {
        if (auto r = residency.lock()) r->remove(mtl);
    }
};

void*    Buffer::data() const        { return impl_ ? impl_->mtl.contents : nullptr; }
size_t   Buffer::size() const        { return impl_ ? impl_->size : 0; }
uint64_t Buffer::gpu_address() const { return impl_ ? impl_->mtl.gpuAddress : 0; }

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
struct Kernel::Impl {
    id<MTLComputePipelineState> pso = nil;
    std::string name;
};

const std::string& Kernel::name() const { return impl_->name; }
uint32_t Kernel::thread_execution_width() const { return (uint32_t)impl_->pso.threadExecutionWidth; }
uint32_t Kernel::max_threads_per_threadgroup() const { return (uint32_t)impl_->pso.maxTotalThreadsPerThreadgroup; }

// ---------------------------------------------------------------------------
// Args
// ---------------------------------------------------------------------------
Args& Args::buffer(uint32_t index, const Buffer& b, size_t offset) {
    if (!b.valid()) throw std::invalid_argument("coral::gpu::Args: invalid buffer");
    if (offset > b.size()) throw std::out_of_range("coral::gpu::Args: buffer offset out of range");
    bindings_.push_back({index, b.gpu_address() + offset, nullptr, 0});
    max_index_ = std::max(max_index_, index);
    return *this;
}
Args& Args::bytes(uint32_t index, const void* p, size_t n) {
    bindings_.push_back({index, 0, p, n});
    max_index_ = std::max(max_index_, index);
    return *this;
}

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------
struct Device::Impl {
    id<MTLDevice>          device = nil;
    id<MTL4CommandQueue>   queue = nil;
    id<MTL4Compiler>       compiler = nil;
    std::shared_ptr<Residency> residency;
    id<MTLLibrary>         library = nil;
    id<MTLSharedEvent>     event = nil;
    uint64_t               event_value = 0;      // monotonically increasing signal value
    DeviceInfo             info;
    std::mutex             mutex;                // guards pipeline cache + event value
    std::unordered_map<std::string, Kernel> pipelines;

    Buffer make_buffer(id<MTLBuffer> b, size_t size) {
        residency->add(b);
        Buffer out;
        out.impl_ = std::make_shared<Buffer::Impl>();
        out.impl_->mtl = b;
        out.impl_->size = size;
        out.impl_->residency = residency;
        return out;
    }
};

Device::Device() : impl_(std::make_unique<Impl>()) {}
Device::~Device() = default;

std::unique_ptr<Device> Device::create() {
    @autoreleasepool {
        auto dev = std::unique_ptr<Device>(new Device());
        Impl& I = *dev->impl_;
        I.device = MTLCreateSystemDefaultDevice();
        if (!I.device) fail("no Metal device");
        if (![I.device supportsFamily:MTLGPUFamilyMetal4])
            fail("this GPU/OS does not support the Metal 4 family (requires Apple silicon + macOS 26)");

        NSError* err = nil;
        MTL4CommandQueueDescriptor* qd = [MTL4CommandQueueDescriptor new];
        qd.label = @"coral.queue";
        I.queue = [I.device newMTL4CommandQueueWithDescriptor:qd error:&err];
        if (!I.queue) fail("newMTL4CommandQueue", err);

        MTL4CompilerDescriptor* cd = [MTL4CompilerDescriptor new];
        cd.label = @"coral.compiler";
        I.compiler = [I.device newCompilerWithDescriptor:cd error:&err];
        if (!I.compiler) fail("newCompiler", err);

        MTLResidencySetDescriptor* rd = [MTLResidencySetDescriptor new];
        rd.label = @"coral.residency";
        rd.initialCapacity = 1024;
        I.residency = std::make_shared<Residency>();
        I.residency->set = [I.device newResidencySetWithDescriptor:rd error:&err];
        if (!I.residency->set) fail("newResidencySet", err);
        [I.queue addResidencySet:I.residency->set];

        I.event = [I.device newSharedEvent];
        if (!I.event) fail("newSharedEvent");

        I.info.name = ns_to_std(I.device.name);
        I.info.max_buffer_bytes = I.device.maxBufferLength;
        I.info.recommended_working_set_bytes = I.device.recommendedMaxWorkingSetSize;
        I.info.unified_memory = I.device.hasUnifiedMemory;
        I.info.metal4 = true;
        I.info.max_threads_per_threadgroup = (uint32_t)I.device.maxThreadsPerThreadgroup.width;
        I.info.max_threadgroup_memory_bytes = (uint32_t)I.device.maxThreadgroupMemoryLength;
        I.info.simd_width = 32;  // all Apple GPUs; confirmed per-pipeline via threadExecutionWidth
        return dev;
    }
}

const DeviceInfo& Device::info() const { return impl_->info; }

Buffer Device::alloc(size_t bytes, bool zero) {
    @autoreleasepool {
        if (bytes == 0) bytes = 1;
        id<MTLBuffer> b = [impl_->device newBufferWithLength:bytes
                                                     options:MTLResourceStorageModeShared |
                                                             MTLResourceHazardTrackingModeUntracked];
        if (!b) fail("newBufferWithLength(" + std::to_string(bytes) + ")");
        if (zero) memset(b.contents, 0, bytes);
        return impl_->make_buffer(b, bytes);
    }
}

Buffer Device::wrap_no_copy(void* ptr, size_t bytes) {
    @autoreleasepool {
        const size_t pg = page_size();
        if ((reinterpret_cast<uintptr_t>(ptr) % pg) != 0) fail("wrap_no_copy: pointer is not page-aligned");
        size_t rounded = (bytes + pg - 1) / pg * pg;
        id<MTLBuffer> b = [impl_->device newBufferWithBytesNoCopy:ptr
                                                           length:rounded
                                                          options:MTLResourceStorageModeShared |
                                                                  MTLResourceHazardTrackingModeUntracked
                                                      deallocator:nil];
        if (!b) fail("newBufferWithBytesNoCopy(" + std::to_string(rounded) + ")");
        return impl_->make_buffer(b, bytes);
    }
}

void Device::compile_library(std::string_view msl, const std::string& name) {
    @autoreleasepool {
        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.mathMode = MTLMathModeFast;
        opts.languageVersion = MTLLanguageVersion3_2;
        MTL4LibraryDescriptor* ld = [MTL4LibraryDescriptor new];
        ld.source = [[NSString alloc] initWithBytes:msl.data() length:msl.size() encoding:NSUTF8StringEncoding];
        ld.options = opts;
        ld.name = [NSString stringWithUTF8String:name.c_str()];
        NSError* err = nil;
        id<MTLLibrary> lib = [impl_->compiler newLibraryWithDescriptor:ld error:&err];
        if (!lib) fail("MSL compile failed", err);
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->library = lib;
        impl_->pipelines.clear();
    }
}

void Device::make_resident(double timeout_seconds) {
    CommandStream cs = stream();
    cs.begin();
    cs.submit();
    cs.wait(timeout_seconds);
}

Kernel Device::kernel(const std::string& fn) {
    @autoreleasepool {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (!impl_->library) fail("kernel(" + fn + "): no library compiled");
        if (auto it = impl_->pipelines.find(fn); it != impl_->pipelines.end()) return it->second;

        MTL4LibraryFunctionDescriptor* fd = [MTL4LibraryFunctionDescriptor new];
        fd.name = [NSString stringWithUTF8String:fn.c_str()];
        fd.library = impl_->library;
        MTL4ComputePipelineDescriptor* pd = [MTL4ComputePipelineDescriptor new];
        pd.computeFunctionDescriptor = fd;
        pd.label = fd.name;
        pd.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
        NSError* err = nil;
        id<MTLComputePipelineState> pso =
            [impl_->compiler newComputePipelineStateWithDescriptor:pd compilerTaskOptions:nil error:&err];
        if (!pso) fail("pipeline for '" + fn + "'", err);

        Kernel k;
        k.impl_ = std::make_shared<Kernel::Impl>();
        k.impl_->pso = pso;
        k.impl_->name = fn;
        impl_->pipelines.emplace(fn, k);
        return k;
    }
}

// ---------------------------------------------------------------------------
// CommandStream
// ---------------------------------------------------------------------------
struct CommandStream::Impl {
    Device::Impl*                     dev = nullptr;
    id<MTL4CommandAllocator>          allocator = nil;
    id<MTL4CommandBuffer>             cmd = nil;
    id<MTL4ComputeCommandEncoder>     enc = nil;
    NSMutableArray<id<MTL4ArgumentTable>>* tables = nil;  // ring
    Buffer                            params;             // ring for inline params
    size_t                            param_head = 0;
    uint32_t                          table_head = 0;
    size_t                            dispatches = 0;
    bool                              recording = false;
    bool                              pending = false;      // submitted, not yet waited
    uint64_t                          signal_value = 0;
    uint64_t                          t_submit = 0;
    bool                              need_barrier = false;

    id<MTL4ArgumentTable> next_table() {
        if (table_head >= kArgTableRing)
            fail("too many dispatches in one submit (" + std::to_string(kArgTableRing) + ")");
        return tables[table_head++];
    }

    uint64_t place_params(const void* p, size_t n) {
        size_t aligned = (n + kParamAlign - 1) / kParamAlign * kParamAlign;
        if (param_head + aligned > params.size()) fail("inline parameter ring exhausted in one submit");
        memcpy(static_cast<uint8_t*>(params.data()) + param_head, p, n);
        uint64_t addr = params.gpu_address() + param_head;
        param_head += aligned;
        return addr;
    }
};

CommandStream::CommandStream(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CommandStream::~CommandStream() {
    if (!impl_ || !impl_->pending) return;
    try { wait(); } catch (...) { /* never throw from a destructor; the GPU work is orphaned */ }
}
CommandStream::CommandStream(CommandStream&&) noexcept = default;
CommandStream& CommandStream::operator=(CommandStream&&) noexcept = default;

CommandStream Device::stream() {
    @autoreleasepool {
        auto I = std::make_unique<CommandStream::Impl>();
        I->dev = impl_.get();
        I->allocator = [impl_->device newCommandAllocator];
        if (!I->allocator) fail("newCommandAllocator");
        I->cmd = [impl_->device newCommandBuffer];
        if (!I->cmd) fail("newCommandBuffer");

        MTL4ArgumentTableDescriptor* td = [MTL4ArgumentTableDescriptor new];
        td.maxBufferBindCount = kMaxBufferBindings;
        td.initializeBindings = NO;
        I->tables = [NSMutableArray arrayWithCapacity:kArgTableRing];
        for (uint32_t i = 0; i < kArgTableRing; ++i) {
            NSError* err = nil;
            id<MTL4ArgumentTable> t = [impl_->device newArgumentTableWithDescriptor:td error:&err];
            if (!t) fail("newArgumentTable", err);
            [I->tables addObject:t];
        }
        I->params = alloc(kParamRingBytes);
        return CommandStream(std::move(I));
    }
}

void CommandStream::begin() {
    Impl& I = *impl_;
    if (I.recording) fail("CommandStream::begin: already recording");
    if (I.pending) wait();
    [I.allocator reset];
    [I.cmd beginCommandBufferWithAllocator:I.allocator];
    I.enc = [I.cmd computeCommandEncoder];
    if (!I.enc) fail("computeCommandEncoder");
    // Metal 4 command buffers committed to one queue may overlap. Order this
    // command buffer's work after everything committed before it (GPU-side,
    // no CPU wait), so a stream can consume results of another stream's
    // still-running submit (e.g. the next decode step reading this step's
    // argmax) and streams never race on shared scratch buffers.
    [I.enc barrierAfterQueueStages:MTLStageDispatch | MTLStageBlit
                      beforeStages:MTLStageDispatch | MTLStageBlit
                 visibilityOptions:MTL4VisibilityOptionDevice];
    I.param_head = 0;
    I.table_head = 0;
    I.dispatches = 0;
    I.need_barrier = false;
    I.recording = true;
}

void CommandStream::barrier() {
    Impl& I = *impl_;
    [I.enc barrierAfterEncoderStages:MTLStageDispatch
                 beforeEncoderStages:MTLStageDispatch
                   visibilityOptions:MTL4VisibilityOptionNone];
    I.need_barrier = false;
}

static void bind_and_dispatch(CommandStream::Impl& I, id<MTLComputePipelineState> pso, const Args& args,
                              Grid tg_count, Grid tg_size, bool independent, bool by_threads) {
    if (!I.recording) fail("dispatch outside begin()/submit()");
    if (!pso) fail("dispatch: invalid kernel");
    if (args.max_index() >= kMaxBufferBindings) fail("dispatch: buffer index exceeds argument table size");
    if (I.need_barrier && !independent) {
        [I.enc barrierAfterEncoderStages:MTLStageDispatch
                     beforeEncoderStages:MTLStageDispatch
                       visibilityOptions:MTL4VisibilityOptionNone];
    }
    id<MTL4ArgumentTable> table = I.next_table();
    for (const auto& b : args.bindings()) {
        uint64_t addr = b.bytes ? I.place_params(b.bytes, b.size) : b.address;
        [table setAddress:addr atIndex:b.index];
    }
    [I.enc setComputePipelineState:pso];
    [I.enc setArgumentTable:table];
    MTLSize tgs = MTLSizeMake(tg_size.x, tg_size.y, tg_size.z);
    if (by_threads) {
        MTLSize threads = MTLSizeMake(tg_count.x, tg_count.y, tg_count.z);
        [I.enc dispatchThreads:threads threadsPerThreadgroup:tgs];
    } else {
        MTLSize groups = MTLSizeMake(tg_count.x, tg_count.y, tg_count.z);
        [I.enc dispatchThreadgroups:groups threadsPerThreadgroup:tgs];
    }
    I.dispatches++;
    I.need_barrier = true;
}

void CommandStream::dispatch(const Kernel& k, const Args& a, Grid groups, Grid tg, bool independent) {
    bind_and_dispatch(*impl_, k.valid() ? k.impl_->pso : nil, a, groups, tg, independent, false);
}
void CommandStream::dispatch_threads(const Kernel& k, const Args& a, Grid threads, Grid tg, bool independent) {
    bind_and_dispatch(*impl_, k.valid() ? k.impl_->pso : nil, a, threads, tg, independent, true);
}

void CommandStream::fill(const Buffer& b, size_t offset, size_t size, uint8_t value) {
    Impl& I = *impl_;
    if (!I.recording) fail("fill outside begin()/submit()");
    if (I.need_barrier) barrier();
    [I.enc fillBuffer:b.impl_->mtl range:NSMakeRange(offset, size) value:value];
    I.need_barrier = true;
}

void CommandStream::submit() {
    @autoreleasepool {
        Impl& I = *impl_;
        if (!I.recording) fail("submit without begin");
        [I.enc endEncoding];
        I.enc = nil;
        [I.cmd endCommandBuffer];
        I.recording = false;

        id<MTL4CommandBuffer> cbs[1] = { I.cmd };
        {
            std::lock_guard<std::mutex> lock(I.dev->mutex);
            I.signal_value = ++I.dev->event_value;
            [I.dev->queue commit:cbs count:1];
            [I.dev->queue signalEvent:I.dev->event value:I.signal_value];
        }
        I.t_submit = mach_absolute_time();
        I.pending = true;
    }
}

double CommandStream::wait(double timeout_seconds) {
    Impl& I = *impl_;
    if (!I.pending) return 0.0;
    uint64_t ms = timeout_seconds <= 0 ? 1 : uint64_t(timeout_seconds * 1000.0);
    BOOL ok = [I.dev->event waitUntilSignaledValue:I.signal_value timeoutMS:ms];
    if (!ok) fail("GPU timeout (" + std::to_string(timeout_seconds) + " s) waiting for command buffer");
    I.pending = false;
    uint64_t t1 = mach_absolute_time();
    static mach_timebase_info_data_t tb = [] { mach_timebase_info_data_t t; mach_timebase_info(&t); return t; }();
    return double(t1 - I.t_submit) * tb.numer / tb.denom * 1e-9;
}

size_t CommandStream::dispatch_count() const { return impl_->dispatches; }

} // namespace coral::gpu
