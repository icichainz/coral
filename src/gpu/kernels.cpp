#include "coral/kernels.h"

#include <chrono>

#include "shaders.inc"  // generated: build/gen/shaders.inc

namespace coral {

std::string_view kernel_source() { return shaders::kSource; }

double load_kernels(gpu::Device& dev) {
    auto t0 = std::chrono::steady_clock::now();
    dev.compile_library(kernel_source(), "coral_kernels");
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

} // namespace coral
