// Access to the embedded MSL kernel sources and their one-time compilation.
#pragma once

#include <string_view>

#include "coral/gpu.h"

namespace coral {

// The full MSL translation unit embedded at build time (all src/kernels/*).
std::string_view kernel_source();

// Compile the embedded kernels into `dev`. Throws with the compiler log on error.
// Returns wall-clock seconds spent compiling.
double load_kernels(gpu::Device& dev);

} // namespace coral
