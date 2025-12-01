#pragma once

#include "flash_attention.cuh"

#include <cstdio>
#include <cuda_runtime.h>

namespace flash {

__global__ void
flash_forward_kernel(__grid_constant__ const ForwardKernelArgs args) {}

} // namespace flash
