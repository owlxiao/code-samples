#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <torch/types.h>

namespace elementwise {

// ElementWise Add for float32
__global__ void elementwise_add_f32_kernel(float *a, float *b, float *c,
                                           int N) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    c[idx] = a[idx] + b[idx];
  }
}

// ElementWise Add + Vec4
template <typename T, typename U> __device__ inline T AsType(U &value) {
  return *reinterpret_cast<T *>(&value);
}

template <typename T, typename U> __device__ inline T &AsTypeRef(U &value) {
  return *reinterpret_cast<T *>(&value);
}

__global__ void elementwise_add_f32x4_kernel(float *a, float *b, float *c,
                                             int N) {
  auto idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    float4 reg_a = AsType<float4>(a[idx]);
    float4 reg_b = AsType<float4>(b[idx]);
    float4 reg_c;

    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;
    AsTypeRef<float4>(c[idx]) = reg_c;
  }
}

// -------------------------------------- FP16
__global__ void elementwise_add_f16_kernel(__half *a, __half *b, __half *c,
                                           int N) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    c[idx] = __hadd(a[idx], b[idx]);
  }
}

__global__ void elementwise_add_f16x2_kernel(__half *a, __half *b, __half *c,
                                             int N) {
  auto idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    half2 reg_a = AsType<half2>(a[idx]);
    half2 reg_b = AsType<half2>(b[idx]);
    half2 reg_c;

    reg_c.x = __hadd(reg_a.x, reg_b.x);
    reg_c.y = __hadd(reg_a.y, reg_b.y);
    AsTypeRef<half2>(c[idx]) = reg_c;
  }
}

// --------------------- PyTorch bindings for custom kernel
struct LaunchConfig {
  dim3 block;
  dim3 grid;
  int num;
};

template <int VECTOR_SIZE = 1>
LaunchConfig getLaunchConfig(int S, int K, int VecSize) {
  LaunchConfig config;
  config.num = S * K;
  if (K / VecSize <= 1024) {
    config.block = dim3(K / VecSize);
    config.grid = dim3(S);
  } else {
    config.block = dim3(256 / VecSize);
    config.grid = dim3((config.num + 256 - 1) / 256);
  }

  return config;
}

#define TORCH_BINDING_ELEM_ADD(name, dtype, ctype, vec_size)                   \
  void elementwise_add_##name(torch::Tensor a, torch::Tensor b,                \
                              torch::Tensor c) {                               \
    TORCH_CHECK(a.dim() == 2, "Expected 2D tensor");                           \
    TORCH_CHECK(a.scalar_type() == dtype, "Expected " #dtype " tensor");       \
    auto config = getLaunchConfig(a.size(0), a.size(1), vec_size);             \
    elementwise_add_##name##_kernel<<<config.grid, config.block>>>(            \
        reinterpret_cast<ctype *>(a.data_ptr()),                               \
        reinterpret_cast<ctype *>(b.data_ptr()),                               \
        reinterpret_cast<ctype *>(c.data_ptr()), config.num);                  \
  }

// F32
TORCH_BINDING_ELEM_ADD(f32, torch::kFloat32, float, 1)
TORCH_BINDING_ELEM_ADD(f32x4, torch::kFloat32, float, 4)

// F16
TORCH_BINDING_ELEM_ADD(f16, torch::kHalf, __half, 1)
TORCH_BINDING_ELEM_ADD(f16x2, torch::kHalf, __half, 2)

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32)
        TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32x4)
            TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16)
                TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x2)

};

} // namespace elementwise