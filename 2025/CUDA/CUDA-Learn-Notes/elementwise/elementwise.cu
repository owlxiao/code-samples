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

// --------------------- PyTorch bindings for custom kernel
void elementwise_add_f32(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  const auto ndim = a.dim();
  assert(ndim == 2);

  const auto S = a.size(0);
  const auto K = a.size(1);
  const auto N = S * K;

  if (K < 1024) {
    dim3 block(K);
    dim3 grid(S);
    elementwise_add_f32_kernel<<<grid, block>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), N);
  } else {
    int N = 1;
    for (int i = 0; i < ndim; ++i)
      N *= a.size(i);
    dim3 block(256);
    dim3 grid((N + 256 - 1) / 256);
    elementwise_add_f32_kernel<<<grid, block>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), N);
  }
}

void elementwise_add_f32x4(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  const auto ndim = a.dim();
  assert(ndim == 2);

  const auto S = a.size(0);
  const auto K = a.size(1);
  const auto N = S * K;

  if ((K / elements) <= 1024) {
    dim3 block(K / elements);
    dim3 grid(S);
    elementwise_add_f32x4_kernel<<<grid, block>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), N);
  } else {
    int N = 1;
    for (int i = 0; i < ndim; ++i)
      N *= a.size(i);
    dim3 block(256 / elements);
    dim3 grid((N + 256 - 1) / 256);
    elementwise_add_f32x4_kernel<<<grid, block>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), N);
  }
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

PYBIND11_MODULE(TORCH_EXTENSION_NAME,
                m){TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32)
                       TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32x4)

};

} // namespace elementwise