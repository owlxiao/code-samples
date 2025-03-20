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

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

PYBIND11_MODULE(TORCH_EXTENSION_NAME,
                m){TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32)};

} // namespace elementwise