#include "forward_kernel.cuh"

#include <torch/types.h>

#include <cuda_runtime.h>

torch::Tensor forward(torch::Tensor &q, torch::Tensor &k, torch::Tensor &v) {
  auto check_input = [&](torch::Tensor &in, const char *name) {
    TORCH_CHECK(in.device().is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(in.is_contiguous(), name, " must be contiguous");
  };

  check_input(q, "q");
  check_input(k, "k");
  check_input(v, "v");

  // Check data types
  const auto Q_dtype = q.dtype();
  TORCH_CHECK(Q_dtype == torch::kFloat16 || Q_dtype == torch::kBFloat16,
              "Only fp16 and bf16 are supported");
  TORCH_CHECK(k.dtype() == Q_dtype,
              "Input tensors must have the same data type");
  TORCH_CHECK(v.dtype() == Q_dtype,
              "Input tensors must have the same data type");

  // Get shapes
  const auto batch_size = q.size(0);
  const auto num_heads = q.size(1);
  const auto seq_len = q.size(2);
  const auto head_dim = q.size(3);

  TORCH_CHECK(q.sizes() == k.sizes(), "q and k must have the same shape");
  TORCH_CHECK(q.sizes() == v.sizes(), "q and v must have the same shape");

  //
  flash::FlashForwardKernelConfig Config = {
      .HeadDim = 128,
      .Br = 64,
      .Bc = 64,
      .Warps = 4,
  };

  const int Br = Config.Br;
  const int Bc = Config.Bc;
  TORCH_CHECK(seq_len % Br == 0, "seq_len must be divisible by Br");
  TORCH_CHECK(seq_len % Bc == 0, "seq_len must be divisible by Bc");

  const auto batch_stride = q.stride(0);
  const auto head_stride = q.stride(1);
  const auto seq_stride = q.stride(2);
  const auto dim_stride = q.stride(3);

  torch::Tensor O = torch::empty_like(q);

  auto ceil_div = [](auto m, auto n) { return (m + n - 1) / n; };

  const int q_blocks = ceil_div(seq_len, Br);
  const int kv_blocks = ceil_div(seq_len, Bc);
  const int n_threads = 32 * Config.Warps;

  flash::ForwardKernelArgs args{q.data_ptr(), k.data_ptr(), v.data_ptr(),
                                O.data_ptr(), batch_stride, seq_stride,
                                head_stride,  seq_len,      num_heads,
                                q_blocks,     kv_blocks};

  // Launch kernel
  dim3 blockDim(n_threads);
  dim3 gridDim(static_cast<uint>(q_blocks), static_cast<uint>(num_heads),
               static_cast<uint>(batch_size));

  const int smem_bytes = Config.smem_bytes();

  // Get max smem size
  // Check device shared memory limits
  int max_possible_shared = 0;
  cudaDeviceGetAttribute(&max_possible_shared,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);

  TORCH_CHECK(smem_bytes <= max_possible_shared,
              "Requested shared memory exceeds hardware physical limit!");

  if (smem_bytes > 48 * 1024) {
    cudaFuncSetAttribute(flash::flash_forward_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_bytes);
  }

  flash::flash_forward_kernel<<<gridDim, blockDim, smem_bytes>>>(args);

  return O;
}