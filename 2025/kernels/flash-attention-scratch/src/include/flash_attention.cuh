#pragma once

namespace flash {

struct ForwardKernelArgs {
  using index_t = int64_t;

  void *__restrict__ Q;
  void *__restrict__ K;
  void *__restrict__ V;
  void *__restrict__ O;

  const index_t batch_stride;
  const index_t seq_stride;
  const index_t head_stride;

  const index_t seq_len;
  const index_t n_heads;

  const int q_blocks;
  const int k_blocks;
};

struct FlashForwardKernelConfig {
  const int HeadDim;
  const int Br;
  const int Bc;
  const int Warps;

  int smem_bytes(int ElemSize = 2) const {
    // return (Br * Bc * 2) * HeadDim * ElemSize;
    return 2 * (Br * HeadDim + Bc * HeadDim + Bc * HeadDim) * ElemSize;
  }
};

} // namespace flash