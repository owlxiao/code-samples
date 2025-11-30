#include <torch/types.h>

decltype(auto) forward(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  return q;
}