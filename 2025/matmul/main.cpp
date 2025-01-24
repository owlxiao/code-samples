#include "amh.h"

#include <benchmark/benchmark.h>

static constexpr int N{1024};

void amh_benchmark(benchmark::State &state) {
  std::vector<float> a(N * N, 0);
  std::vector<float> b(N * N, 0);
  std::vector<float> c(N * N, 0);

  for (auto _ : state) {
    amh::NaiveMatmul(a.data(), b.data(), c.data(), N);
  }
}

BENCHMARK(amh_benchmark)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();