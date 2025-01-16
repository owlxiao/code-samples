#include <immintrin.h>

#include <benchmark/benchmark.h>

#include <cstdint>
#include <random>

static std::size_t std_unique(uint32_t *out, std::size_t len) {
  return std::unique(out, out + len) - out;
}

static std::size_t unique(uint32_t *out, std::size_t len) {
  if (len == 0)
    return 0;

  std::size_t pos{1};
  auto old = out[0];

  for (size_t i = 1; i < len; i++) {
    auto newv = out[i];
    if (newv != old) {
      out[pos++] = newv;
    }
    old = newv;
  }

  return pos;
}

static std::size_t hope_unique(uint32_t *out, std::size_t len) {
  if (len == 0)
    return 0;

  std::size_t pos{1};
  auto old = out[0];

  for (size_t i = 1; i < len; i++) {
    auto newv = out[i];
    out[pos] = newv;
    pos += (newv != old);
    old = newv;
  }

  return pos;
}

std::vector<uint32_t> generate(std::size_t len) {
  std::vector<uint32_t> buffer(len);
  std::mt19937 gen;
  std::uniform_int_distribution<uint32_t> numberDist(
      0, std::numeric_limits<uint32_t>::max());
  std::uniform_int_distribution<uint32_t> repeatDist(
      0, std::numeric_limits<uint32_t>::max());

  while (buffer.size() < len) {
    auto number = numberDist(gen);
    auto repeat = repeatDist(gen);

    repeat = std::min(static_cast<std::size_t>(repeat), len - buffer.size());
    buffer.insert(buffer.end(), repeat, number);
  }

  return buffer;
}

static constexpr std::size_t N = 1024 * 1024 * 1024;
auto list = generate(N);

#define BENCHMARK_UNIQUE(Func)                                                 \
  static void Func##_bench(benchmark::State &state) {                          \
                                                                               \
    volatile auto count{0};                                                    \
                                                                               \
    for (auto _ : state) {                                                     \
      auto list_backup = list;                                                 \
      count = Func(list_backup.data(), list_backup.size());                    \
    }                                                                          \
  }                                                                            \
  BENCHMARK(Func##_bench)->Unit(benchmark::kMillisecond);

BENCHMARK_UNIQUE(std_unique);
BENCHMARK_UNIQUE(unique);
BENCHMARK_UNIQUE(hope_unique);

BENCHMARK_MAIN();