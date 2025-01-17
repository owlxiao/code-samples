#include <benchmark/benchmark.h>

#include <algorithm>
#include <random>
#include <vector>

std::size_t baseline_despace(std::vector<char> &Buffer) {
  std::size_t pos{0};

  for (auto const &c : Buffer) {
    if (c == '\r' || c == '\n' || c == ' ')
      continue;
    Buffer[pos++] = c;
  }

  return pos;
}

std::size_t stl_despace(std::vector<char> &Buffer) {
  auto end = std::remove_if(Buffer.begin(), Buffer.end(), [&](char c) {
    return c == '\r' || c == '\n' || c == ' ';
  });

  Buffer.erase(end, Buffer.end());
  return Buffer.size();
}

class StringText {
public:
  explicit StringText(std::size_t Size) {
    data.resize(Size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::uniform_int_distribution<int> distChar(0, 255);

    std::generate_n(data.begin(), Size, [&]() {
      auto r = dist(gen);
      if (r < 0.01) {
        return ' ';
      } else if (r < 0.02) {
        return '\n';
      } else if (r < 0.03) {
        return '\r';
      } else {
        return static_cast<char>(distChar(gen));
      }
    });
  }

  std::vector<char> &operator()() { return data; }

private:
  std::vector<char> data;
};

static constexpr std::size_t N = 8 * 1024 * 1024;

#define BENCHMARK_DESPACE(Func)                                                \
  static void Func##_benchmark(benchmark::State &state) {                      \
    StringText Buffer(N);                                                      \
    volatile std::size_t count{0};                                             \
                                                                               \
    for (auto _ : state) {                                                     \
      auto string = Buffer();                                                  \
      count = Func(string);                                                    \
    }                                                                          \
  }                                                                            \
  BENCHMARK(Func##_benchmark);

BENCHMARK_DESPACE(baseline_despace);
BENCHMARK_DESPACE(stl_despace);

BENCHMARK_MAIN();
