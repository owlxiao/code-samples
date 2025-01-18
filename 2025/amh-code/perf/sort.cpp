#include <algorithm>
#include <array>
#include <cstdlib>

constexpr std::size_t N = 1e6;
std::array<int, N> Array;

void setup() {
  for (int i = 0; i < N; ++i)
    Array[i] = rand();

  std::sort(Array.begin(), Array.end());
}

int query() {
  int checksum{0};
  for (int i = 0; i < N; ++i) {
    auto idx =
        std::lower_bound(Array.begin(), Array.end(), rand()) - Array.data();
    checksum += idx;
  }
  return checksum;
}

int main() {
  setup();
  query();

  return 0;
}