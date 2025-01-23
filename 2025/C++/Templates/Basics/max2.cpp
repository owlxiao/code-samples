#include <type_traits>

template <typename T1, typename T2> auto max(T1 a, T2 b) {
  return b < a ? a : b;
}

// decltype
template <typename T1, typename T2>
auto max(T1 a, T2 b) -> decltype(b < a ? a : b) {
  return b < a ? a : b;
}

template <typename T1, typename T2>
auto max(T1 a, T2 b) -> decltype(true ? a : b) {
  return b < a ? a : b;
}

template <typename T1, typename T2> std::common_type<T1, T2> max(T1 a, T2 b) {
  return b < a ? a : b;
}
