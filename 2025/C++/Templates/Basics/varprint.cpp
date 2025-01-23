#include <iostream>
#include <string>

template <typename T> void print(T arg) { std::cout << arg << "\n"; }

template <typename T, typename... Types> void print(T firstArg, Types... args) {
  print(firstArg);
  print(args...);
}

int main() {
  std::string s("world");
  ::print(7.5, "hello", s);
}
