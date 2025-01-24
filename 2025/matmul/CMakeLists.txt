cmake_minimum_required(VERSION 3.30)
project(matmul)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-march=native)

find_package(benchmark)

add_executable(matmul main.cpp)
target_link_libraries(matmul benchmark::benchmark)
