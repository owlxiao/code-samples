cmake_minimum_required(VERSION 3.30)
project(FastDespacer)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-march=native)

find_package(benchmark REQUIRED)

add_executable(FastDespacer main.cpp)
target_link_libraries(FastDespacer benchmark::benchmark)