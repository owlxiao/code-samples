build: sort.cpp
	clang++ -O3 -march=native $< -o run

perf: build
	perf stat ./run

.PHONY: build perf