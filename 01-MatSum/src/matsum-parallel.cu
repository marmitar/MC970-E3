#include <cmath>
#include <omp.h>

#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <tuple>

/** Marker for conditional expressions that are unlikely to happen. */
#define unlikely(condition) (__builtin_expect(!!(condition), false))

/** Parse matrix dimensions from test case. */
static std::pair<unsigned, unsigned> read_input(const char *filename) {
  auto input = std::fstream(filename, std::fstream::in);
  input.exceptions(input.badbit | input.failbit | input.eofbit);

  unsigned rows, cols;
  input >> rows >> cols;

  constexpr unsigned max_size = std::numeric_limits<unsigned>::max();
  if unlikely (rows > max_size / cols) {
    throw std::length_error("matrix has more elements than UINT_MAX");
  }
  return std::pair(rows, cols);
}

static __global__ void matrix_sum(/* ... */) {
  // TODO: Implement this kernel!
  printf("Hello, World from the GPU!\n");
}

int main(const int argc, const char *const *const argv) {
  if unlikely (argc != 2) {
    throw std::invalid_argument("missing path to input file");
  }

  const auto [rows, cols] = read_input(argv[1]);

  // Allocate memory on the host
  unsigned *A = new unsigned[rows * cols];
  unsigned *B = new unsigned[rows * cols];
  unsigned *C = new unsigned[rows * cols];

  // Initialize memory
  for (unsigned i = 0; i < rows; i++) {
    for (unsigned j = 0; j < cols; j++) {
      A[i * cols + j] = B[i * cols + j] = i + j;
    }
  }

  // Copy data to device
  // ...

  // Compute matrix sum on device
  // Leave only the kernel and synchronize inside the timing region!
  const double start = omp_get_wtime();
  matrix_sum<<<1, 1>>>(/* ... */);
  cudaDeviceSynchronize();
  const double time = omp_get_wtime() - start;

  // Copy data back to host
  // ...

  long long unsigned sum = 0;

  // Keep this computation on the CPU
  for (unsigned i = 0; i < rows; i++) {
    for (unsigned j = 0; j < cols; j++) {
      sum += C[i * cols + j];
    }
  }

  std::cout << sum << std::endl;
  std::cerr << std::fixed << time << std::endl;

  delete[] A;
  delete[] B;
  delete[] C;

  return EXIT_SUCCESS;
}
