#include <cmath>
#include <cuda.h>
#include <omp.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>

/** Marker for conditional expressions that are unlikely to happen. */
#define unlikely(condition) (__builtin_expect(!!(condition), false))
/** C-like restrict keyword. */
#define restrict __restrict__

template <typename Num>
/** Does 'a * b' if possible, or returns 'nullopt' if an overflow or underflow would occur. */
static constexpr inline std::optional<Num> checked_mul(const Num a, const Num b) noexcept {
  if (b == 0) {
    return a * b;
  }

  constexpr Num MINUMUM = std::numeric_limits<Num>::min();
  constexpr Num MAXIMUM = std::numeric_limits<Num>::max();
  const Num lo = std::min(MINUMUM / b, MAXIMUM / b);
  const Num hi = std::max(MINUMUM / b, MAXIMUM / b);

  if unlikely (a < lo || a > hi) {
    return std::nullopt;
  }
  return a * b;
}

/** CUDA helper functions. */
namespace cuda {
  /** Number of threads per block. */
  static constexpr unsigned BLOCK_SIZE = THREADS_PER_BLOCK;
  static_assert(BLOCK_SIZE > 0);
  static_assert(BLOCK_SIZE % 32 == 0);

  /** Reports a cudaError_t. */
  class error final : public std::runtime_error {
  private:
    explicit error(cudaError_t errnum) : std::runtime_error(cudaGetErrorString(errnum)) {}

    /** Always fails with 'cuda::error(errnum)'.
     *
     * Reduces code bloat from inlined 'cuda::error::check(result)' calls.
     */
    [[gnu::cold, noreturn]] static void fail(const cudaError_t errnum) {
      throw error(errnum);
    }

  public:
    /** Throws an error if 'result' is not 'cudaSuccess'.  */
    [[gnu::hot]] static inline void check(const cudaError_t result) {
      if unlikely (result != cudaSuccess) {
        fail(result);
      }
    }
  };

  namespace last_error {
    /** Removes that last error value and set it to 'cudaSuccess'. */
    static void clear() noexcept {
      cudaGetLastError();
    }

    /** Throws an error if 'cudaGetLastError' returns something other than 'cudaSuccess'. */
    static void check() {
      error::check(cudaGetLastError());
    }
  }; // namespace last_error

  /** Checked 'cudaDeviceSynchronize'. */
  static void device_synchronize() {
    error::check(cudaDeviceSynchronize());
  }

  /** Calculate the number of blocks for 'count' elements. */
  static constexpr inline unsigned blocks(const unsigned count) noexcept {
    if unlikely (count == 0) {
      return 0;
    } else {
      // ceil(count / BLOCK_SIZE)
      return (count - 1) / BLOCK_SIZE + 1;
    }
  }

  /** Rounds 'count' to the nearest multiple of 'BLOCK_SIZE'. */
  static constexpr inline unsigned nearest_block_multiple(const unsigned count) {
    const auto multiple = checked_mul(BLOCK_SIZE, blocks(count));
    if unlikely (!multiple.has_value()) {
      throw std::length_error("array is too big");
    }
    return *multiple;
  }

  template <typename T>
  /** Size in bytes for an array of 'count' elements of 'T'. */
  static constexpr inline size_t byte_size(const unsigned count) noexcept {
    constexpr size_t max_count = std::numeric_limits<unsigned>::max();
    // this guarantees that 'count * sizeof(T)' cannot overflow
    static_assert(checked_mul<size_t>(max_count, sizeof(T)).has_value());

    return static_cast<size_t>(count) * sizeof(T);
  }

  template <typename T>
  /** Allocate 'count' elements of 'T' in the GPU. */
  [[gnu::malloc]] static T *malloc(const unsigned count) {
    T *ptr = nullptr;
    // the number of elements is rounded so that the size is evenly divisible by BLOCK_SIZE
    const unsigned closest_count = nearest_block_multiple(count);
    error::check(cudaMalloc(&ptr, byte_size<T>(closest_count)));
    // set the unused elements to zero
    if (count < closest_count) {
      error::check(cudaMemset(&ptr[count], 0, byte_size<T>(closest_count - count)));
    }
    return ptr;
  }

  /** Checked 'cudaFree'. */
  static void free(void *ptr) {
    error::check(cudaFree(ptr));
  }

  template <typename T>
  /** Copy 'count' elements of 'T' from the Host to the GPU. */
  static void memcpy_host_to_device(T *restrict dst, const T *restrict src, const unsigned count) {
    error::check(cudaMemcpy(dst, src, byte_size<T>(count), cudaMemcpyHostToDevice));
  }

  template <typename T>
  /** Copy 'count' elements of 'T' from the GPU to the Host. */
  static void memcpy_device_to_host(T *restrict dst, const T *restrict src, const unsigned count) {
    error::check(cudaMemcpy(dst, src, byte_size<T>(count), cudaMemcpyDeviceToHost));
  }
}; // namespace cuda

/** Parse matrix dimensions from test case. */
static std::pair<unsigned, unsigned> read_input(const char *filename) {
  auto input = std::fstream(filename, std::fstream::in);
  input.exceptions(input.badbit | input.failbit | input.eofbit);

  unsigned rows, cols;
  input >> rows >> cols;

  if unlikely (!checked_mul(rows, cols).has_value()) {
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
