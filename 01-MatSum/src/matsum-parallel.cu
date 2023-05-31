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

  /** Memory context, associated with a Function Execution Space Specifier. */
  enum context : bool {
    /** __host__ space (usually the CPU and main memory context). */
    host,
    /** __device__ space (one or more GPUs contexts). */
    device
  };

  template <typename T, context ctx>
  /** Allocates memory in 'ctx' for exactly 'count' elements of 'T'. */
  static T *malloc_exact(const unsigned count) {
    T *ptr = nullptr;

    if (ctx == context::device) {
      error::check(cudaMalloc(&ptr, byte_size<T>(count)));
    } else {
      error::check(cudaMallocHost(&ptr, byte_size<T>(count)));
    }
    return ptr;
  }

  template <typename T, context ctx>
  /** Zeroes memory for 'count' elements of 'T' in 'ptr'. */
  static void memset_zero(T *ptr, const unsigned count) noexcept(ctx == context::host) {
    if (ctx == context::device) {
      error::check(cudaMemset(ptr, 0, byte_size<T>(count)));
    } else {
      memset(ptr, 0, byte_size<T>(count));
    }
  }

  template <typename T, context ctx>
  /** Allocate 'count' elements of 'T' in the 'ctx'. */
  [[gnu::malloc]] static T *malloc(const unsigned count) {
    // the number of elements is rounded so that the size is evenly divisible by BLOCK_SIZE
    const unsigned closest_count = nearest_block_multiple(count);

    T *ptr = malloc_exact<T, ctx>(closest_count);
    // set the unused elements to zero
    if (count < closest_count) {
      memset_zero<T, ctx>(&ptr[count], closest_count - count);
    }
    return ptr;
  }

  template <typename T, context ctx>
  /** Allocate 'count' elements of 'T' in the 'ctx' with memory zeroed. */
  [[gnu::malloc]] static T *calloc(const unsigned count) {
    // the number of elements is rounded so that the size is evenly divisible by BLOCK_SIZE
    const unsigned closest_count = nearest_block_multiple(count);

    T *ptr = malloc_exact<T, ctx>(closest_count);
    memset_zero<T, ctx>(ptr, closest_count);
    return ptr;
  }

  template <typename T, context ctx>
  /** Release memory allocated in 'cuda::malloc'. */
  static void free(T *ptr) {
    if (ctx == context::device) {
      error::check(cudaFree(ptr));
    } else {
      error::check(cudaFreeHost(ptr));
    }
  }

  template <context src_ctx, context dst_ctx>
  /** Dictates the kind of memcpy used in 'cudaMemcpy', given source and destination contexts. */
  static constexpr cudaMemcpyKind memcpy_kind() noexcept {
    switch (src_ctx) {
    case context::host:
      switch (dst_ctx) {
      case context::host:
        return cudaMemcpyHostToHost;
      case context::device:
        return cudaMemcpyHostToDevice;
      default:
        return cudaMemcpyDefault;
      }
    case context::device:
      switch (dst_ctx) {
      case context::host:
        return cudaMemcpyDeviceToHost;
      case context::device:
        return cudaMemcpyDeviceToDevice;
      default:
        return cudaMemcpyDefault;
      }
    default:
      return cudaMemcpyDefault;
    }
  }

  template <typename T, context dst_ctx, context src_ctx>
  /** Copies 'count' elements of 'T' from 'src' to 'dst', given their contexts. */
  static void memcpy(T *dst, const T *src, const unsigned count) {
    error::check(cudaMemcpy(dst, src, byte_size<T>(count), memcpy_kind<src_ctx, dst_ctx>()));
  }

  template <typename T, context ctx = cuda::context::host>
  /** A CUDA Execution-Space aware smart pointer that behaves like an array of 'T'. */
  class array final {
  private:
    const unsigned size_;
    T *const data_;

    /** Data pointer must be valid in 'ctx'. */
    explicit array(const unsigned size, T *const data) : size_(size), data_(data) {}

  public:
    using element_type = T;

    /** Allocates an array of 'size' elements. */
    explicit array(const unsigned count) : array(count, malloc<T, ctx>(count)) {}

    /** Prevent implicit copies. */
    array(array<T, ctx> &) = delete;
    array(const array<T, ctx> &) = delete;
    /** Moves should still be okay. */
    constexpr array(array<T, ctx> &&) noexcept = default;

    /** Copies data from another array, possibly in another context. */
    static array<T, ctx> zeroed(const unsigned count) {
      return array<T, ctx>(count, calloc<T, ctx>(count));
    }

    template <context other>
    /** Copies data from another array, possibly in another context. */
    static array<T, ctx> copy_from(const array<T, other> &source) {
      auto dst = array<T, ctx>(source.size());
      memcpy<T, ctx, other>(dst.data(), source.data(), dst.size());
      return dst;
    }

    ~array() {
      free<T, ctx>(data());
    }

    /** Pointer to the underlying array. */
    constexpr T *data() noexcept {
      return data_;
    }
    constexpr const T *data() const noexcept {
      return data_;
    }

    /** Array size. */
    constexpr unsigned size() const noexcept {
      return size_;
    }

    inline T *begin() noexcept {
      static_assert(ctx == context::host);
      return data();
    }
    inline const T *begin() const noexcept {
      static_assert(ctx == context::host);
      return data();
    }

    inline T *end() noexcept {
      static_assert(ctx == context::host);
      return data() + size();
    }
    inline const T *end() const noexcept {
      static_assert(ctx == context::host);
      return data() + size();
    }

    inline T &operator[](const unsigned index) noexcept {
      static_assert(ctx == context::host);
      return data()[index];
    }
    inline const T &operator[](const unsigned index) const noexcept {
      static_assert(ctx == context::host);
      return data()[index];
    }
  };
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

static __launch_bounds__(cuda::BLOCK_SIZE) __global__
    void matrix_sum(const unsigned *restrict A, const unsigned *restrict B, unsigned *restrict C) {

  const unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
  // no need to check the range, since 'cuda::malloc' rounds to the nearest multiple of BLOCK_SIZE
  C[idx] = A[idx] + B[idx];
}

int main(const int argc, const char *const *const argv) {
  if unlikely (argc != 2) {
    throw std::invalid_argument("missing path to input file");
  }

  const auto [rows, cols] = read_input(argv[1]);

  // Allocate memory on the host
  auto A = cuda::array<unsigned>(rows * cols);
  auto B = cuda::array<unsigned>(rows * cols);

  // Initialize memory
  for (unsigned i = 0; i < rows; i++) {
    for (unsigned j = 0; j < cols; j++) {
      A[i * cols + j] = B[i * cols + j] = i + j;
    }
  }

  // Copy data to device
  const auto cA = cuda::array<unsigned, cuda::device>::copy_from(A);
  const auto cB = cuda::array<unsigned, cuda::device>::copy_from(B);
  auto cC = cuda::array<unsigned, cuda::device>(rows * cols);

  // Compute matrix sum on device
  // Leave only the kernel and synchronize inside the timing region!
  const double start = omp_get_wtime();
  // launch kernel checking for errors
  cuda::last_error::clear();
  const unsigned total = rows * cols;
  matrix_sum<<<cuda::blocks(total), cuda::BLOCK_SIZE>>>(cA.data(), cB.data(), cC.data());
  cuda::last_error::check();

  cuda::device_synchronize();
  const double time = omp_get_wtime() - start;

  // Copy data back to host
  auto C = cuda::array<unsigned>::copy_from(cC);

  long long unsigned sum = 0;

  // Keep this computation on the CPU
  for (unsigned i = 0; i < rows; i++) {
    for (unsigned j = 0; j < cols; j++) {
      sum += C[i * cols + j];
    }
  }

  std::cout << sum << std::endl;
  std::cerr << std::fixed << time << std::endl;

  return EXIT_SUCCESS;
}
