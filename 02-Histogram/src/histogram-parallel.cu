#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <omp.h>

#include <algorithm>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>

/** Marker for conditional expressions that are unlikely to happen. */
#define unlikely(condition) (__builtin_expect(!!(condition), false))
/** C-like restrict keyword. */
#define restrict __restrict__

template <typename Num, class = std::enable_if_t<std::is_integral_v<Num>>>
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
  enum context {
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

    T *ptr = malloc_exact<T, ctx>(count);
    // set the unused elements to zero
    if (count < closest_count) {
      memset_zero<T, ctx>(&ptr[count], closest_count - count);
    }
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

  template <typename T, context ctx>
  /** A CUDA Execution-Space aware pointer that behaves like an array of 'T'. */
  class array_like {};

  template <typename T>
  /** A pointer that behaves like an array of 'T' in host execution space. */
  class array_like<T, context::host> {
  public:
    virtual ~array_like() {}

    /** Pointer to the underlying array, accessible in any context. */
    virtual __host__ __device__ T *data() const noexcept = 0;
    /** Array size, accessible in any context. */
    virtual __host__ __device__ unsigned size() const noexcept = 0;

    inline __host__ T *begin() noexcept {
      return data();
    }
    inline __host__ const T *begin() const noexcept {
      return data();
    }

    inline __host__ T *end() noexcept {
      return data() + size();
    }
    inline __host__ const T *end() const noexcept {
      return data() + size();
    }

    inline __host__ T &operator[](const unsigned index) noexcept {
      return data()[index];
    }
    inline __host__ const T &operator[](const unsigned index) const noexcept {
      return data()[index];
    }

    template <context other>
    /** Copies data from another array, possibly in another context. */
    __host__ void copy_from(const array_like<T, other> &source) {
      if unlikely (size() < source.size()) {
        throw std::range_error("source array is not big enough");
      }
      memcpy<T, context::host, other>(data(), source.data(), size());
    }
  };

  template <typename T>
  /** A pointer that behaves like an array of 'T' in device execution space. */
  class array_like<T, context::device> {
  public:
    virtual ~array_like() {}

    /** Pointer to the underlying array, accessible in any context. */
    virtual __host__ __device__ T *data() const noexcept = 0;
    /** Array size, accessible in any context. */
    virtual __host__ __device__ unsigned size() const noexcept = 0;

    inline __device__ T *begin() noexcept {
      return data();
    }
    inline __device__ const T *begin() const noexcept {
      return data();
    }

    inline __device__ T *end() noexcept {
      return data() + size();
    }
    inline __device__ const T *end() const noexcept {
      return data() + size();
    }

    inline __device__ T &operator[](const unsigned index) noexcept {
      return data()[index];
    }
    inline __device__ const T &operator[](const unsigned index) const noexcept {
      return data()[index];
    }

    template <context other>
    /** Copies data from another array, possibly in another context. */
    __host__ void copy_from(const array_like<T, other> &source) {
      if unlikely (size() < source.size()) {
        throw std::range_error("source array is not big enough");
      }
      memcpy<T, context::device, other>(data(), source.data(), size());
    }
  };

  template <typename T, context ctx>
  /** A CUDA Execution-Space aware smart pointer that behaves like an array of 'T'. */
  class array final : public array_like<T, ctx> {
  private:
    const unsigned size_;
    T *const data_;

  public:
    /** Allocates an array of 'size' elements. */
    explicit array(const unsigned size) : size_(size), data_(malloc<T, ctx>(size)) {}

    /** Prevent implicit copies. */
    array(array<T, ctx> &) = delete;
    array(const array<T, ctx> &) = delete;
    /** Moves should still be okay. */
    constexpr array(array<T, ctx> &&) noexcept = default;

    template <context other, class = std::enable_if_t<other != ctx>>
    /** Allows copies from different context. */
    array(const array<T, other> &source) : array(source.size()) {
      array_like<T, ctx>::copy_from(source);
    }

    ~array() {
      free<T, ctx>(data());
    }

    inline __device__ __host__ T *data() const noexcept {
      return data_;
    }

    inline __device__ __host__ unsigned size() const noexcept {
      return size_;
    }
  };
} // namespace cuda

#define CUDACHECK(cmd) cuda::error::check(cmd)

static constexpr const char *COMMENT = "Histogram_GPU";
static constexpr unsigned RGB_COMPONENT_COLOR = 255;

namespace PPM {
  struct [[gnu::packed]] Pixel final {
  public:
    Pixel() = delete;

    uint8_t red;
    uint8_t green;
    uint8_t blue;
  };
  static_assert(sizeof(Pixel) == 3 * sizeof(uint8_t));

  template <cuda::context ctx>
  /** Image implemented as a Execution-Space-aware array of pixels. */
  struct Image final : public cuda::array_like<Pixel, ctx> {
  private:
    static unsigned total_size(unsigned width, unsigned height) {
      const auto size = checked_mul(width, height);
      if unlikely (!size.has_value()) {
        throw std::bad_alloc();
      }
      return *size;
    }

  private:
    const unsigned width_, height_;
    cuda::array<Pixel, ctx> content;

  public:
    Image(const unsigned width, const unsigned height)
        : width_(width), height_(height), content(total_size(width, height)) {}

    constexpr Image(Image<ctx> &&image) noexcept = default;

    template <cuda::context other, class = std::enable_if_t<other != ctx>>
    Image(const Image<other> &image)
        : width_(image.width()), height_(image.height()), content(image.size()) {
      this->copy_from(image);
    }

    inline __device__ __host__ Pixel *data() const noexcept {
      return content.data();
    }

    inline __device__ __host__ unsigned size() const noexcept {
      return content.size();
    }

    constexpr __device__ __host__ unsigned width() const noexcept {
      return width_;
    }

    constexpr __device__ __host__ unsigned height() const noexcept {
      return height_;
    }

    static Image<ctx> read(const char *filename) {
      auto file = std::ifstream();
      file.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
      file.open(filename, std::fstream::in);

      auto line = std::string();
      std::getline(file, line);
      if unlikely (line != "P6") {
        throw std::invalid_argument("Invalid image format (must be 'P6')");
      }

      constexpr auto max_size = std::numeric_limits<std::streamsize>::max();
      while (file.get() == '#') {
        file.ignore(max_size, '\n');
      }
      file.unget();

      unsigned width, height;
      unsigned component_color;
      file >> width >> height >> component_color;
      if unlikely (component_color != RGB_COMPONENT_COLOR) {
        throw std::invalid_argument("Image does not have 8-bits components");
      }
      file.ignore(max_size, '\n');

      auto image = Image<cuda::host>(width, height);
      const auto bytes = checked_mul<std::streamsize>(image.size(), sizeof(Pixel)).value();
      file.read(reinterpret_cast<char *>(image.data()), bytes);

      return image;
    }
  };
}; // namespace PPM

static __launch_bounds__(1) __global__ void histogram_kernel() {
  printf("Warning: histogram_kernel not implemented!\n");
}

static double Histogram(PPM::Image<cuda::host> &image, cuda::array<float, cuda::host> &h) {
  // Create Events
  cudaEvent_t start, stop;
  CUDACHECK(cudaEventCreate(&start));
  CUDACHECK(cudaEventCreate(&stop));

  // Launch kernel and compute kernel runtime.
  // Warning: make sure only the kernel is being profiled, memcpies should be
  // out of this region.
  CUDACHECK(cudaEventRecord(start));
  histogram_kernel<<<1, 1>>>();
  CUDACHECK(cudaEventRecord(stop));
  CUDACHECK(cudaEventSynchronize(stop));
  float ms;
  CUDACHECK(cudaEventElapsedTime(&ms, start, stop));

  // Destroy events
  CUDACHECK(cudaEventDestroy(start));
  CUDACHECK(cudaEventDestroy(stop));

  return ((double)ms) / 1000.0;
}

int main(const int argc, const char *const *const argv) {
  if (argc != 2) {
    fprintf(stderr, "Error: missing path to input file\n");
    return EXIT_FAILURE;
  }

  auto image = PPM::Image<cuda::host>::read(argv[1]);
  auto h = cuda::array<float, cuda::host>(64);

  // Initialize histogram
  for (int i = 0; i < 64; i++)
    h[i] = 0.0;

  // Compute histogram
  double t = Histogram(image, h);

  for (int i = 0; i < 64; i++)
    printf("%0.3f ", h[i]);
  printf("\n");

  fprintf(stderr, "%lf\n", t);

  return EXIT_SUCCESS;
}
