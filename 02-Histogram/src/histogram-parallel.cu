#include <cuda.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
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
  /** Allocate 'count' elements of 'T' in the 'ctx'. */
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

    explicit array(const unsigned size, T *const data) : size_(size), data_(data) {}

  public:
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

  /** Smart wrapper for 'cudaEvent_t'. */
  class event final {
  private:
    cudaEvent_t handle = nullptr;

    explicit event() {
      error::check(cudaEventCreateWithFlags(&handle, cudaEventDefault | cudaEventBlockingSync));
    }

  public:
    /** Prevent implicit copies. */
    event(event &) = delete;
    event(const event &) = delete;
    /** Moves should still be okay. */
    constexpr event(event &&) noexcept = default;

    static event create() {
      return event();
    }

    ~event() {
      error::check(cudaEventDestroy(handle));
      handle = nullptr;
    }

    void query() const {
      error::check(cudaEventQuery(handle));
    }

    void record(cudaStream_t stream = 0) {
      error::check(cudaEventRecord(handle, stream));
    }

    void synchronize() {
      error::check(cudaEventSynchronize(handle));
    }

    using milliseconds = std::chrono::duration<float, std::milli>;

    static milliseconds elapsed_time(const event &start, const event &end) {
      float ms = 0.0f;
      error::check(cudaEventElapsedTime(&ms, start.handle, end.handle));
      return milliseconds(ms);
    }

    milliseconds elapsed_from(const event &start) const {
      return elapsed_time(start, *this);
    }

    milliseconds operator-(const event &start) const {
      return elapsed_from(start);
    }
  };
} // namespace cuda

/** Image utilities for PPM format. */
namespace PPM {
  /** A single pixel in a PPM image. */
  struct [[gnu::packed]] Pixel final {
  public:
    Pixel() = delete;

    /** Represents a single color in a pixel. */
    using Component = uint8_t;
    // each color component should be a single byte
    static_assert(sizeof(Component) == 1);

    Component red;
    Component green;
    Component blue;

    /** Number of color components in a pixel. */
    static constexpr unsigned components() noexcept {
      return (sizeof(Pixel::red) + sizeof(Pixel::green) + sizeof(Pixel::blue)) /
             sizeof(Pixel::Component);
    }

    /** Maximum value for a color component. */
    static constexpr unsigned component_color() noexcept {
      return std::numeric_limits<Component>::max();
    }
  };
  // each pixel must have its components tightly packed
  static_assert(sizeof(Pixel) == Pixel::components() * sizeof(Pixel::Component));

  /** Image implemented as an array of pixels. */
  struct Image final {
  private:
    /** Size for the allocated array, given the image dimensions. */
    static unsigned alloc_size(unsigned width, unsigned height) {
      const auto size = checked_mul(width, height);
      if unlikely (!size.has_value()) {
        throw std::bad_alloc();
      }
      return *size;
    }

    cuda::array<Pixel> content_;

    constexpr Pixel *data() noexcept {
      return content_.data();
    }

    /** Allocate a new image with 'width * height' pixels. */
    Image(const unsigned width, const unsigned height) : content_(alloc_size(width, height)) {}

  public:
    Image(Image &) = delete;
    Image(const Image &) = delete;
    constexpr Image(Image &&image) noexcept = default;

    /** The pixels that form the image. */
    constexpr const cuda::array<Pixel> &content() const noexcept {
      return content_;
    }

    /** Number of pixels in the image. */
    constexpr unsigned size() const noexcept {
      return content_.size();
    }

    /** Size in byte for all the image pixels. */
    constexpr std::streamsize bytes() const noexcept {
      return checked_mul<std::streamsize>(size(), sizeof(Pixel)).value();
    }

    /** Read a PPM image from file located at 'filename'. */
    static Image read(const char *filename) {
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
      if unlikely (component_color != Pixel::component_color()) {
        throw std::invalid_argument("Image does not have 8-bits components");
      }
      file.ignore(max_size, '\n');

      auto image = Image(width, height);
      file.read(reinterpret_cast<char *>(image.data()), image.bytes());

      return image;
    }
  };
}; // namespace PPM

/** Histogram utilities. */
namespace histogram {
  /** Calculates 'base ** exponent'. */
  static constexpr unsigned pow(unsigned base, unsigned exponent) {
    unsigned result = 1;
    while (exponent > 0) {
      if (exponent % 2 != 0) {
        result = checked_mul(result, base).value();
      }
      base = checked_mul(base, base).value();
      exponent /= 2;
    }
    return result;
  }

  /**
   * Represents how a color component (in range [0,255]) is represented in the histogram
   * (range [0, WIDTH-1]).
   */
  static constexpr unsigned WIDTH = 4;
  /** The size of a histogram array ('WIDTH ** number of components') */
  static constexpr unsigned SIZE = pow(WIDTH, PPM::Pixel::components());

  template <typename T, cuda::context ctx>
  /** Read a fixed-size histogram array from another context. */
  static std::array<T, SIZE> read_from(const cuda::array<T, ctx> &array) {
    auto output = std::array<T, SIZE>();
    if unlikely(array.size() != output.size()) {
      throw std::invalid_argument("array does not match histogram size");
    }

    cuda::memcpy<T, cuda::host, ctx>(output.data(), array.data(), output.size());
    return output;
  }

  /**
   * Maps a single component color to its part in the histogram index.
   *
   * This is is just a linear transformation from [0,256) to [0,WIDTH).
   */
  static constexpr __host__ __device__ unsigned
  map_component(const PPM::Pixel::Component value) noexcept {
    // multiple checks to guarantee no overflow happens in the transformation
    constexpr auto min = std::numeric_limits<PPM::Pixel::Component>::min();
    constexpr auto max = std::numeric_limits<PPM::Pixel::Component>::max();
    static_assert(std::numeric_limits<unsigned>::min() <= min);
    static_assert(std::numeric_limits<unsigned>::max() > max);
    static_assert(checked_mul(unsigned(max) - unsigned(min), WIDTH).has_value());
    // then apply the linear transformation
    return ((unsigned(value) - unsigned(min)) * WIDTH) / (unsigned(max) + 1 - unsigned(min));
  }

  /**
   * Calculate the index in the histogram for a given PPM pixel.
   *
   * This index can be though as 3 digit number in base WIDTH, such that each digit is a mapping
   * (see 'map_component') of one of its color components.
   */
  static constexpr __host__ __device__ unsigned index(const PPM::Pixel pixel) noexcept {
    constexpr auto max = map_component(std::numeric_limits<PPM::Pixel::Component>::max());
    static_assert((max * WIDTH + max) * WIDTH + max < SIZE);
    // translate the pixel to a number (r,g,b) in base WIDTH.
    const auto [r, g, b] = pixel;
    return (map_component(r) * WIDTH + map_component(g)) * WIDTH + map_component(b);
  }
}; // namespace histogram

/** Update absolute histogram for pixel at (blockIdx, threadIdx). */
static __launch_bounds__(cuda::BLOCK_SIZE) __global__
    void histogram_kernel(const PPM::Pixel *const image, unsigned *const hist_count) {

  const PPM::Pixel px = image[blockDim.x * blockIdx.x + threadIdx.x];
  const unsigned idx = histogram::index(px);
  atomicAdd(&hist_count[idx], 1);
}

using seconds = std::chrono::duration<double>;

/** Calculates the normalized histogram of 'image' and stores the result in 'hist'. */
static seconds calculate_histogram(const PPM::Image &image,
                                   std::array<double, histogram::SIZE> &hist) {
  // Create Events
  auto start = cuda::event::create();
  auto stop = cuda::event::create();
  // Copy data to the device
  auto dev_image = cuda::array<PPM::Pixel, cuda::device>::copy_from(image.content());
  auto count = cuda::array<unsigned, cuda::device>::zeroed(histogram::SIZE);

  start.record();
  // launch kernel and compute kernel runtime.
  cuda::last_error::clear();
  histogram_kernel<<<cuda::blocks(image.size()), cuda::BLOCK_SIZE>>>(dev_image.data(),
                                                                     count.data());
  cuda::last_error::check(); // check for kernel lauch errors
  cuda::device_synchronize();
  // calculate normalized histogram from absolute counters in CPU
  const auto host_count = histogram::read_from(count);
  const double total = static_cast<double>(image.size());
  for (unsigned i = 0; i < histogram::SIZE; i++) {
    hist[i] = static_cast<double>(host_count[i]) / total;
  }
  // stop timer after normalized histogram is calculated
  stop.record();
  stop.synchronize();
  return stop - start;
}

int main(const int argc, const char *const *const argv) {
  if unlikely (argc != 2) {
    throw std::invalid_argument("missing path to input file");
    return EXIT_FAILURE;
  }

  const auto image = PPM::Image::read(argv[1]);
  auto hist = std::array<double, histogram::SIZE>();

  // Compute histogram
  const auto elapsed = calculate_histogram(image, hist);

  for (const double h : hist) {
    std::cout << std::fixed << std::setprecision(3) << h << ' ';
  }
  std::cout << std::endl;

  std::cerr << std::fixed << elapsed.count() << std::endl;
  return EXIT_SUCCESS;
}
