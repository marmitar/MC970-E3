#include <cuda.h>
#include <omp.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
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

    const std::pair<unsigned, unsigned> dim;
    cuda::array<Pixel> content_;

    constexpr Pixel *data() noexcept {
      return content_.data();
    }

    /** Allocate a new image with 'width * height' pixels. */
    Image(const unsigned width, const unsigned height)
        : dim(width, height), content_(alloc_size(width, height)) {}

  public:
    Image(Image &) = delete;
    Image(const Image &) = delete;
    constexpr Image(Image &&image) noexcept = default;

    /** The pixels that form the image. */
    constexpr const cuda::array<Pixel> &content() const noexcept {
      return content_;
    }

    constexpr const Pixel *data() const noexcept {
      return content().data();
    }

    /** Number of pixels in a row. */
    constexpr unsigned width() const noexcept {
      return dim.first;
    }

    /** Number of pixels in a column. */
    constexpr unsigned height() const noexcept {
      return dim.second;
    }

    /** Number of pixels in the image. */
    constexpr unsigned size() const noexcept {
      return content_.size();
    }

    /** Size in byte for all the image pixels. */
    constexpr std::streamsize bytes() const noexcept {
      return checked_mul<std::streamsize>(size(), sizeof(Pixel)).value();
    }

    /** Copy pixels from CUDA array. */
    template <cuda::context ctx> void copy_from(const cuda::array<Pixel, ctx> &pixels) {
      if unlikely (pixels.size() != size()) {
        throw std::invalid_argument("Array of pixels does not match the image shape");
      }
      cuda::memcpy<Pixel, cuda::host, ctx>(data(), pixels.data(), pixels.size());
    }

    /** An explicit copy constructor. */
    Image clone() const {
      auto cloned = Image(width(), height());
      cuda::memcpy<Pixel, cuda::host, cuda::host>(cloned.data(), data(), size());
      return cloned;
    }

    /** Read a PPM image from file located at 'filename'. */
    static Image read(const std::string &filename) {
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

    /** Write image to stream. */
    [[gnu::noinline]] friend std::ostream &operator<<(std::ostream &os, const Image &image) {
      constexpr auto COMMENT = "Smoothing GPU";

      auto &outs = os << "P6" << '\n'
                      << "# " << COMMENT << '\n'
                      << image.width() << ' ' << image.height() << '\n'
                      << Pixel::component_color() << '\n';
      return outs.write(reinterpret_cast<const char *>(image.data()), image.bytes());
    }
  };
} // namespace PPM

namespace saturating {
  template <typename Num, class = std::enable_if_t<std::is_integral_v<Num>>>
  /** Safely implements 'MIN(a + b, max)'. */
  constexpr Num add(Num a, Num b, Num max = std::numeric_limits<Num>::max()) noexcept {
    if unlikely (a > max - b) {
      return max;
    } else {
      return a + b;
    }
  }

  template <typename Num, class = std::enable_if_t<std::is_integral_v<Num>>>
  /** Safely implements 'MAX(a - b, min)'. */
  constexpr Num sub(Num a, Num b, Num min = std::numeric_limits<Num>::min()) noexcept {
    if unlikely (a < min + b) {
      return min;
    } else {
      return a - b;
    }
  }
} // namespace saturating

namespace mask {
  static constexpr unsigned width() noexcept {
#ifdef MASK_WIDTH
    return MASK_WIDTH;
#else
    return 15;
#endif
  }

  static constexpr unsigned radius() noexcept {
    static_assert(width() % 2 == 1);
    return (width() - 1) / 2;
  }

  static constexpr unsigned size() noexcept {
    static_assert(width() > 0);
    return width() * width();
  }
} // namespace mask

static __launch_bounds__(cuda::BLOCK_SIZE) __global__
    void smoothing_kernel(PPM::Pixel *restrict out, const PPM::Pixel *restrict img,
                          const unsigned width, const unsigned height) {

  const unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
  // pixel position (i, j)
  const unsigned i = idx % width;
  const unsigned j = idx / width;

  constexpr unsigned RADIUS = mask::radius();
  // calculate the total RGB in neighborhood
  unsigned red = 0, green = 0, blue = 0;
  for (unsigned x = saturating::sub(i, RADIUS); x <= saturating::add(i, RADIUS, width); x++) {
    for (auto y = saturating::sub(j, RADIUS); y <= saturating::add(j, RADIUS, height); y++) {
      red += img[y * width + x].red;
      green += img[y * width + x].green;
      blue += img[y * width + x].blue;
    }
  }

  // save average to output image
  out[idx].red = red / mask::size();
  out[idx].green = green / mask::size();
  out[idx].blue = blue / mask::size();
}

static double smoothing(PPM::Image &result, const PPM::Image &image) {
  // copy image to device memory
  const auto input = cuda::array<PPM::Pixel, cuda::device>::copy_from(image.content());
  auto output = cuda::array<PPM::Pixel, cuda::device>::zeroed(image.size());

  const double start = omp_get_wtime();
  // launch kernel and check for errors
  cuda::last_error::clear();
  smoothing_kernel<<<cuda::blocks(image.size()), cuda::BLOCK_SIZE>>>(output.data(), input.data(),
                                                                     image.width(), image.height());
  cuda::last_error::check();
  cuda::device_synchronize();
  // only measure the kernel code
  const double end = omp_get_wtime();

  // move result to host memory
  result.copy_from(output);
  return end - start;
}

/** Trims leading and trailing whitespaces. */
static std::string trim(const std::string &str) noexcept {
  constexpr auto whitespace = " \f\n\r\t\v";

  const auto start = str.find_first_not_of(whitespace);
  if (start == std::string::npos) {
    return ""; // no content
  }

  const auto end = str.find_last_not_of(whitespace);
  return str.substr(start, (end + 1 - start));
}

/** Open filename, reads the first line and returns the line trimmed of whitespaces. */
static std::string read_first_line(const char *const filename) {
  auto file = std::ifstream();
  file.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
  file.open(filename, std::fstream::in);

  auto line = std::string();
  std::getline(file, line);

  return trim(line);
}

int main(const int argc, const char *const *const argv) {
  if unlikely (argc != 2) {
    throw std::invalid_argument("missing path to input file");
    return EXIT_FAILURE;
  }

  // Read input filename
  const auto filename = read_first_line(argv[1]);

  // Read input file
  const auto image = PPM::Image::read(filename);
  auto output_image = image.clone();

  // Call Smoothing Kernel
  const double elapsed = smoothing(output_image, image);

  // Write result to stdout
  std::cout << output_image << std::endl;

  // Print time to stderr
  std::cerr << std::fixed << elapsed << std::endl;

  return EXIT_SUCCESS;
}
