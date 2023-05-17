#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <omp.h>

#include <algorithm>

static constexpr const char *COMMENT = "Histogram_GPU";
static constexpr unsigned RGB_COMPONENT_COLOR = 255;

static void check_cuda(cudaError_t error, const char *filename, const int line) {
  if (error != cudaSuccess) {
    fprintf(stderr, "Error: %s:%d: %s: %s\n", filename, line, cudaGetErrorName(error),
            cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

#define CUDACHECK(cmd) check_cuda(cmd, __FILE__, __LINE__)

struct PPMPixel final {
  uint8_t red;
  uint8_t green;
  uint8_t blue;
};

struct PPMImage final {
  unsigned x, y;
  PPMPixel *data;
};

static PPMImage *readPPM(const char *filename) {
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Unable to open file '%s'\n", filename);
    exit(1);
  }

  char buff[16];
  if (!fgets(buff, sizeof(buff), fp)) {
    perror(filename);
    exit(1);
  }

  if (buff[0] != 'P' || buff[1] != '6') {
    fprintf(stderr, "Invalid image format (must be 'P6')\n");
    exit(1);
  }

  PPMImage *img = new PPMImage;
  if (!img) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }

  int c = getc(fp);
  while (c == '#') {
    while (getc(fp) != '\n')
      ;
    c = getc(fp);
  }

  ungetc(c, fp);
  if (fscanf(fp, "%u %u", &img->x, &img->y) != 2) {
    fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
    exit(1);
  }

  unsigned rgb_comp_color;
  if (fscanf(fp, "%u", &rgb_comp_color) != 1) {
    fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
    exit(1);
  }

  if (rgb_comp_color != RGB_COMPONENT_COLOR) {
    fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
    exit(1);
  }

  while (fgetc(fp) != '\n')
    ;
  img->data = new PPMPixel[img->x * img->y];

  if (!img) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }

  if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
    fprintf(stderr, "Error loading image '%s'\n", filename);
    exit(1);
  }

  fclose(fp);
  return img;
}

static __launch_bounds__(1) __global__ void histogram_kernel() {
  printf("Warning: histogram_kernel not implemented!\n");
}

static double Histogram(PPMImage *image, float *h_h) {
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

  PPMImage *image = readPPM(argv[1]);
  float *h = new float[64];

  // Initialize histogram
  for (int i = 0; i < 64; i++)
    h[i] = 0.0;

  // Compute histogram
  double t = Histogram(image, h);

  for (int i = 0; i < 64; i++)
    printf("%0.3f ", h[i]);
  printf("\n");

  fprintf(stderr, "%lf\n", t);
  delete[] h;

  return EXIT_SUCCESS;
}
