#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <omp.h>

static __global__ void matrix_sum(/* ... */) {
  // TODO: Implement this kernel!
  printf("Hello, World from the GPU!\n");
}

int main(const int argc, const char *const *const argv) {
  if (argc < 2) {
    fprintf(stderr, "Error: missing path to input file\n");
    return EXIT_FAILURE;
  }

  FILE *input = fopen(argv[1], "r");
  if (input == NULL) {
    fprintf(stderr, "Error: could not open file\n");
    return EXIT_FAILURE;
  }

  // Input
  int rows = 0, cols = 0;
  assert(fscanf(input, "%d", &rows) == 1);
  assert(fscanf(input, "%d", &cols) == 1);
  fclose(input);

  // Allocate memory on the host
  int *A = new int[rows * cols];
  int *B = new int[rows * cols];
  int *C = new int[rows * cols];

  // Initialize memory
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      A[i * cols + j] = B[i * cols + j] = i + j;
    }
  }

  // Copy data to device
  // ...

  // Compute matrix sum on device
  // Leave only the kernel and synchronize inside the timing region!
  double t = omp_get_wtime();
  matrix_sum<<<1, 1>>>(/* ... */);
  cudaDeviceSynchronize();
  t = omp_get_wtime() - t;

  // Copy data back to host
  // ...

  long long int sum = 0;

  // Keep this computation on the CPU
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      sum += C[i * cols + j];
    }
  }

  fprintf(stdout, "%lli\n", sum);
  fprintf(stderr, "%lf\n", t);

  delete[] A;
  delete[] B;
  delete[] C;

  return EXIT_SUCCESS;
}
