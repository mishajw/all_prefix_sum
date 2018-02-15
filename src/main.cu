#include <stdio.h>

#define CUDA_ERROR(statement, message) \
  do { \
    cudaError_t err = statement; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(err)); \
      exit(1); \
    } \
  } while (0);

#define GLOBAL_INDEX blockIdx.x * blockDim.x + threadIdx.x

int main() {
  printf("Hello, world!\n");
}

