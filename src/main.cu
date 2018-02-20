#include <stdio.h>
#include <inttypes.h>

#define CUDA_ERROR(err, message) \
  do { \
    if (err != cudaSuccess) { \
      fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(err)); \
      exit(1); \
    } \
  } while (0);

#define GLOBAL_INDEX blockIdx.x * blockDim.x + threadIdx.x

#define BLOCK_SIZE 1024

// The size of the array to test on
static const size_t ARRAY_SIZE = 1000;

typedef int32_t num_t;

// Performs exclusive Blelloch scan on a block level
__global__
void blelloch_block_scan(const num_t *g_input, num_t *g_output, size_t length) {
  __shared__ num_t s_temp[BLOCK_SIZE * 2];
  size_t global_index = GLOBAL_INDEX;
  size_t index = threadIdx.x;
  size_t offset = 1;

  // Copy global memory into shared
  if (global_index * 2 < length) {
    s_temp[index * 2] = g_input[global_index * 2];
  }
  if (global_index * 2 + 1 < length) {
    s_temp[index * 2 + 1] = g_input[global_index * 2 + 1];
  }

  // Up sweep
  for (size_t d = BLOCK_SIZE; d > 0; d /= 2) {
    __syncthreads();

    if (global_index < d) {
      size_t a = offset * (2 * index + 1) - 1;
      size_t b = offset * (2 * index + 2) - 1;
      s_temp[b] += s_temp[a];
    }

    offset *= 2;
  }

  // Reset last element
  if (index == 0) {
    s_temp[BLOCK_SIZE * 2 - 1] = 0;
  }

  // Down sweep
  for (size_t d = 1; d < BLOCK_SIZE * 2; d *= 2) {
    offset /= 2;
    __syncthreads();
    if (global_index < d) {
      size_t a = offset * (2 * index + 1) - 1;
      size_t b = offset * (2 * index + 2) - 1;
      num_t t = s_temp[a];
      s_temp[a] = s_temp[b];
      s_temp[b] += t;
    }
  }

  __syncthreads();

  // Copy results into global memory
  if (global_index * 2 < length) {
    g_output[global_index * 2] = s_temp[index * 2];
  }
  if (global_index * 2 + 1 < length) {
    g_output[global_index * 2 + 1] = s_temp[index * 2 + 1];
  }
}

// Performs all prefix sum on `input` and stores the result in `output` in
// parallel on a GPU
// Assumes both `input` and `output` are allocated with size `length`
void scan(const num_t *input, num_t *output, size_t length) {
  cudaError_t err;
  size_t array_size = sizeof(num_t) * length;

  // Set up input on device
  num_t *g_input = NULL;
  err = cudaMalloc((void **)&g_input, array_size);
  CUDA_ERROR(err, "Couldn't allocate memory for input on device");
  err = cudaMemcpy(g_input, input, array_size, cudaMemcpyHostToDevice);
  CUDA_ERROR(err, "Couldn't copy input to device");

  // Setup output on device
  num_t *g_output = NULL;
  err = cudaMalloc((void **)&g_output, array_size);
  CUDA_ERROR(err, "Couldn't allocate memory for output on device");

  if (length <= BLOCK_SIZE) {
    blelloch_block_scan<<<1, BLOCK_SIZE>>>(g_input, g_output, length);
    CUDA_ERROR(cudaGetLastError(), "Couldn't perform block scan");
  } else {
    // TODO: Implement
  }

  // TODO: Check if necessary
  cudaDeviceSynchronize();

  // Copy results to host
  err = cudaMemcpy(output, g_output, array_size, cudaMemcpyDeviceToHost);
  CUDA_ERROR(err, "Couldn't copy output to host");
}

// Performs all prefix sum on `input` and stores the result in `output`
// sequentially on the CPU
// Assumes both `input` and `output` are allocated with size `length`
void sequential_scan(const num_t *input, num_t *output, size_t length) {
  for (size_t i = 1; i < length; i++) {
    output[i] = output[i - 1] + input[i - 1];
  }
}

// Fills the array `array` with `length` random values from 0-9 inclusive
void fill_random_array(num_t *array, size_t length) {
  for (size_t i = 0; i < length; i++) {
    array[i] = rand() % 10;
  }
}

// Print how two arrays `a` and `b` differ, up to some length `length`
// Returns true if the arrays are equal, and false otherwise
bool print_array_equality(num_t *a, num_t *b, size_t length) {
  bool are_equal = true;

  for (size_t i = 0; i < length; i++) {
    if (a[i] != b[i]) {
      are_equal = false;
      printf(
          "Arrays differ at index %ld, with values %d and %d\n", i, a[i], b[i]);
    }
  }

  return are_equal;
}

int main() {
  // Set up the input to scan
  num_t *input = (num_t *)malloc(sizeof(num_t) * ARRAY_SIZE);
  fill_random_array(input, ARRAY_SIZE);

  // Set up the output for sequential function, to be used as ground truth for
  // comparison
  num_t *truth_output = (num_t *)malloc(sizeof(num_t) * ARRAY_SIZE);

  // Set up the output for the parallel function
  num_t *output = (num_t *)malloc(sizeof(num_t) * ARRAY_SIZE);

  // Run the sequential scan
  // TODO: Time this operation
  sequential_scan(input, truth_output, ARRAY_SIZE);

  // Run the parallel scan
  // TODO: Time this operation
  scan(input, output, ARRAY_SIZE);

  // Compare solutions
  bool are_equal = print_array_equality(truth_output, output, ARRAY_SIZE);

  if (are_equal) {
    printf("Success!\n");
  }

  return are_equal ? 0 : 1;
}

