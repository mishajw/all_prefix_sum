#include <stdio.h>
#include <inttypes.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#define CUDA_ERROR(err, message) \
  do { \
    cudaError_t err2 = err; \
    if (err2 != cudaSuccess) { \
      fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(err2)); \
      exit(1); \
    } \
  } while (0);

#define GLOBAL_INDEX blockIdx.x * blockDim.x + threadIdx.x

#define BLOCK_SIZE ((size_t)128)

// `#define`s for addessing shared memory bank conflicts
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define OFFSET_ARRAY_INDEX(index) \
  ((index) >> NUM_BANKS + (index) >> (2 * LOG_NUM_BANKS))

// The size of the array to test on
static const size_t ARRAY_SIZE = 10000000;

typedef int32_t num_t;

// Performs exclusive Blelloch scan on a block level
// Also stores the sum of a total block in the `g_block_ends`
__global__
void blelloch_block_scan(
    const num_t *g_input, num_t *g_output, num_t *g_block_ends, size_t length) {
  __shared__ num_t s_temp[BLOCK_SIZE * 2];
  size_t global_index = GLOBAL_INDEX;
  size_t index = threadIdx.x;
  size_t offset = 1;

  // Copy global memory into shared
  if (global_index * 2 < length) {
    size_t i = global_index * 2;
    s_temp[index * 2] = g_input[i + OFFSET_ARRAY_INDEX(i)];
  }
  if (global_index * 2 + 1 < length) {
    size_t i = global_index * 2 + 1;
    s_temp[index * 2 + 1] = g_input[i + OFFSET_ARRAY_INDEX(i)];
  }

  // Up sweep
  for (size_t d = BLOCK_SIZE; d > 0; d /= 2) {
    __syncthreads();

    if (index < d) {
      size_t a = offset * (2 * index + 1) - 1;
      size_t b = offset * (2 * index + 2) - 1;
      a += OFFSET_ARRAY_INDEX(a);
      b += OFFSET_ARRAY_INDEX(b);
      s_temp[b] += s_temp[a];
    }

    offset *= 2;
  }

  // Reset last element
  if (index == 0) {
    size_t i = BLOCK_SIZE * 2 - 1;
    i += OFFSET_ARRAY_INDEX(i);
    // Save the block end
    if (g_block_ends != NULL) {
      g_block_ends[global_index / BLOCK_SIZE] = s_temp[i];
    }

    s_temp[i] = 0;
  }

  // Down sweep
  for (size_t d = 1; d < BLOCK_SIZE * 2; d *= 2) {
    offset /= 2;
    __syncthreads();
    if (index < d) {
      size_t a = offset * (2 * index + 1) - 1;
      size_t b = offset * (2 * index + 2) - 1;
      a += OFFSET_ARRAY_INDEX(a);
      b += OFFSET_ARRAY_INDEX(b);
      num_t t = s_temp[a];
      s_temp[a] = s_temp[b];
      s_temp[b] += t;
    }
  }

  __syncthreads();

  // Copy results into global memory
  if (global_index * 2 < length) {
    size_t i = index * 2;
    g_output[global_index * 2] = s_temp[i + OFFSET_ARRAY_INDEX(i)];
  }
  if (global_index * 2 + 1 < length) {
    size_t i = index * 2 + 1;
    g_output[global_index * 2 + 1] = s_temp[i + OFFSET_ARRAY_INDEX(i)];
  }
}

// Add the block scan ends back onto the original array
__global__
void add_block_scan_ends(
    num_t *g_input, const num_t *g_block_ends, size_t length) {

  size_t global_index = GLOBAL_INDEX;
  if (global_index < length) {
    g_input[global_index] += g_block_ends[global_index / (BLOCK_SIZE * 2)];
  }
}

// Perform level 1 scan on individual blocks of size `BLOCK_SIZE * 2`
void level1_scan(
    const num_t *g_input,
    num_t *g_output,
    const size_t length,
    num_t *g_block_ends,
    const size_t num_blocks) {
  blelloch_block_scan<<<num_blocks, BLOCK_SIZE>>>(
      g_input, g_output, g_block_ends, length);
  CUDA_ERROR(cudaGetLastError(), "Couldn't perform block scan");
}

// Perform level 2 scan on groups of blocks where blocks are of size
// `BLOCK_SIZE * 2` and the groups are of size `BLOCK_SIZE * 2`
void level2_scan(
    const num_t *g_input,
    num_t *g_output,
    const size_t length,
    num_t *g_block_ends,
    const size_t num_blocks) {

  // Perform level 1 scan first
  level1_scan(g_input, g_output, length, g_block_ends, num_blocks);

  // Perform prefix sum of block scan ends
  size_t ends_num_blocks = 1 + (length - 1) / (BLOCK_SIZE * BLOCK_SIZE);
  blelloch_block_scan<<<ends_num_blocks, BLOCK_SIZE>>>(
      g_block_ends, g_block_ends, NULL, num_blocks);
  CUDA_ERROR(
      cudaGetLastError(), "Couldn't perform block scan on block ends");

  // Add the block ends to the output
  add_block_scan_ends<<<num_blocks * 2, BLOCK_SIZE>>>(
      g_output, g_block_ends, length);
  CUDA_ERROR(cudaGetLastError(), "Couldn't add block scan ends for level 2");
}

// Perform level 3 scan on groups of groups of blocks where each is of size
// `BLOCK_SIZE * 2`
void level3_scan(
    const num_t *g_input,
    num_t *g_output,
    const size_t length,
    num_t *g_block_ends,
    const size_t num_blocks,
    num_t *g_block_ends_ends,
    const size_t ends_num_blocks) {

  // Perform level 1 scan first
  level1_scan(g_input, g_output, length, g_block_ends, num_blocks);

  // Perform full level 2 scan on block ends
  level2_scan(
      g_block_ends, g_block_ends, num_blocks,
      g_block_ends_ends, ends_num_blocks);

  // Add the block ends to the output
  add_block_scan_ends<<<num_blocks * 2, BLOCK_SIZE>>>(
      g_output, g_block_ends, length);
  CUDA_ERROR(cudaGetLastError(), "Couldn't add block scan ends for level 3");
}

// Performs all prefix sum on `input` and stores the result in `output` in
// parallel on a GPU
// Assumes both `input` and `output` are allocated with size `length`
// Returns the time it took to run the scan
double scan(const num_t *input, num_t *output, const size_t length) {
  cudaError_t err;
  size_t array_size = sizeof(num_t) * length;
  size_t num_blocks = 1 + (length - 1) / (BLOCK_SIZE * 2);
  size_t ends_num_blocks = 1 + (length - 1) / (BLOCK_SIZE * BLOCK_SIZE);

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

  // Create array for block ends
  num_t *g_block_ends = NULL;
  err = cudaMalloc((void**)&g_block_ends, sizeof(num_t) * num_blocks);
  CUDA_ERROR(err, "Couldn't allocated memory for scan_ends");

  // Create array for the ends of block ends
  num_t *g_block_ends_ends = NULL;
  err = cudaMalloc((void**)&g_block_ends_ends, sizeof(num_t) * ends_num_blocks);
  CUDA_ERROR(err, "Couldn't allocated memory for scan_ends");

  // Setup timing kernels
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // Perform the scan
  if (length <= BLOCK_SIZE * 2) {
    level1_scan(g_input, g_output, length, NULL, num_blocks);
  } else if (length <= BLOCK_SIZE * BLOCK_SIZE * 4) {
    level2_scan(g_input, g_output, length, g_block_ends, num_blocks);
  } else if (length <= BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE * 8) {
    level3_scan(
        g_input, g_output, length,
        g_block_ends, num_blocks,
        g_block_ends_ends, ends_num_blocks);
  } else {
    fprintf(stderr, "Couldn't handle array of size %ld\n", length);
    exit(1);
  }

  // Stop timing kernels
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsed_time_ms;
  cudaEventElapsedTime(&elapsed_time_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Copy results to host
  err = cudaMemcpy(output, g_output, array_size, cudaMemcpyDeviceToHost);
  CUDA_ERROR(err, "Couldn't copy output to host");

  // Free device allocated memory
  err = cudaFree(g_input);
  CUDA_ERROR(err, "Couldn't free input on host");
  err = cudaFree(g_output);
  CUDA_ERROR(err, "Couldn't free output on host");
  err = cudaFree(g_block_ends);
  CUDA_ERROR(err, "Couldn't free block scan ends on host");

  return (double)elapsed_time_ms;
}

// Performs all prefix sum on `input` and stores the result in `output`
// sequentially on the CPU
// Assumes both `input` and `output` are allocated with size `length`
// Returns the time it took to run the scan
float sequential_scan(const num_t *input, num_t *output, size_t length) {
  // Start timer for sequential scan
  StopWatchInterface *sequential_timer = NULL;
  sdkCreateTimer(&sequential_timer);
  sdkStartTimer(&sequential_timer);

  for (size_t i = 1; i < length; i++) {
    output[i] = output[i - 1] + input[i - 1];
  }

  // Stop timers for sequential scan
  sdkStopTimer(&sequential_timer);
  return sdkGetTimerValue(&sequential_timer);
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
  double sequential_time_elapsed_ms = sequential_scan(
      input, truth_output, ARRAY_SIZE);

  // Run the parallel scan
  double parallel_time_elapsed_ms = scan(input, output, ARRAY_SIZE);

  // Compare solutions
  bool are_equal = print_array_equality(truth_output, output, ARRAY_SIZE);

  if (are_equal) {
    printf("Success!\n");
  }

  printf("Sequential time: %f ms\n", sequential_time_elapsed_ms);
  printf("Parallel time: %f ms\n", parallel_time_elapsed_ms);
  printf(
      "Speed up: %f times\n",
      sequential_time_elapsed_ms / parallel_time_elapsed_ms);

  free(input);
  free(truth_output);
  free(output);

  return are_equal ? 0 : 1;
}

