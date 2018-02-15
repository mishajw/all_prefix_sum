#include <stdio.h>
#include <inttypes.h>

#define CUDA_ERROR(statement, message) \
  do { \
    cudaError_t err = statement; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(err)); \
      exit(1); \
    } \
  } while (0);

#define GLOBAL_INDEX blockIdx.x * blockDim.x + threadIdx.x

// The size of the array to test on
static const size_t ARRAY_SIZE = 1000;

typedef int32_t num_t;

// Performs all prefix sum on `input` and stores the result in `output` in
// parallel on a GPU
// Assumes both `input` and `output` are allocated with size `length`
void scan(const num_t *input, num_t *output, size_t length) {
  // TODO: Implement
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
  bool found_difference = false;

  for (size_t i = 0; i < length; i++) {
    if (a[i] != b[i]) {
      found_difference = true;
      printf(
          "Arrays differ at index %ld, with values %d and %d\n", i, a[i], b[i]);
    }
  }

  return found_difference;
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

  return (int)are_equal;
}

