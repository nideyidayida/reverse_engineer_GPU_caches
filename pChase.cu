/**
 * Source code for GPU class's final project:
 *   Reverse engineering the memory hierarchy of a GPU.
 *
 * Compile with: nvcc -o pChase pChase.cu
 * Run with: ./benchmark
 *
 * Author: Yida Xu, Rihan Yang
 * Date: 12/5/2017
 */

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define EMPTY_LINE "\n"
#define ERROR_MESSAGE \
    "usage: ./pChase <arraySize(KB)> <iterations> <strideSize(B)>\n\
arraySize: KB, arraySize * 256 % 1 = 0\n\
iterations: number of iterations.\n\
strideSize: Byte, strideSize % 4 = 0\n"
#define OUTPUT_FORMAT "%d: %d | %d\n"
#define OUTPUT_TITLE "\
Input: arraySize: %.2fKB  iterations: %d  strideSize: %dB\n\
===================================================\n\
Output: #iteration: index visited | access latency\n"
#define UNSIGNED_INT_SIZE sizeof(unsigned int)

__global__ void pChaseKernel(unsigned int * array_d, unsigned int * indexRecords_d,
    unsigned int * timeRecords_d, int iterations) {
  const int SHARED_MEMORY_SIZE = 6000;
  __shared__ unsigned int s_index[SHARED_MEMORY_SIZE / 2];
  __shared__ unsigned int s_tvalue[SHARED_MEMORY_SIZE / 2];

  clock_t start_time;
  clock_t end_time;
  unsigned int j = 0;
  for (int i = 0; i < iterations; i++) {
    start_time = clock();
    j = array_d[j];
    s_index[i] = j;
    end_time = clock();
    s_tvalue[i] = end_time - start_time;
  }

  for (int i = 0; i < iterations; i++) {
    indexRecords_d[i] = s_index[i];
    timeRecords_d[i] = s_tvalue[i];
  }
}

void printErrorAndExist() {
  fprintf(stderr, ERROR_MESSAGE);
  exit(1);
}

int main(int argc, char * argv[]) {
  const int multiple = 1024 / UNSIGNED_INT_SIZE;
  if (argc != 4) {
    printErrorAndExist();
  }
  const double arraySize = atof(argv[1]);
  if (arraySize * multiple != ceil(arraySize * multiple)) {
    printErrorAndExist();
  }
  const int arrayLength = arraySize * multiple;
  const int iterations = atoi(argv[2]);
  const int strideSize = atoi(argv[3]);
  if (atoi(argv[3]) % 4 != 0) {
    printErrorAndExist();
  }
  const int strideLength = strideSize / UNSIGNED_INT_SIZE;

  unsigned int * array = (unsigned int*)calloc(arrayLength, UNSIGNED_INT_SIZE);
  for (int i = 0; i < arrayLength; i++) {
    array[i] = (i + strideLength) % arrayLength;
  }

  unsigned int * array_d;
  cudaMalloc((void**)&array_d, arrayLength * UNSIGNED_INT_SIZE);
  cudaMemcpy(array_d, array, arrayLength * UNSIGNED_INT_SIZE, cudaMemcpyHostToDevice);
  unsigned int * indexRecords_d;
  cudaMalloc((void**)&indexRecords_d, iterations * UNSIGNED_INT_SIZE);
  unsigned int * timeRecords_d;
  cudaMalloc((void**)&timeRecords_d, iterations * UNSIGNED_INT_SIZE);

  // Luanch only 1 thread per experiment.
  pChaseKernel<<<1, 1>>>(array_d, indexRecords_d, timeRecords_d, iterations);

  unsigned int * indexRecords = (unsigned int*)calloc(iterations, UNSIGNED_INT_SIZE);
  cudaMemcpy(indexRecords, indexRecords_d, iterations * UNSIGNED_INT_SIZE, cudaMemcpyDeviceToHost);
  unsigned int * timeRecords = (unsigned int*)calloc(iterations, UNSIGNED_INT_SIZE);
  cudaMemcpy(timeRecords, timeRecords_d, iterations * UNSIGNED_INT_SIZE, cudaMemcpyDeviceToHost);

  printf(EMPTY_LINE);
  printf(OUTPUT_TITLE, arraySize, iterations, strideSize);
  for (int i = 0; i < iterations; i++) {
    printf(OUTPUT_FORMAT, i, indexRecords[i], timeRecords[i]);
  }
  printf(EMPTY_LINE);
}
