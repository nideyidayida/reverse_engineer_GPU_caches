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

__global__ void pChaseKernel(unsigned int * array_d, float * timeRecords_d, int iterations) {
  __shared__ unsigned int s_tvalue[];
  __shared__ unsigned int s_indexp[];

  for (i = 0; i < iterations; i++) {
    clock_t start_time = clock();
    unsigned int j = array_d[j];
    s_index[i] = j;
    clock_t end_time = clock();
    s_tvalue[i] = (end_time - start_time) / CLOCKS_PER_SEC;
  }
}

int main(int argc, char * argv[]) {
  if (argc != 4) {
    fprintf(stderr, "usage: ./pChase <arraySize> <iterations> <stride>\n");
    exit(1);
  }

  int arraySize = argv[1];
  int iterations = argv[2];
  int stride = argv[3];

  unsigned int * array = calloc(arraySize, sizeof(unsigned int));
  for (int i = 0; i < arraySize; i++) {
    array[i] = (i + stride) % arraySize;
  }

  unsigned int * array_d;
  cudaMalloc((void**)&array_d, arraySize * sizeof(unsigned int));
  cudaMemcpy(array_d, array, arraySize * sizeof(unsigned int), cudaMemcpyHostToDevice);
  float * timeRecords_d;
  cudaMalloc((void**)&timeRecords_d, iterations * sizeof(float));

  // Luanch only 1 thread each time.
  dim3 blockDim(1);
  dim3 gridDim(1);
  pChaseKernel<<<gridDim, blockDim>>>(array_d, timeRecords_d, iterations);

  float * timeRecords = calloc(iterations, sizeof(float));
  cudaMemcpy(timeRecords, timeRecords_d, iterations * sizeof(float), cudaMemcpyDeviceToHost);

  printf("====== P Chase ======\n
      arraySize: %d\n
      iterations: %d\n
      stride: %d\n",
      arraySize, iterations, stride);
  for (int i = 0; i < iterations; i++) {
    printf("time of iteration %d: %d\n", i, timeRecords[i]);
  }
}