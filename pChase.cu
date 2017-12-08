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

__global__ void pChaseKernel(unsigned int * array_d, unsigned int * indexRecords_d,
    unsigned int * timeRecords_d, int iterations) {
  extern __shared__ unsigned int s_index[];
  extern __shared__ unsigned int s_tvalue[];

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

int main(int argc, char * argv[]) {
  if (argc != 4) {
    fprintf(stderr, "usage: ./pChase <arraySize> <iterations> <stride>\n");
    exit(1);
  }

  const int arraySize = atoi(argv[1]);
  const int iterations = atoi(argv[2]);
  const int stride = atoi(argv[3]);

  unsigned int * array = (unsigned int*)calloc(arraySize, sizeof(unsigned int));
  for (int i = 0; i < arraySize; i++) {
    array[i] = (i + stride) % arraySize;
  }

  unsigned int * array_d;
  cudaMalloc((void**)&array_d, arraySize * sizeof(unsigned int));
  cudaMemcpy(array_d, array, arraySize * sizeof(unsigned int), cudaMemcpyHostToDevice);
  unsigned int * indexRecords_d;
  cudaMalloc((void**)&indexRecords_d, iterations * sizeof(unsigned int));
  unsigned int * timeRecords_d;
  cudaMalloc((void**)&timeRecords_d, iterations * sizeof(unsigned int));

  // Luanch only 1 thread per experiment.
  dim3 blockDim(1);
  dim3 gridDim(1);
  pChaseKernel<<<gridDim, blockDim>>>(array_d, indexRecords_d, timeRecords_d, iterations);

  unsigned int * indexRecords = (unsigned int*)calloc(iterations, sizeof(unsigned int));
  cudaMemcpy(indexRecords, indexRecords_d, iterations * sizeof(unsigned int),
      cudaMemcpyDeviceToHost);
  unsigned int * timeRecords = (unsigned int*)calloc(iterations, sizeof(unsigned int));
  cudaMemcpy(timeRecords, timeRecords_d, iterations * sizeof(unsigned int), cudaMemcpyDeviceToHost);

  printf("arraySize: %d  iterations: %d  stride: %d\n", arraySize, iterations, stride);
  printf("=========================================\n");
  printf("#iteration: index visited | access latency\n");
  for (int i = 0; i < iterations; i++) {
    printf("%d: %d | %d\n", i, indexRecords[i], timeRecords[i]);
  }
}
