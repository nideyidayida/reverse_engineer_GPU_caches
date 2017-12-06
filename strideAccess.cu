/**
 * Source code for GPU class's final project:
 *   Reverse engineering the memory hierarchy of a GPU.
 *
 * Compile with: nvcc -o strideAccess strideAccess.cu
 * Run with: ./strideAccess
 *
 * Author: Yida Xu, Rihan Yang
 * Date: 12/6/2017
 */

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void strideAccessKernel(float * timeRecords_d, int stride) {
  __shared__ unsigned int shared_data[];

  unsigned int data = threadIdx.x * stride;
  clock_t start_time;
  clock_t end_time;
  float totalTime = 0;
  for (int i = 0; i <= 64; i++) {
    if (i == 1) {
      totalTime = 0;
    }
    start_time = clock();
    data = shared_data[data];
    end_time = clock();
    totalTime += (end_time - start_time) / CLOCKS_PER_SEC;
  }
  &timeRecords_d = totalTime;
}

int main(int argc, char * argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: ./strideAccess <stride>\n");
    exit(1);
  }
  int stride = argv[1];

  float * timeRecords_d;
  cudaMalloc((void**)&timeRecords_d, sizeof(float));

  // Launch a warp of threads each time.
  dim3 blockDim(32);
  dim3 gridDim(1);
  strideAccessKernel<<<gridDim, blockDim>>>(timeRecords_d, stride);

  float * timeRecords = calloc(1, sizeof(float));
  cudaMemcpy(timeRecords, timeRecords_d, sizeof(float), cudaMemcpyDeviceToHost);

  printf("====== Stride Access ======\n
      stride: %d\n
      --------------------------------\n
      time: %d\n",
      iterations, stride, &timeRecords);
}