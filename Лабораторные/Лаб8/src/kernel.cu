#include "kernel.h"
#include <cuda_runtime.h>

__global__ void vector_add_kernel(float* a, float* b, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        a[tid] = a[tid] + b[tid];
    }
}