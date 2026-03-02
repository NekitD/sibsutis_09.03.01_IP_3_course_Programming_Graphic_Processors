#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "kernel.h"

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    int N = 1024;
    size_t size = N * sizeof(float);
    
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    
    for (int i = 0; i < N; i++) {
        h_a[i] = i * 2.0f;
        h_b[i] = i * 2.0f + 1.0f;
    }
    
    printf("Исходные данные (первые 5):\n");
    for (int i = 0; i < 5; i++) {
        printf("  a[%d]=%.0f, b[%d]=%.0f\n", i, h_a[i], i, h_b[i]);
    }
    
    float *d_a, *d_b;
    checkCudaError(cudaMalloc(&d_a, size), "cudaMalloc d_a");
    checkCudaError(cudaMalloc(&d_b, size), "cudaMalloc d_b");
    
    checkCudaError(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice), "cudaMemcpy d_a");
    checkCudaError(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice), "cudaMemcpy d_b");
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    printf("\nЗапуск ядра: %d блоков x %d нитей\n", blocks, threads);
    vector_add_kernel<<<blocks, threads>>>(d_a, d_b, N);
    checkCudaError(cudaDeviceSynchronize(), "kernel execution");
    
    checkCudaError(cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost), "cudaMemcpy result");
    
    printf("\nРезультаты (первые 5):\n");
    for (int i = 0; i < 5; i++) {
        printf("  a[%d]+b[%d] = %.0f+%.0f=%.0f\n", 
               i, i, h_a[i] - h_b[i], h_b[i], h_a[i]);
    }
    
    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    free(h_b);
    
    return 0;
}