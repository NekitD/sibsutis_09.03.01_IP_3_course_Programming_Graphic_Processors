#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

__global__ void vector_add(int *a, int *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(threadIdx.x % 8 == 0) {          
        int *bad = a + 9999999999999999999999999999;
        *bad = 100; 
    }
    else if(i < n) {
        a[i] += b[i]; 
    }
    printf("a[%d] = %d\n", i, a[i]);
}

int main() {
    printf("GPU: NVIDIA GeForce RTX 3060 Mobile\n");
    printf("Макс. нитей в блоке: 1024\n\n");

    int size = 4096;
        
    int blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
    int *d_a, *d_b;
    int *h_a, *h_b;
        
    h_a = (int*)malloc(size * sizeof(int));
    h_b = (int*)malloc(size * sizeof(int));
        
    for(int i = 0; i < size; i++) {
        h_a[i] = i;
        h_b[i] = i + 1;
    }
        
    cudaError_t err;
        
    err = cudaMalloc(&d_a, size * sizeof(int));
    if(err != cudaSuccess) {
        printf("Ошибка cudaMalloc для d_a: %s\n", cudaGetErrorString(err));
        free(h_a);
        free(h_b);
    }
        
    err = cudaMalloc(&d_b, size * sizeof(int));
    if(err != cudaSuccess) {
        printf("Ошибка cudaMalloc для d_b: %s\n", cudaGetErrorString(err));
        cudaFree(d_a);
        free(h_a);
        free(h_b);
    }
        
    cudaMemcpy(d_a, h_a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size * sizeof(int), cudaMemcpyHostToDevice);
        
    for(int i = 0; i < 10; i++) {
        vector_add<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, size);
    }
    cudaDeviceSynchronize();
        
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
        
    cudaEventRecord(start, 0);
    vector_add<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, size);
    cudaEventRecord(stop, 0);
        
    err = cudaDeviceSynchronize();
        
    float time_ms = 0;
    double bandwidth = 0;
    char status[20] = "УСПЕХ";
        
    if(err != cudaSuccess) {
        sprintf(status, "ОШИБКА");
        printf("Ошибка: %s\n", cudaGetErrorString(err));
        cudaGetLastError(); 
    } else {
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_ms, start, stop);
        bandwidth = (3.0 * size * sizeof(int)) / (time_ms * 1e-3) / 1e9; 
    }
    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    free(h_b);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}