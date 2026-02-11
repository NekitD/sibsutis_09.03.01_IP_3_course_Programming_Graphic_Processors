#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void vector_add(int *a, int *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        a[i] += b[i];
    }
}

int main() {
    printf("GPU: NVIDIA GeForce RTX 3060 Mobile\n");
    printf("Макс. нитей в блоке: 1024\n\n");
    
    int sizes[] = {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    const int THREADS_PER_BLOCK = 256;
    
    printf("%-15s %-25s %-21s %-15s\n", "Размер", "Блоков", "Время(мс)", "Проп.спос.");
    printf("--------------------------------------------------------------------------------\n");
    
    for(int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        
        int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        int *d_a, *d_b;
        int *h_a, *h_b;
        
        h_a = (int*)malloc(n * sizeof(int));
        h_b = (int*)malloc(n * sizeof(int));
        
        for(int i = 0; i < n; i++) {
            h_a[i] = i;
            h_b[i] = i + 1;
        }
        
        cudaMalloc(&d_a, n * sizeof(int));
        cudaMalloc(&d_b, n * sizeof(int));
        
        cudaMemcpy(d_a, h_a, n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, n * sizeof(int), cudaMemcpyHostToDevice);
        
        for(int i = 0; i < 10; i++) {
            vector_add<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, n);
        }
        cudaDeviceSynchronize();
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start, 0);
        vector_add<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, n);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        
        float time_ms;
        cudaEventElapsedTime(&time_ms, start, stop);
        
        double bandwidth = (3.0 * n * sizeof(int)) / (time_ms * 1e-3) / 1e9;
        
        printf("%-12d %-15d %-15.6f %-15.2f \n", 
               n, blocks, time_ms, bandwidth);
        
        cudaFree(d_a);
        cudaFree(d_b);
        free(h_a);
        free(h_b);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    return 0;
}