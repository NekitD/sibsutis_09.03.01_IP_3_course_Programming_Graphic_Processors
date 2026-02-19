#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define VECTOR_SIZE (1 << 20)
#define WARMUP_ITERATIONS 100
#define MEASURE_ITERATIONS 1000

__global__ void vector_add(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        c[i] = a[i] + b[i];
    }
}

float measure_time(int threads_per_block, int *d_a, int *d_b, int *d_c, int n) {
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    for(int i = 0; i < WARMUP_ITERATIONS; i++) {
        vector_add<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);
    
    for(int i = 0; i < MEASURE_ITERATIONS; i++) {
        vector_add<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    float elapsed_time_ms;
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    
    float avg_time_ms = elapsed_time_ms / MEASURE_ITERATIONS;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return avg_time_ms;
}

double calculate_bandwidth(int n, float time_ms) {
    double bytes = 3.0 * n * sizeof(int);
    double time_s = time_ms * 1e-3;
    return bytes / time_s / 1e9;  
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("========================================================\n");
    printf("GPU: %s\n", prop.name);
    printf("Макс. нитей в блоке: %d\n", prop.maxThreadsPerBlock);
    printf("Размер вектора: %d (%.2f MB)\n", VECTOR_SIZE, (double)(VECTOR_SIZE * 3 * sizeof(int)) / (1024 * 1024));
    printf("========================================================\n\n");
    
    int *h_a, *h_b, *h_c;
    h_a = (int*)malloc(VECTOR_SIZE * sizeof(int));
    h_b = (int*)malloc(VECTOR_SIZE * sizeof(int));
    h_c = (int*)malloc(VECTOR_SIZE * sizeof(int));
    
    for(int i = 0; i < VECTOR_SIZE; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, VECTOR_SIZE * sizeof(int));
    cudaMalloc(&d_b, VECTOR_SIZE * sizeof(int));
    cudaMalloc(&d_c, VECTOR_SIZE * sizeof(int));
    
    cudaMemcpy(d_a, h_a, VECTOR_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, VECTOR_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    
    int thread_configs[] = {1, 16, 32, 64, 128, 256, 512, 1024};
    int num_configs = sizeof(thread_configs) / sizeof(thread_configs[0]);
    
    printf("%-25s %-20s %-25s %-20s\n", "Нитей/блок", "Блоков", "Время(мкс)", "Проп.спос.(GB/s)");
    printf("--------------------------------------------------------------------------------------------------------\n");
    
    for(int t = 0; t < num_configs; t++) {
        int threads = thread_configs[t];
        
        if(threads > prop.maxThreadsPerBlock) {
            printf("%-25d %-25s %-30s %-30s %-30s\n", threads, "> макс", "-", "-", "N/A");
            continue;
        }
        
        float time_ms = measure_time(threads, d_a, d_b, d_c, VECTOR_SIZE);
        float time_us = time_ms * 1000;
        
        double bandwidth = calculate_bandwidth(VECTOR_SIZE, time_ms);
        
        int blocks = (VECTOR_SIZE + threads - 1) / threads;
        
        printf("%-15d %-15d %-20.3f %-20.2f\n", threads, blocks, time_us, bandwidth);
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}