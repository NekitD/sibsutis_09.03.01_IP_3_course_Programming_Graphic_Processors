#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

double test_pageable(size_t size, int iterations) {
    char *h_pageable, *d_data;
    
    h_pageable = (char*)malloc(size);
    
    cudaMalloc(&d_data, size);
    
    for (int i = 0; i < 5; i++) {
        cudaMemcpy(d_data, h_pageable, size, cudaMemcpyHostToDevice);
        cudaMemcpy(h_pageable, d_data, size, cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
    
    double start = get_time();
    for (int i = 0; i < iterations; i++) {
        cudaMemcpy(d_data, h_pageable, size, cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();
    double h2d_time = (get_time() - start) / iterations;
    
    start = get_time();
    for (int i = 0; i < iterations; i++) {
        cudaMemcpy(h_pageable, d_data, size, cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
    double d2h_time = (get_time() - start) / iterations;
    
    cudaFree(d_data);
    free(h_pageable);
    
    return h2d_time + d2h_time;
}

double test_pinned(size_t size, int iterations) {
    char *h_pinned, *d_data;
    
    cudaMallocHost(&h_pinned, size);
    cudaMalloc(&d_data, size);
    
    for (int i = 0; i < 5; i++) {
        cudaMemcpy(d_data, h_pinned, size, cudaMemcpyHostToDevice);
        cudaMemcpy(h_pinned, d_data, size, cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
    
    double start = get_time();
    for (int i = 0; i < iterations; i++) {
        cudaMemcpy(d_data, h_pinned, size, cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();
    double h2d_time = (get_time() - start) / iterations;
    
    start = get_time();
    for (int i = 0; i < iterations; i++) {
        cudaMemcpy(h_pinned, d_data, size, cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
    double d2h_time = (get_time() - start) / iterations;
    
    cudaFree(d_data);
    cudaFreeHost(h_pinned);
    
    return h2d_time + d2h_time;
}

int main() {
    printf("========================================================\n");
    printf("СРАВНЕНИЕ ОБЫЧНОЙ И ЗАКРЕПЛЕННОЙ ПАМЯТИ\n");
    printf("========================================================\n");
    
    size_t sizes[] = {1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int iterations = 100;
    
    printf("%-18s %-22s %-24s %-23s %-23s %-22s %-30s\n", 
           "Размер", "Обычная(с)", "Закрепл.(с)", "Ускорение", 
           "Обычная(GB/s)", "Закрепл.(GB/s)", "Выигрыш %");
    printf("------------------------------------------------------------------------------------------------------------------------------------\n");
    
    for (int i = 0; i < num_sizes; i++) {
        size_t size = sizes[i];
        double size_mb = size / (1024.0 * 1024.0);
        
        double pageable_time = test_pageable(size, iterations);
        double pinned_time = test_pinned(size, iterations);
        double speedup = pageable_time / pinned_time;
        
        double pageable_bw = (2.0 * size) / pageable_time / 1e9;
        double pinned_bw = (2.0 * size) / pinned_time / 1e9;
        double percent_improvement = (pinned_bw - pageable_bw) / pageable_bw * 100.0;
        
        printf("%-12zu %-15.6f %-15.6f %-15.2f %-15.2f %-15.2f %-15.1f\n", 
               size, pageable_time, pinned_time, speedup, 
               pageable_bw, pinned_bw, percent_improvement);
    }
    printf("------------------------------------------------------------------------------------------------------------------------------------\n");
    
    return 0;
}