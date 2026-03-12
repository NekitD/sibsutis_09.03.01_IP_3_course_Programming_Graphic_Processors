#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

__global__ void dot_product_kernel(const float* a, const float* b, float* partial, int n) {
    __shared__ float cache[256];
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cache_idx = threadIdx.x;
    
    float sum = 0.0f;
    while (tid < n) {
        sum += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    
    cache[cache_idx] = sum;
    __syncthreads();
    
    int i = blockDim.x / 2;
    while (i > 0) {
        if (cache_idx < i) {
            cache[cache_idx] += cache[cache_idx + i];
        }
        __syncthreads();
        i /= 2;
    }
    
    if (cache_idx == 0) {
        partial[blockIdx.x] = cache[0];
    }
}

double dot_product_sequential(int n) {
    size_t size = n * sizeof(float);
    
    float *h_a, *h_b;
    cudaMallocHost(&h_a, size);
    cudaMallocHost(&h_b, size);
    
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 1.0f;
    }
    
    float *d_a, *d_b, *d_partial;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    
    int blocks = (n + 255) / 256;
    blocks = min(blocks, 1024);
    cudaMalloc(&d_partial, blocks * sizeof(float));
    
    float* h_partial = (float*)malloc(blocks * sizeof(float));
    
    double start = get_time();
    
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    dot_product_kernel<<<blocks, 256>>>(d_a, d_b, d_partial, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    float result = 0.0f;
    for (int i = 0; i < blocks; i++) {
        result += h_partial[i];
    }
    
    double end = get_time();
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_partial);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    free(h_partial);
    
    return end - start;
}

double dot_product_parallel(int total_n, int chunk_size, int num_streams) {
    int num_chunks = (total_n + chunk_size - 1) / chunk_size;
    int actual_streams = min(num_streams, num_chunks);
    
    size_t total_size = total_n * sizeof(float);
    
    float *h_a, *h_b;
    cudaMallocHost(&h_a, total_size);
    cudaMallocHost(&h_b, total_size);
    
    for (int i = 0; i < total_n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 1.0f;
    }
    
    float *d_a, *d_b, *d_partial;
    cudaMalloc(&d_a, total_size);
    cudaMalloc(&d_b, total_size);
    
    int blocks_per_chunk = (chunk_size + 255) / 256;
    int total_blocks = blocks_per_chunk * num_chunks;
    cudaMalloc(&d_partial, total_blocks * sizeof(float));
    
    float* h_partial = (float*)malloc(total_blocks * sizeof(float));
    
    cudaStream_t streams[actual_streams];
    for (int i = 0; i < actual_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    double start = get_time();
    
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int stream_id = chunk % actual_streams;
        int offset = chunk * chunk_size;
        int current_size = min(chunk_size, total_n - offset);
        size_t current_bytes = current_size * sizeof(float);
        
        if (current_size <= 0) continue;
        
        int blocks_this_chunk = (current_size + 255) / 256;
        int partial_offset = chunk * blocks_per_chunk;
        
        cudaMemcpyAsync(&d_a[offset], &h_a[offset], current_bytes, cudaMemcpyHostToDevice, streams[stream_id]);
        cudaMemcpyAsync(&d_b[offset], &h_b[offset], current_bytes, cudaMemcpyHostToDevice, streams[stream_id]);
        
        dot_product_kernel<<<blocks_this_chunk, 256, 0, streams[stream_id]>>>(&d_a[offset], &d_b[offset], &d_partial[partial_offset], current_size);
        
        cudaMemcpyAsync(&h_partial[partial_offset], &d_partial[partial_offset],blocks_this_chunk * sizeof(float), cudaMemcpyDeviceToHost, streams[stream_id]);
    }
    
    for (int i = 0; i < actual_streams; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    float result = 0.0f;
    for (int i = 0; i < total_blocks; i++) {
        result += h_partial[i];
    }
    
    double end = get_time();
    
    for (int i = 0; i < actual_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_partial);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    free(h_partial);
    
    return end - start;
}

int main() {
    printf("========================================================\n");
    printf("СКАЛЯРНОЕ ПРОИЗВЕДЕНИЕ ВЕКТОРОВ\n");
    printf("========================================================\n");
    
    int total_n = 1 << 26; 
    printf("Общий размер данных: %d элементов (%.2f MB)\n\n", 
           total_n, 2.0 * total_n * sizeof(float) / (1024*1024));
    
    int num_streams = 4;
    
    printf("%-25s %-30s %-30s %-30s %-30s\n", 
           "Размер порции", "Посл. время (с)", "Паралл. время (с)", "Проп. спос. (GB/s)", "Ускорение");
    printf("--------------------------------------------------------------------------------------------------------\n");
    
    int chunk_sizes[] = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576};
    int num_sizes = sizeof(chunk_sizes) / sizeof(chunk_sizes[0]);
    
    double best_speedup = 0.0;
    int best_chunk = 0;
    double best_parallel_time = 1e9;
    
    for (int i = 0; i < num_sizes; i++) {
        int chunk = chunk_sizes[i];
        
        double seq_time = dot_product_sequential(total_n);
        double par_time = dot_product_parallel(total_n, chunk, num_streams);
        double speedup = seq_time / par_time;
        double bandwidth = 2.0 * total_n * sizeof(float) / par_time / 1e9;
        
        printf("%-15d %-20.6f %-20.6f %-20.2f %-15.2f\n", 
               chunk, seq_time, par_time, bandwidth, speedup);
        
        if (par_time < best_parallel_time) {
            best_parallel_time = par_time;
            best_chunk = chunk;
            best_speedup = speedup;
        }
    }
    
    printf("--------------------------------------------------------------------------------------------------------\n");
    printf("Оптимальный размер порции: %d элементов\n", best_chunk);
    printf("Максимальное ускорение: %.2fx\n", best_speedup);
    printf("Пропускная способность: %.2f GB/s\n", 3.0 * total_n * sizeof(float) / best_parallel_time / 1e9);
    
    return 0;
}