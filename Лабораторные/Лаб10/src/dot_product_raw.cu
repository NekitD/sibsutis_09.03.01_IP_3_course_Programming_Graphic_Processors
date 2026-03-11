#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

__global__ void dot_product_kernel(const float* a, const float* b, float* partial_sums, int n) {
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
        partial_sums[blockIdx.x] = cache[0];
    }
}

float dot_product_raw(int n, const float* h_a, const float* h_b) {
    float *d_a, *d_b, *d_partial;
    float *h_partial;
    float result = 0.0f;
    
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    blocks = min(blocks, 1024);  
    
    CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_partial, blocks * sizeof(float)));
    
    h_partial = (float*)malloc(blocks * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice));
    
    dot_product_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_partial, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < blocks; i++) {
        result += h_partial[i];
    }
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_partial);
    free(h_partial);
    
    return result;
}

int main(int argc, char* argv[]) {
    int n = 1024 * 1024; 
    if (argc >= 2) n = atoi(argv[1]);
    
    printf("========================================================\n");
    printf("СКАЛЯРНОЕ ПРОИЗВЕДЕНИЕ (сырой CUDA)\n");
    printf("========================================================\n");
    printf("Размер векторов: %d\n", n);
    printf("Память: %.2f MB\n", 2.0 * n * sizeof(float) / (1024*1024));
    
    float *h_a = (float*)malloc(n * sizeof(float));
    float *h_b = (float*)malloc(n * sizeof(float));
    
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 1.0f;
    }
    
    float warmup = dot_product_raw(n, h_a, h_b);
    
    double start = get_time();
    float result = dot_product_raw(n, h_a, h_b);
    double end = get_time();
    
    double elapsed = end - start;
    double gflops = (2.0 * n) / elapsed / 1e9;
    
    printf("\nРЕЗУЛЬТАТЫ:\n");
    printf("  Результат: %.2f (ожидается %.2f)\n", result, (float)n);
    printf("  Время: %.6f сек\n", elapsed);
    //printf("  Производительность: %.2f GFLOPS\n", gflops);
    
    free(h_a);
    free(h_b);
    
    return 0;
}