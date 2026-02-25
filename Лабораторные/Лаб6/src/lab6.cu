#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK_RETURN(value) { \
    cudaError_t _m_cudaStat = value; \
    if (_m_cudaStat != cudaSuccess) { \
        fprintf(stderr, "Error %s at line %d in file %s\n", \
                cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__); \
        exit(1); \
    } \
}

#define SH_DIM 32
#define WARMUP_ITERATIONS 10
#define MEASURE_ITERATIONS 100

__global__ void gInit(float* a, int N, int K) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    int n = threadIdx.y + blockIdx.y * blockDim.y;
    int K_grid = blockDim.x * gridDim.x;  
    
    if (k < K && n < N) {
        a[k + n * K_grid] = (float)(k + n * K_grid);
    }
}

__global__ void gTranspose1(float* a, float* b, int N, int K) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;  
    int n = threadIdx.y + blockIdx.y * blockDim.y;  
    int K_grid = blockDim.x * gridDim.x;
    int N_grid = blockDim.y * gridDim.y;
    
    if (k < K && n < N) {
        b[k + n * K_grid] = a[n + k * N_grid];
    }
}

__global__ void gTransposeSM(float* a, float* b, int N, int K) {
    __shared__ float cache[SH_DIM][SH_DIM]; 
    
    int k = threadIdx.x + blockIdx.x * blockDim.x; 
    int n = threadIdx.y + blockIdx.y * blockDim.y;  
    int K_grid = blockDim.x * gridDim.x;
    int N_grid = blockDim.y * gridDim.y;
    
    if (k < K && n < N) {
        cache[threadIdx.y][threadIdx.x] = a[k + n * K_grid];
    }
    
    __syncthreads();
    
    k = threadIdx.x + blockIdx.y * blockDim.x;  
    n = threadIdx.y + blockIdx.x * blockDim.y;  
    
    if (k < K && n < N) {
        b[k + n * K_grid] = cache[threadIdx.x][threadIdx.y];  // Конфликт банков
    }
}

__global__ void gTransposeSM_WC(float* a, float* b, int N, int K) {
    __shared__ float cache[SH_DIM][SH_DIM + 1];  
    
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    int n = threadIdx.y + blockIdx.y * blockDim.y;
    int K_grid = blockDim.x * gridDim.x;
    int N_grid = blockDim.y * gridDim.y;
    
    if (k < K && n < N) {
        cache[threadIdx.y][threadIdx.x] = a[k + n * K_grid];
    }
    
    __syncthreads();
    
    k = threadIdx.x + blockIdx.y * blockDim.x;
    n = threadIdx.y + blockIdx.x * blockDim.y;
    
    if (k < K && n < N) {
        b[k + n * K_grid] = cache[threadIdx.x][threadIdx.y];
    }
}

float measure_time_kernel(dim3 blocks, dim3 threads, void (*kernel)(float*, float*, int, int), float *d_a, float *d_b, int N, int K, const char* name) {
    
    for(int i = 0; i < WARMUP_ITERATIONS; i++) {
        kernel<<<blocks, threads>>>(d_a, d_b, N, K);
    }
    cudaDeviceSynchronize();
    CUDA_CHECK_RETURN(cudaGetLastError());
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);
    
    for(int i = 0; i < MEASURE_ITERATIONS; i++) {
        kernel<<<blocks, threads>>>(d_a, d_b, N, K);
    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    float elapsed_time_ms;
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    
    float avg_time_ms = elapsed_time_ms / MEASURE_ITERATIONS;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("  %s: %.6f ms\n", name, avg_time_ms);
    
    return avg_time_ms;
}

int main(int argc, char* argv[]) {
    if(argc < 3) {
        fprintf(stderr, "USAGE: %s <dimension of matrix> <dimension of threads>\n", argv[0]);
        fprintf(stderr, "Example: %s 256 32\n", argv[0]);
        return -1;
    }
    
    int N = atoi(argv[1]);    
    int dim_of_threads = atoi(argv[2]); 
    
    if(N % dim_of_threads) {
        fprintf(stderr, "Error: Matrix dimension must be divisible by thread dimension\n");
        return -1;
    }
    
    int dim_of_blocks = N / dim_of_threads;
    const int max_size = 1 << 12;
    
    if(dim_of_blocks > max_size) {
        fprintf(stderr, "Error: Too many blocks\n");
        return -1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("========================================================\n");
    printf("ТРАНСПОНИРОВАНИЕ МАТРИЦЫ\n");
    printf("========================================================\n");
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Размер матрицы: %d x %d\n", N, N);
    printf("Размер блока: %d x %d\n", dim_of_threads, dim_of_threads);
    printf("Сетка блоков: %d x %d\n", dim_of_blocks, dim_of_blocks);
    printf("========================================================\n\n");
    
    float *d_a, *d_b;
    cudaMalloc((void**)&d_a, N * N * sizeof(float));
    cudaMalloc((void**)&d_b, N * N * sizeof(float));
    CUDA_CHECK_RETURN(cudaGetLastError());
    
    dim3 threads(dim_of_threads, dim_of_threads);
    dim3 blocks(dim_of_blocks, dim_of_blocks);
    
    gInit<<<blocks, threads>>>(d_a, N, N);
    cudaDeviceSynchronize();
    CUDA_CHECK_RETURN(cudaGetLastError());
    
    
    cudaMemset(d_b, 0, N * N * sizeof(float));
    
    float t1 = measure_time_kernel(blocks, threads, gTranspose1, d_a, d_b, N, N, "Без разделяемой памяти");
    
    cudaMemset(d_b, 0, N * N * sizeof(float));
    
    float t2 = measure_time_kernel(blocks, threads, gTransposeSM, d_a, d_b, N, N, "С разделяемой памятью (с конфликтами)");
    
    cudaMemset(d_b, 0, N * N * sizeof(float));
    
    float t3 = measure_time_kernel(blocks, threads, gTransposeSM_WC, d_a, d_b, N, N, "С разделяемой памятью (без конфликтов)");
    
    printf("\nПроизводительность:\n");
    printf("----------------------------------------\n");
    printf("Без разделяемой памяти:           %.6f ms (%.2fx)\n", t1, t1/t1);
    printf("С разделяемой памятью (конфликты): %.6f ms (%.2fx)\n", t2, t1/t2);
    printf("С разделяемой памятью (без конфл.): %.6f ms (%.2fx)\n", t3, t1/t3);
    
    cudaFree(d_a);
    cudaFree(d_b);
    
    return 0;
}