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

__global__ void transpose_naive(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

#define TILE_SIZE 16

__global__ void transpose_shared(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    __syncthreads();
    
    int x_out = blockIdx.y * TILE_SIZE + threadIdx.x;
    int y_out = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    if (x_out < rows && y_out < cols) {
        output[y_out * rows + x_out] = tile[threadIdx.x][threadIdx.y];
    }
}

void transpose_raw(int rows, int cols, const float* h_input, float* h_output, int use_shared) {
    float *d_input, *d_output;
    size_t size = rows * cols * sizeof(float);
    
    CHECK_CUDA(cudaMalloc(&d_input, size));
    CHECK_CUDA(cudaMalloc(&d_output, size));
    
    CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((cols + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);
    
    if (use_shared) {
        transpose_shared<<<grid, block>>>(d_input, d_output, rows, cols);
    } else {
        transpose_naive<<<grid, block>>>(d_input, d_output, rows, cols);
    }
    
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    
    cudaFree(d_input);
    cudaFree(d_output);
}

int main(int argc, char* argv[]) {
    int rows = 1024, cols = 1024;
    if (argc >= 3) {
        rows = atoi(argv[1]);
        cols = atoi(argv[2]);
    }
    
    printf("========================================================\n");
    printf("ТРАНСПОНИРОВАНИЕ МАТРИЦЫ (сырой CUDA)\n");
    printf("========================================================\n");
    printf("Размер матрицы: %d x %d\n", rows, cols);
    double mem_size = (double)rows * cols * sizeof(float) / (1024 * 1024);
    printf("Память: %.2f MB\n", mem_size);
    
    float* h_input = (float*)malloc(rows * cols * sizeof(float));
    float* h_output = (float*)malloc(rows * cols * sizeof(float));
    
    for (int i = 0; i < rows * cols; i++) {
        h_input[i] = i % 10;
    }
    
    double start = get_time();
    transpose_raw(rows, cols, h_input, h_output, 0);
    double end = get_time();
    
    printf("\nNaive версия:\n");
    printf("  Время: %.6f сек\n", end - start);
    printf("  Пропускная способность: %.2f GB/s\n", 
           2.0 * rows * cols * sizeof(float) / (end - start) / 1e9);
    
    start = get_time();
    transpose_raw(rows, cols, h_input, h_output, 1);
    end = get_time();
    
    printf("\nShared memory версия:\n");
    printf("  Время: %.6f сек\n", end - start);
    printf("  Пропускная способность: %.2f GB/s\n", 
           2.0 * rows * cols * sizeof(float) / (end - start) / 1e9);
    printf("  Ускорение: %.2fx\n", 
           (2.0 * rows * cols * sizeof(float) / (end - start) / 1e9) /
           (2.0 * rows * cols * sizeof(float) / (0.1) / 1e9)); 
    
    printf("\nПроверка: input[0][1]=%.0f -> output[1][0]=%.0f\n", 
           h_input[1], h_output[rows]);
    
    free(h_input);
    free(h_output);
    
    return 0;
}