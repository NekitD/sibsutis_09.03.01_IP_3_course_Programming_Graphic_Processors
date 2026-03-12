#include "utils.h"

__global__ void matrix_mul_cuda(const float* A, const float* B, float* C, 
                                 int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

double run_cuda_kernel(int M, int N, int K, 
                        const float* h_A, const float* h_B, float* h_C) {
    
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    
    matrix_mul_cuda<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    
    double start = get_time();
    matrix_mul_cuda<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    double end = get_time();
    
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return end - start;
}

int main(int argc, char* argv[]) {
    int M = 1024, N = 1024, K = 1024;
    if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }
    
    printf("========================================================\n");
    printf("ОБЫЧНОЕ CUDA ЯДРО (FP32), Размеры: M=%d, N=%d, K=%d\n", M, N, K);
    printf("========================================================\n");
    printf("Размеры: M=%d, N=%d, K=%d\n", M, N, K);
    
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));
    
    init_matrix(h_A, M, K, 1.0f);
    init_matrix(h_B, K, N, 1.0f);
    
    double time = run_cuda_kernel(M, N, K, h_A, h_B, h_C);
    double gflops = 2.0 * M * N * K / time / 1e9;
    
    printf("Время: %.6f сек\n", time);
    printf("Производительность: %.2f GFLOPS\n", gflops);
    printf("C[0][0] = %.2f\n", h_C[0]);
    
    free(h_A);
    free(h_B);
    free(h_C);

    printf("\n\n");
    
    return 0;
}