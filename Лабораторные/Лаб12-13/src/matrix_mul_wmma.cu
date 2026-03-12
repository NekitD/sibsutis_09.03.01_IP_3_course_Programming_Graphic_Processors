#include "utils.h"
#include <mma.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void matrix_mul_wmma(const half* A, const half* B, float* C, int M, int N, int K) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    int block_row = blockIdx.x * WMMA_M;
    int block_col = blockIdx.y * WMMA_N;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    for (int k = 0; k < K; k += WMMA_K) {
        const half* A_slice = A + block_row * K + k;
        const half* B_slice = B + k * N + block_col;
        
        wmma::load_matrix_sync(a_frag, A_slice, K);
        wmma::load_matrix_sync(b_frag, B_slice, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    float* C_slice = C + block_row * N + block_col;
    wmma::store_matrix_sync(C_slice, c_frag, N, wmma::mem_row_major);
}

double run_wmma(int M, int N, int K, const half* h_A, const half* h_B, float* h_C) {
    
    half *d_A, *d_B;
    float *d_C;
    
    size_t size_A = M * K * sizeof(half);
    size_t size_B = K * N * sizeof(half);
    size_t size_C = M * N * sizeof(float);
    
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    dim3 grid(M / WMMA_M, N / WMMA_N);
    
    matrix_mul_wmma<<<grid, 32>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    
    double start = get_time();
    matrix_mul_wmma<<<grid, 32>>>(d_A, d_B, d_C, M, N, K);
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
    
    if (M % WMMA_M != 0 || N % WMMA_N != 0 || K % WMMA_K != 0) {
        printf("Размеры должны быть кратны %dx%dx%d\n", WMMA_M, WMMA_N, WMMA_K);
        return 1;
    }
    
    printf("========================================================\n");
    printf("WMMA (ПРЯМОЕ ПРОГРАММИРОВАНИЕ ТЕНЗОРНЫХ ЯДЕР)\n");
    printf("========================================================\n");
    printf("Размеры: M=%d, N=%d, K=%d\n", M, N, K);
    
    half *h_A = (half*)malloc(M * K * sizeof(half));
    half *h_B = (half*)malloc(K * N * sizeof(half));
    float *h_C = (float*)malloc(M * N * sizeof(float));
    
    init_matrix(h_A, M, K, 1.0f);
    init_matrix(h_B, K, N, 1.0f);
    
    double time = run_wmma(M, N, K, h_A, h_B, h_C);
    double gflops = 2.0 * M * N * K / time / 1e9;
    double bandwidth = (M*K + K*N + M*N) * sizeof(half) / time / 1e9;
    
    printf("\nРЕЗУЛЬТАТЫ:\n");
    printf("  Время: %.6f сек\n", time);
    printf("  Производительность: %.2f GFLOPS\n", gflops);
    printf("  Пропускная способность: %.2f GB/s\n", bandwidth);
    printf("  C[0][0] = %.2f\n", h_C[0]);
    
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}