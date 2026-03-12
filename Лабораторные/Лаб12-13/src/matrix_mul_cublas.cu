#include "utils.h"


template<typename T>
double run_cublas(int M, int N, int K, 
                   const T* h_A, const T* h_B, T* h_C,
                   cublasComputeType_t computeType,
                   cudaDataType_t dataType) {
    
    T *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(T);
    size_t size_B = K * N * sizeof(T);
    size_t size_C = M * N * sizeof(T);
    
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    
    T alpha = (T)1.0f;
    T beta = (T)0.0f;
    
    cublasGemmEx(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                d_B, dataType, N,
                d_A, dataType, K,
                &beta,
                d_C, dataType, N,
                computeType,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    
    cudaDeviceSynchronize();
    
    double start = get_time();
    cublasGemmEx(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                d_B, dataType, N,
                d_A, dataType, K,
                &beta,
                d_C, dataType, N,
                computeType,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    
    cudaDeviceSynchronize();
    double end = get_time();
    
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    
    cublasDestroy(handle);
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
    printf("CUBLAS С ТЕНЗОРНЫМИ ЯДРАМИ, Размеры: M=%d, N=%d, K=%d\n", M, N, K);
    printf("========================================================\n");
    
    struct TestConfig {
        const char* name;
        cublasComputeType_t computeType;
        cudaDataType_t dataType;
        int bytes;
        float scale;
    };
    
    TestConfig configs[] = {
        {"FP16", CUBLAS_COMPUTE_16F, CUDA_R_16F, 2, 1.0f},
        {"FP32", CUBLAS_COMPUTE_32F, CUDA_R_32F, 4, 1.0f},
        {"TF32", CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F, 4, 1.0f},
        {"INT8", CUBLAS_COMPUTE_32I, CUDA_R_8I, 1, 1.0f}
    };
    
    int num_configs = sizeof(configs) / sizeof(configs[0]);
    
    printf("%-15s %-20s %-15s %-15s %-15s %-15s\n", 
           "Тип", "Время (с)", "GFLOPS", "GB/s", "C[0][0]", "Ожидалось");
    printf("--------------------------------------------------------------------------------\n");
    
    float expected = 0.0f;
    for (int i = 0; i < K; i++) {
        expected += 1.0f * 1.0f;
    }
    expected *= 1.0f;
    
    for (int i = 0; i < num_configs; i++) {
        auto& cfg = configs[i];
        
        void* h_A = malloc(M * K * cfg.bytes);
        void* h_B = malloc(K * N * cfg.bytes);
        void* h_C = malloc(M * N * cfg.bytes);
        
        if (cfg.bytes == 2) {
            half* ha = (half*)h_A;
            half* hb = (half*)h_B;
            for (int j = 0; j < M * K; j++) ha[j] = __float2half(1.0f);
            for (int j = 0; j < K * N; j++) hb[j] = __float2half(1.0f);
        } else if (cfg.bytes == 4) {
            float* fa = (float*)h_A;
            float* fb = (float*)h_B;
            for (int j = 0; j < M * K; j++) fa[j] = 1.0f;
            for (int j = 0; j < K * N; j++) fb[j] = 1.0f;
        } else if (cfg.bytes == 1) {
            int8_t* ia = (int8_t*)h_A;
            int8_t* ib = (int8_t*)h_B;
            for (int j = 0; j < M * K; j++) ia[j] = 1;
            for (int j = 0; j < K * N; j++) ib[j] = 1;
        }
        
        double time = 0.0;
        float result = 0.0f;
        
        if (cfg.bytes == 2) {
            time = run_cublas<half>(M, N, K, (half*)h_A, (half*)h_B, (half*)h_C, 
                                    cfg.computeType, cfg.dataType);
            result = __half2float(((half*)h_C)[0]);
        } else if (cfg.bytes == 4) {
            time = run_cublas<float>(M, N, K, (float*)h_A, (float*)h_B, (float*)h_C, 
                                     cfg.computeType, cfg.dataType);
            result = ((float*)h_C)[0];
        } else if (cfg.bytes == 1) {
            time = run_cublas<int8_t>(M, N, K, (int8_t*)h_A, (int8_t*)h_B, (int8_t*)h_C, 
                                      cfg.computeType, cfg.dataType);
            result = (float)((int8_t*)h_C)[0];
        }
        
        double gflops = 2.0 * M * N * K / time / 1e9;
        double bandwidth = (M*K + K*N + M*N) * cfg.bytes / time / 1e9;
        
        printf("%-10s %-15.6f %-15.2f %-15.2f %-15.2f %-15.2f\n",
               cfg.name, time, gflops, bandwidth, result, expected);
        
        free(h_A);
        free(h_B);
        free(h_C);
    }

    printf("\n\n");
    
    return 0;
}