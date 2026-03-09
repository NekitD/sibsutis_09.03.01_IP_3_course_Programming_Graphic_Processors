#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>

#define CHECK_CUDA(call) { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char* err_str; \
        cuGetErrorString(err, &err_str); \
        printf("CUDA Error: %s at %s:%d\n", err_str, __FILE__, __LINE__); \
        exit(1); \
    } \
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char* argv[]) {
    int M = 1024, N = 1024, K = 1024;
    
    if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    printf("========================================================\n");
    printf("CUDA Driver API (C++) - Matrix Multiplication\n");
    printf("========================================================\n");
    printf("Matrix sizes: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Memory: A=%.2f MB, B=%.2f MB, C=%.2f MB\n", 
           size_A/(1024.0*1024.0), size_B/(1024.0*1024.0), size_C/(1024.0*1024.0));
    
    CHECK_CUDA(cuInit(0));
    
    CUdevice device;
    CHECK_CUDA(cuDeviceGet(&device, 0));
    
    char device_name[256];
    cuDeviceGetName(device_name, sizeof(device_name), device);
    printf("Device: %s\n", device_name);
    
    CUcontext context;
    CHECK_CUDA(cuCtxCreate(&context, 0, device));
    
    CUmodule module;
    CHECK_CUDA(cuModuleLoad(&module, "obj/matrix_mul_kernel.sm_86.cubin"));
    
    CUfunction kernel;
    CHECK_CUDA(cuModuleGetFunction(&kernel, module, "matrix_mul_kernel"));
    
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    
    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = 1.0f;
    
    CUdeviceptr d_A, d_B, d_C;
    CHECK_CUDA(cuMemAlloc(&d_A, size_A));
    CHECK_CUDA(cuMemAlloc(&d_B, size_B));
    CHECK_CUDA(cuMemAlloc(&d_C, size_C));
    
    CHECK_CUDA(cuMemcpyHtoD(d_A, h_A, size_A));
    CHECK_CUDA(cuMemcpyHtoD(d_B, h_B, size_B));
    
    void* kernel_params[] = { 
        &d_A, &d_B, &d_C, 
        &M, &N, &K 
    };
    
    int block_size = 16;
    int grid_x = (N + block_size - 1) / block_size;
    int grid_y = (M + block_size - 1) / block_size;
    
    printf("\nKernel launch configuration:\n");
    printf("  Grid: %d x %d x 1\n", grid_x, grid_y);
    printf("  Block: %d x %d x 1\n", block_size, block_size);
    
    CHECK_CUDA(cuLaunchKernel(kernel, grid_x, grid_y, 1, block_size, block_size, 1,  0, NULL, kernel_params, NULL));                     
    CHECK_CUDA(cuCtxSynchronize());
    
    double start = get_time();
    
    CHECK_CUDA(cuLaunchKernel(kernel,grid_x, grid_y, 1, block_size, block_size, 1, 0, NULL, kernel_params, NULL));
    
    CHECK_CUDA(cuCtxSynchronize());
    double end = get_time();
    
    CHECK_CUDA(cuMemcpyDtoH(h_C, d_C, size_C));
    
    double elapsed = end - start;
    double gflops = 2.0 * M * N * K / elapsed / 1e9;
    
    printf("\nRESULTS:\n");
    printf("  Time: %.6f sec\n", elapsed);
    printf("  Performance: %.2f GFLOPS\n", gflops);
    printf("  Verification: C[0][0] = %.2f (expected %.2f)\n", 
           h_C[0], (float)K);

    cuMemFree(d_A);
    cuMemFree(d_B);
    cuMemFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cuModuleUnload(module);
    cuCtxDestroy(context);
    
    return 0;
}