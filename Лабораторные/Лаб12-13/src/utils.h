#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <sys/time.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
}

#define CUBLAS_CHECK(call) { \
    cublasStatus_t stat = call; \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS Error: %d at %s:%d\n", stat, __FILE__, __LINE__); \
        exit(1); \
    } \
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void init_matrix(float* mat, int rows, int cols, float scale = 1.0f) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)(rand() % 10) * scale;
    }
}

void init_matrix(half* mat, int rows, int cols, float scale = 1.0f) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = __float2half((float)(rand() % 10) * scale);
    }
}

void init_matrix(int8_t* mat, int rows, int cols, float scale = 1.0f) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (int8_t)((rand() % 10) * scale);
    }
}

void init_matrix(int32_t* mat, int rows, int cols, float scale = 1.0f) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (int32_t)((rand() % 10) * scale);
    }
}

bool verify_result(const float* c, const float* c_ref, int n, float eps = 1e-3) {
    for (int i = 0; i < n; i++) {
        if (abs(c[i] - c_ref[i]) > eps) {
            printf("Mismatch at %d: %f vs %f\n", i, c[i], c_ref[i]);
            return false;
        }
    }
    return true;
}

bool verify_result(const half* c, const half* c_ref, int n, float eps = 1e-1) {
    for (int i = 0; i < n; i++) {
        float fc = __half2float(c[i]);
        float fc_ref = __half2float(c_ref[i]);
        if (abs(fc - fc_ref) > eps) {
            printf("Mismatch at %d: %f vs %f\n", i, fc, fc_ref);
            return false;
        }
    }
    return true;
}

bool verify_result(const int8_t* c, const int8_t* c_ref, int n, float eps = 0) {
    for (int i = 0; i < n; i++) {
        if (c[i] != c_ref[i]) {
            printf("Mismatch at %d: %d vs %d\n", i, c[i], c_ref[i]);
            return false;
        }
    }
    return true;
}

#endif