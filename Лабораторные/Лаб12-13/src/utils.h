#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <sys/time.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

template<typename T>
void init_matrix(T* mat, int rows, int cols, T scale = 1.0f) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (T)(rand() % 10) * scale;
    }
}

template<typename T>
bool verify_result(const T* c, const T* c_ref, int n, T eps = 1e-3) {
    for (int i = 0; i < n; i++) {
        if (abs(c[i] - c_ref[i]) > eps) {
            printf("Mismatch at %d: %f vs %f\n", i, (float)c[i], (float)c_ref[i]);
            return false;
        }
    }
    return true;
}

#endif