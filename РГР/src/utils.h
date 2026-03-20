#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <sys/time.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Функция активации сигмоида
__device__ __host__ inline float sigmoid(float z) {
    return 1.0f / (1.0f + expf(-z));
}

// Производная сигмоида
__device__ __host__ inline float sigmoid_prime(float z) {
    float s = sigmoid(z);
    return s * (1.0f - s);
}

// Функция стоимости (MSE)
__device__ __host__ inline float cost_derivative(float output, float target) {
    return output - target;
}

#endif