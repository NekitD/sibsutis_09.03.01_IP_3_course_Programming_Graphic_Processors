#ifndef KERNEL_H
#define KERNEL_H

#ifdef __cplusplus
extern "C" {
#endif

__global__ void vector_add_kernel(float* a, float* b, int n);

#ifdef __cplusplus
}
#endif

#endif