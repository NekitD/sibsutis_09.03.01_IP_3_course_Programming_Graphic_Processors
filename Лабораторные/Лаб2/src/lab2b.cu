#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void vector_add(int *a, int *b, int n) {
    int i = threadIdx.x; 
    if(i < n) {
        a[i] += b[i];
    }
}



int main() {
    printf("Исследование зависимости времени выполнения на GPU от длины вектора\n");
    printf("Количество нитей = длине вектора\n\n");

    
    int sizes[] = {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576};
    int num_sizes = 6;
    
    printf("Размер\tВремя(мс)\tПропускная способность(GB/s)\n");
    printf("----------------------------------------\n");
    
    for(int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        
        int *d_a, *d_b;
        int *h_a, *h_b;
        
        h_a = (int*)malloc(n * sizeof(int));
        h_b = (int*)malloc(n * sizeof(int));
        
        for(int i = 0; i < n; i++) {
            h_a[i] = i;
            h_b[i] = i + 1;
        }
        
        cudaMalloc(&d_a, n * sizeof(int));
        cudaMalloc(&d_b, n * sizeof(int));
        
        cudaMemcpy(d_a, h_a, n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, n * sizeof(int), cudaMemcpyHostToDevice);
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start, 0);
        
        vector_add<<<1, n>>>(d_a, d_b, n);
        
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        double bandwidth = (2.0 * n * sizeof(int)) / (milliseconds * 1e-3) / 1e9;
        
        printf("%d\t%.3f\t\t%.2f\n", n, milliseconds, bandwidth);
        
        cudaFree(d_a);
        cudaFree(d_b);
        free(h_a);
        free(h_b);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    return 0;
}