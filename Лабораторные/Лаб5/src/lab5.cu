#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define WARMUP_ITERATIONS 100
#define MEASURE_ITERATIONS 1000

__global__ void reorder_vectors(int *input, int *output, int num_vectors, int vector_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_vectors * vector_size;
    
    if(idx < total_elements) {
        int v = idx / vector_size;      
        int e = idx % vector_size;        
        
        int new_idx = e * num_vectors + v;
        
        output[new_idx] = input[idx];
    }
}

void reorder_vectors_cpu(int *h_input, int *h_output, int num_vectors, int vector_size) {
    printf("\nПроверка на CPU:\n");
    printf("Исходный массив: ");
    for(int v = 0; v < num_vectors; v++) {
        for(int e = 0; e < vector_size; e++) {
            printf("%d ", h_input[v * vector_size + e]);
        }
        printf("| ");
    }
    printf("\n");
    
    for(int v = 0; v < num_vectors; v++) {
        for(int e = 0; e < vector_size; e++) {
            int idx = v * vector_size + e;
            int new_idx = e * num_vectors + v;
            h_output[new_idx] = h_input[idx];
        }
    }
    
    printf("Результат CPU:   ");
    for(int i = 0; i < num_vectors * vector_size; i++) {
        printf("%d ", h_output[i]);
    }
    printf("\n");
}

float measure_time(int threads_per_block, int *d_input, int *d_output, 
                   int num_vectors, int vector_size) {
    int total_elements = num_vectors * vector_size;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    for(int i = 0; i < WARMUP_ITERATIONS; i++) {
        reorder_vectors<<<blocks, threads_per_block>>>(d_input, d_output, num_vectors, vector_size);
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);
    
    for(int i = 0; i < MEASURE_ITERATIONS; i++) {
        reorder_vectors<<<blocks, threads_per_block>>>(d_input, d_output, num_vectors, vector_size);
    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    float elapsed_time_ms;
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    
    float avg_time_ms = elapsed_time_ms / MEASURE_ITERATIONS;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return avg_time_ms;
}

double calculate_bandwidth(int total_elements, float time_ms) {
    double bytes = 2.0 * total_elements * sizeof(int);
    double time_s = time_ms * 1e-3;
    return bytes / time_s / 1e9;
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("========================================================\n");
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Макс. нитей в блоке: %d\n", prop.maxThreadsPerBlock);
    printf("========================================================\n\n");
    
    int num_vectors, vector_size;
    printf("Введите количество векторов: ");
    scanf("%d", &num_vectors);
    printf("Введите размер вектора: ");
    scanf("%d", &vector_size);
    
    int total_elements = num_vectors * vector_size;
    
    int *h_input = (int*)malloc(total_elements * sizeof(int));
    int *h_output_cpu = (int*)malloc(total_elements * sizeof(int));
    int *h_output_gpu = (int*)malloc(total_elements * sizeof(int));
    
    for(int v = 0; v < num_vectors; v++) {
        for(int e = 0; e < vector_size; e++) {
            h_input[v * vector_size + e] = v * 10 + e;
        }
    }
    
    int *d_input, *d_output;
    cudaMalloc(&d_input, total_elements * sizeof(int));
    cudaMalloc(&d_output, total_elements * sizeof(int));
    
    cudaMemcpy(d_input, h_input, total_elements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, total_elements * sizeof(int));
    
    int threads_per_block = 256;
    float time_gpu = measure_time(threads_per_block, d_input, d_output, num_vectors, vector_size);
    
    cudaMemcpy(h_output_gpu, d_output, total_elements * sizeof(int), cudaMemcpyDeviceToHost);
    
    reorder_vectors_cpu(h_input, h_output_cpu, num_vectors, vector_size);
    
    printf("\nРезультат GPU:    ");
    for(int i = 0; i < total_elements; i++) {
        printf("%d ", h_output_gpu[i]);
    }
    printf("\n");
    
    int correct = 1;
    for(int i = 0; i < total_elements; i++) {
        if(h_output_cpu[i] != h_output_gpu[i]) {
            correct = 0;
            break;
        }
    }
    

    printf("Время выполнения на GPU: %.6f мс\n", time_gpu);
    printf("Пропускная способность: %.2f GB/s\n", calculate_bandwidth(total_elements, time_gpu));
    
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output_cpu);
    free(h_output_gpu);
    
    return 0;
}