#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define WARMUP_ITERATIONS 100
#define MEASURE_ITERATIONS 1000
#define VECTOR_SIZE (1 << 20)

__global__ void vector_add(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_many_registers(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        int temp0 = a[i];
        int temp1 = b[i];
        int temp2 = temp0 + temp1;
        int temp3 = temp2 * 2;
        int temp4 = temp3 / 3;
        int temp5 = temp4 + 100;
        int temp6 = temp5 - 50;
        int temp7 = temp6 * temp6;
        int temp8 = temp7 / temp7;
        int temp9 = temp8 + temp2;
        int temp10 = temp9 * temp1;
        int temp11 = temp10 - temp0;
        int temp12 = temp11 * temp11;
        int temp13 = temp12 / 256;
        int temp14 = temp13 + temp3;
        int temp15 = temp14 * 2;
        int temp16 = temp15 - 1000;
        int temp17 = temp16 + temp4;
        int temp18 = temp17 * temp5;
        int temp19 = temp18 / 128;
        int temp20 = temp19 + temp6;
        int temp21 = temp20 * 3;
        int temp22 = temp21 - 500;
        int temp23 = temp22 + temp7;
        int temp24 = temp23 * temp8;
        int temp25 = temp24 + temp9;
        int temp26 = temp25 * temp10;
        int temp27 = temp26 / 64;
        int temp28 = temp27 + temp11;
        int temp29 = temp28 * 5;
        int temp30 = temp29 - 200;
        int temp31 = temp30 + temp12;
        int temp32 = temp31 * temp13;
        int temp33 = temp32 / 32;
        int temp34 = temp33 + temp14;
        int temp35 = temp34 * 7;
        int temp36 = temp35 - 100;
        int temp37 = temp36 + temp15;
        int temp38 = temp37 * temp16;
        int temp39 = temp38 / 16;
        int temp40 = temp39 + temp17;
        int temp41 = temp40 * 11;
        int temp42 = temp41 - 50;
        int temp43 = temp42 + temp18;
        int temp44 = temp43 * temp19;
        int temp45 = temp44 / 8;
        int temp46 = temp45 + temp20;
        int temp47 = temp46 * 13;
        int temp48 = temp47 - 25;
        int temp49 = temp48 + temp21;
        int temp50 = temp49 * temp22;
        int temp51 = temp50 / 4;
        int temp52 = temp51 + temp23;
        int temp53 = temp52 * 17;
        int temp54 = temp53 - 10;
        int temp55 = temp54 + temp24;
        int temp56 = temp55 * temp25;
        int temp57 = temp56 / 2;
        int temp58 = temp57 + temp26;
        int temp59 = temp58 * 19;
        int temp60 = temp59 - 5;
        
        c[i] = temp60;
    }
}

float measure_time(int threads_per_block, int *d_a, int *d_b, int *d_c, int n, 
                   void (*kernel)(int*, int*, int*, int)) {
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    for(int i = 0; i < WARMUP_ITERATIONS; i++) {
        kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);
    
    for(int i = 0; i < MEASURE_ITERATIONS; i++) {
        kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
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

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("========================================================\n");
    printf("ИССЛЕДОВАНИЕ МЕТРИК CUDA\n");
    printf("========================================================\n");
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Макс. нитей в блоке: %d\n", prop.maxThreadsPerBlock);
    printf("Количество регистров на блок: %d\n", prop.regsPerBlock);
    printf("Объем локальной памяти: %zu байт\n", prop.localMemSizePerBlock);
    printf("========================================================\n\n");
    
    int num_vectors, vector_size;
    printf("Введите количество векторов: ");
    scanf("%d", &num_vectors);
    printf("Введите размер вектора: ");
    scanf("%d", &vector_size);
    
    int **h_a = (int**)malloc(num_vectors * sizeof(int*));
    int **h_b = (int**)malloc(num_vectors * sizeof(int*));
    int **h_c = (int**)malloc(num_vectors * sizeof(int*));
    
    int **d_a = (int**)malloc(num_vectors * sizeof(int*));
    int **d_b = (int**)malloc(num_vectors * sizeof(int*));
    int **d_c = (int**)malloc(num_vectors * sizeof(int*));
    
    for(int i = 0; i < num_vectors; i++) {
        h_a[i] = (int*)malloc(vector_size * sizeof(int));
        h_b[i] = (int*)malloc(vector_size * sizeof(int));
        h_c[i] = (int*)malloc(vector_size * sizeof(int));
        
        for(int j = 0; j < vector_size; j++) {
            h_a[i][j] = i + j;
            h_b[i][j] = i * j + 1;
        }
        
        cudaMalloc(&d_a[i], vector_size * sizeof(int));
        cudaMalloc(&d_b[i], vector_size * sizeof(int));
        cudaMalloc(&d_c[i], vector_size * sizeof(int));
        
        cudaMemcpy(d_a[i], h_a[i], vector_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b[i], h_b[i], vector_size * sizeof(int), cudaMemcpyHostToDevice);
    }
    
    int threads_per_block = 256;
    
    printf("\n========================================================\n");
    printf("1. ИССЛЕДОВАНИЕ ВРЕМЕНИ ВЫПОЛНЕНИЯ (Пункт 1.1)\n");
    printf("========================================================\n");
    
    float time_normal = measure_time(threads_per_block, d_a[0], d_b[0], d_c[0], vector_size, vector_add);
    printf("Время выполнения обычного ядра: %.6f мс\n", time_normal);
    
    float time_many_regs = measure_time(threads_per_block, d_a[0], d_b[0], d_c[0], vector_size, vector_add_many_registers);
    printf("Время выполнения ядра с множеством регистров: %.6f мс\n", time_many_regs);
    
    printf("\n========================================================\n");
    printf("2. ПРОПУСКНАЯ СПОСОБНОСТЬ ПАМЯТИ (Пункт 1.2)\n");
    printf("========================================================\n");
    
    printf("\nЗапуск профилировщика для обычного ядра...\n");
    fflush(stdout);
    
    double bandwidth = calculate_bandwidth(vector_size, time_normal);
    printf("\nТеоретическая пропускная способность: %.2f GB/s\n", bandwidth);
    

    printf("\nОчистка памяти...\n");
    for(int i = 0; i < num_vectors; i++) {
        cudaFree(d_a[i]);
        cudaFree(d_b[i]);
        cudaFree(d_c[i]);
        free(h_a[i]);
        free(h_b[i]);
        free(h_c[i]);
    }
    
    free(d_a);
    free(d_b);
    free(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    
    printf("Готово!\n");
    
    return 0;
}