#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

double vector_add_sequential(int n) {
    size_t size = n * sizeof(float);
    
    float *h_a, *h_b, *h_c;
    cudaMallocHost(&h_a, size);
    cudaMallocHost(&h_b, size);
    cudaMallocHost(&h_c, size);
    
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    double start = get_time();
    
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    double end = get_time();
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    
    return end - start;
}

double vector_add_parallel(int total_n, int chunk_size, int num_streams) {
    int num_chunks = (total_n + chunk_size - 1) / chunk_size;
    int actual_streams = (num_streams < num_chunks) ? num_streams : num_chunks;
    
    size_t total_size = total_n * sizeof(float);
    
    float *h_a, *h_b, *h_c;
    cudaMallocHost(&h_a, total_size);
    cudaMallocHost(&h_b, total_size);
    cudaMallocHost(&h_c, total_size);
    
    for (int i = 0; i < total_n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, total_size);
    cudaMalloc(&d_b, total_size);
    cudaMalloc(&d_c, total_size);
    
    cudaStream_t streams[actual_streams];
    for (int i = 0; i < actual_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int stream_id = chunk % actual_streams;
        int offset = chunk * chunk_size;
        int current_size = (chunk == num_chunks - 1) ? total_n - offset : chunk_size;
        size_t current_bytes = current_size * sizeof(float);
        
        int threads = 256;
        int blocks = (current_size + threads - 1) / threads;
        
        cudaMemcpyAsync(&d_a[offset], &h_a[offset], current_bytes, 
                        cudaMemcpyHostToDevice, streams[stream_id]);
        cudaMemcpyAsync(&d_b[offset], &h_b[offset], current_bytes, 
                        cudaMemcpyHostToDevice, streams[stream_id]);
        vector_add<<<blocks, threads, 0, streams[stream_id]>>>(
            &d_a[offset], &d_b[offset], &d_c[offset], current_size);
        cudaMemcpyAsync(&h_c[offset], &d_c[offset], current_bytes, 
                        cudaMemcpyDeviceToHost, streams[stream_id]);
    }
    for (int i = 0; i < actual_streams; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    double start = get_time();
    
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int stream_id = chunk % actual_streams;
        int offset = chunk * chunk_size;
        int current_size = (chunk == num_chunks - 1) ? total_n - offset : chunk_size;
        size_t current_bytes = current_size * sizeof(float);
        
        int threads = 256;
        int blocks = (current_size + threads - 1) / threads;
        
        cudaMemcpyAsync(&d_a[offset], &h_a[offset], current_bytes, 
                        cudaMemcpyHostToDevice, streams[stream_id]);
        cudaMemcpyAsync(&d_b[offset], &h_b[offset], current_bytes, 
                        cudaMemcpyHostToDevice, streams[stream_id]);
        vector_add<<<blocks, threads, 0, streams[stream_id]>>>(
            &d_a[offset], &d_b[offset], &d_c[offset], current_size);
        cudaMemcpyAsync(&h_c[offset], &d_c[offset], current_bytes, 
                        cudaMemcpyDeviceToHost, streams[stream_id]);
    }
    
    for (int i = 0; i < actual_streams; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    double end = get_time();
    
    float expected = 3.0f;
    int errors = 0;
    for (int i = 0; i < 5; i++) {
        if (h_c[i] != expected) {
            errors++;
        }
    }
    if (errors > 0) {
        printf("  ОШИБКА: результаты неверны! ");
    }
    
    for (int i = 0; i < actual_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    
    return end - start;
}

int main() {
    printf("========================================================\n");
    printf("СЛОЖЕНИЕ ВЕКТОРОВ\n");
    printf("========================================================\n");
    
    int total_n = 1 << 26;  // 64M элементов
    printf("Общий размер векторов: %d элементов (%.2f MB)\n\n", 
           total_n, 3.0 * total_n * sizeof(float) / (1024*1024));
    
    int num_streams = 4;
    
    printf("%-25s %-30s %-30s %-30s %-30s\n", 
           "Размер порции", "Посл. время (с)", "Паралл. время (с)", 
           "Проп. спос. (GB/s)", "Ускорение");
    printf("--------------------------------------------------------------------------------------------------------\n");
    
    int chunk_sizes[] = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608};
    int num_sizes = sizeof(chunk_sizes) / sizeof(chunk_sizes[0]);
    
    double best_time = 1e9;
    int best_chunk = 0;
    double best_speedup = 0.0;
    
    for (int i = 0; i < num_sizes; i++) {
        int chunk = chunk_sizes[i];
    
        double seq_time = vector_add_sequential(total_n);
        double par_time = vector_add_parallel(total_n, chunk, num_streams);
        double speedup = seq_time / par_time;
        double bandwidth = 3.0 * total_n * sizeof(float) / par_time / 1e9;
        
        printf("%-15d %-20.6f %-20.6f %-20.2f %-15.2f\n", 
               chunk, seq_time, par_time, bandwidth, speedup);
        
        if (par_time < best_time && par_time < seq_time) {
            best_time = par_time;
            best_chunk = chunk;
            best_speedup = speedup;
        }
    }
    
    printf("--------------------------------------------------------------------------------------------------------\n");
    
    if (best_chunk > 0) {
        printf("Оптимальный размер порции: %d элементов\n", best_chunk);
        printf("Максимальное ускорение: %.2fx\n", best_speedup);
        printf("Пропускная способность: %.2f GB/s\n", 
               3.0 * total_n * sizeof(float) / best_time / 1e9);
    } else {
        printf("Параллельная версия не дала ускорения ни для одного размера порции!\n");
    }
    
    return 0;
}