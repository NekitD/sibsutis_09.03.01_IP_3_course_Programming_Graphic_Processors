#include <stdio.h>
#include <stdlib.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <sys/time.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

float dot_product_thrust(int n, const float* h_a, const float* h_b) {
    thrust::device_vector<float> d_a(h_a, h_a + n);
    thrust::device_vector<float> d_b(h_b, h_b + n);
    
    float result = thrust::inner_product(d_a.begin(), d_a.end(), d_b.begin(), 0.0f);
    
    return result;
}

int main(int argc, char* argv[]) {
    int n = 1024 * 1024;
    if (argc >= 2) n = atoi(argv[1]);
    
    printf("========================================================\n");
    printf("СКАЛЯРНОЕ ПРОИЗВЕДЕНИЕ (Thrust)\n");
    printf("========================================================\n");
    printf("Размер векторов: %d\n", n);
    printf("Память: %.2f MB\n", 2.0 * n * sizeof(float) / (1024*1024));
    
    thrust::host_vector<float> h_a(n, 1.0f);
    thrust::host_vector<float> h_b(n, 1.0f);
    
    float warmup = dot_product_thrust(n, h_a.data(), h_b.data());
    
    double start = get_time();
    float result = dot_product_thrust(n, h_a.data(), h_b.data());
    double end = get_time();
    
    double elapsed = end - start;
    double gflops = (2.0 * n) / elapsed / 1e9;
    
    printf("\nРЕЗУЛЬТАТЫ:\n");
    printf("  Результат: %.2f (ожидается %.2f)\n", result, (float)n);
    printf("  Время: %.6f сек\n", elapsed);
    //printf("  Производительность: %.2f GFLOPS\n", gflops);
    
    return 0;
}