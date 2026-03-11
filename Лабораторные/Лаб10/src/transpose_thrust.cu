#include <stdio.h>
#include <stdlib.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <sys/time.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

struct index_transformer {
    int rows, cols;
    index_transformer(int _rows, int _cols) : rows(_rows), cols(_cols) {}
    
    __host__ __device__
    int operator()(int idx) const {
        int row = idx / cols;
        int col = idx % cols;
        return col * rows + row;
    }
};

void transpose_thrust_fastest(int rows, int cols,
                              const thrust::host_vector<float>& h_input,
                              thrust::host_vector<float>& h_output) {
    
    thrust::device_vector<float> d_input = h_input;
    thrust::device_vector<float> d_output(rows * cols);
    
    thrust::counting_iterator<int> first(0);
    thrust::transform_iterator<index_transformer, thrust::counting_iterator<int>> 
        indices(first, index_transformer(rows, cols));
    
    thrust::copy(thrust::make_permutation_iterator(d_input.begin(), indices),
                 thrust::make_permutation_iterator(d_input.begin(), indices + rows * cols),
                 d_output.begin());
    
    h_output = d_output;
}

int main(int argc, char* argv[]) {
    int rows = 1024, cols = 1024;
    if (argc >= 3) {
        rows = atoi(argv[1]);
        cols = atoi(argv[2]);
    }
    
    double mem_size = (double)(rows * cols * sizeof(float)) / (1024 * 1024);
    
    printf("========================================================\n");
    printf("ТРАНСПОНИРОВАНИЕ МАТРИЦЫ (Thrust)\n");
    printf("========================================================\n");
    printf("Размер матрицы: %d x %d\n", rows, cols);
    printf("Память: %.2f MB\n", mem_size);
    
    thrust::host_vector<float> h_input(rows * cols);
    thrust::host_vector<float> h_output(rows * cols);
    
    for (int i = 0; i < rows * cols; i++) {
        h_input[i] = i % 10;
    }
    
    transpose_thrust_fastest(rows, cols, h_input, h_output);
    
    double start = get_time();
    transpose_thrust_fastest(rows, cols, h_input, h_output);
    double end = get_time();
    
    double elapsed = end - start;
    double bandwidth = 2.0 * rows * cols * sizeof(float) / elapsed / 1e9;
    
    printf("\nРЕЗУЛЬТАТЫ:\n");
    printf("  Время: %.6f сек\n", elapsed);
    //printf("  Пропускная способность: %.2f GB/s\n", bandwidth);
    printf("  Проверка: input[0][1]=%.0f -> output[1][0]=%.0f\n", 
           h_input[1], h_output[rows]);
    
    return 0;
}