#include "utils.h"
#include <stdlib.h>
#include <string.h>

// Структура нейросети
typedef struct {
    int* sizes;           // размеры слоёв
    int num_layers;       // количество слоёв
    
    float** h_weights;    // веса (host)
    float** h_biases;     // смещения (host)
    float** d_weights;    // веса (device)
    float** d_biases;     // смещения (device)
    
    float** h_activations; // активации (host)
    float** d_activations; // активации (device)
    float** d_zs;          // z-значения (device)
} NeuralNetwork;

// Ядро для прямого распространения (один слой)
__global__ void feedforward_kernel(float* output, const float* input, 
                                    const float* weights, const float* biases,
                                    int input_size, int output_size) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j >= output_size) return;
    
    float z = biases[j];
    for (int i = 0; i < input_size; i++) {
        z += weights[j * input_size + i] * input[i];
    }
    output[j] = sigmoid(z);
}

// Ядро для обратного распространения (один слой)
__global__ void backprop_kernel(float* delta_next, float* delta_curr,
                                 const float* weights_next, const float* z_curr,
                                 int curr_size, int next_size) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j >= curr_size) return;
    
    float sum = 0.0f;
    for (int k = 0; k < next_size; k++) {
        sum += weights_next[k * curr_size + j] * delta_next[k];
    }
    delta_curr[j] = sum * sigmoid_prime(z_curr[j]);
}

// Ядро для обновления весов
__global__ void update_weights_kernel(float* weights, float* delta, const float* activation_prev,
                                        int input_size, int output_size,
                                       float eta, int batch_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int total = output_size * input_size;
    if (idx >= total) return;
    
    int j = idx / input_size;   // нейрон в текущем слое
    int i = idx % input_size;   // нейрон в предыдущем слое
    
    weights[idx] -= (eta / batch_size) * delta[j] * activation_prev[i];
}

// Ядро для обновления смещений
__global__ void update_biases_kernel(float* biases, float* delta, 
                                      int size, float eta, int batch_size) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j >= size) return;
    biases[j] -= (eta / batch_size) * delta[j];
}

// Инициализация нейросети
void init_network(NeuralNetwork* net, int* sizes, int num_layers) {
    net->sizes = sizes;
    net->num_layers = num_layers;
    
    // Выделение памяти на хосте
    net->h_weights = (float**)malloc((num_layers - 1) * sizeof(float*));
    net->h_biases = (float**)malloc((num_layers - 1) * sizeof(float*));
    net->d_weights = (float**)malloc((num_layers - 1) * sizeof(float*));
    net->d_biases = (float**)malloc((num_layers - 1) * sizeof(float*));
    net->h_activations = (float**)malloc(num_layers * sizeof(float*));
    net->d_activations = (float**)malloc(num_layers * sizeof(float*));
    net->d_zs = (float**)malloc((num_layers - 1) * sizeof(float*));
    
    // Инициализация весов и смещений случайными значениями
    srand(42);
    for (int l = 0; l < num_layers - 1; l++) {
        int input_size = sizes[l];
        int output_size = sizes[l + 1];
        int weight_size = output_size * input_size;
        
        // Выделение на хосте
        net->h_weights[l] = (float*)malloc(weight_size * sizeof(float));
        net->h_biases[l] = (float*)malloc(output_size * sizeof(float));
        
        // Инициализация случайными числами
        for (int i = 0; i < weight_size; i++) {
            net->h_weights[l][i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        }
        for (int i = 0; i < output_size; i++) {
            net->h_biases[l][i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        }
        
        // Копирование на устройство
        cudaMalloc(&net->d_weights[l], weight_size * sizeof(float));
        cudaMalloc(&net->d_biases[l], output_size * sizeof(float));
        cudaMemcpy(net->d_weights[l], net->h_weights[l], weight_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(net->d_biases[l], net->h_biases[l], output_size * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    // Выделение памяти для активаций
    for (int l = 0; l < num_layers; l++) {
        net->h_activations[l] = (float*)malloc(sizes[l] * sizeof(float));
        cudaMalloc(&net->d_activations[l], sizes[l] * sizeof(float));
    }
    
    // Выделение памяти для z-значений
    for (int l = 0; l < num_layers - 1; l++) {
        cudaMalloc(&net->d_zs[l], sizes[l + 1] * sizeof(float));
    }
}

// Прямое распространение
void feedforward(NeuralNetwork* net, const float* input) {
    // Копирование входного слоя
    cudaMemcpy(net->d_activations[0], input, net->sizes[0] * sizeof(float), cudaMemcpyHostToDevice);
    
    int threads = 256;
    for (int l = 0; l < net->num_layers - 1; l++) {
        int input_size = net->sizes[l];
        int output_size = net->sizes[l + 1];
        int blocks = (output_size + threads - 1) / threads;
        
        feedforward_kernel<<<blocks, threads>>>(
            net->d_activations[l + 1],
            net->d_activations[l],
            net->d_weights[l],
            net->d_biases[l],
            input_size, output_size
        );
    }
    cudaDeviceSynchronize();
}

void backprop(NeuralNetwork* net, const float* target, 
              float** d_nabla_w, float** d_nabla_b) {
    int num_layers = net->num_layers;
    int threads = 256;
    
    // Вычисление ошибки выходного слоя
    int output_size = net->sizes[num_layers - 1];
    float* h_delta = (float*)malloc(output_size * sizeof(float));
    
    // Копируем активации на хост для вычисления ошибки
    float* h_output = (float*)malloc(output_size * sizeof(float));
    cudaMemcpy(h_output, net->d_activations[num_layers - 1], output_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < output_size; i++) {
        h_delta[i] = cost_derivative(h_output[i], target[i]) * sigmoid_prime(0); // Здесь нужно получить z
    }
    
    // Копирование обратно на устройство
    cudaMemcpy(d_nabla_b[num_layers - 2], h_delta, output_size * sizeof(float), cudaMemcpyHostToDevice);
    
    int prev_size = net->sizes[num_layers - 2];
    int total_weights = output_size * prev_size;
    int blocks = (total_weights + threads - 1) / threads;
    
    update_weights_kernel<<<blocks, threads>>>(
        net->d_weights[num_layers - 2],
        d_nabla_b[num_layers - 2],
        net->d_activations[num_layers - 2],
        prev_size, output_size,
        1.0f, 1
    );
    
    blocks = (output_size + threads - 1) / threads;
    update_biases_kernel<<<blocks, threads>>>(
        net->d_biases[num_layers - 2],
        d_nabla_b[num_layers - 2],
        output_size, 1.0f, 1
    );
    
    free(h_delta);
    free(h_output);
}

void update_mini_batch(NeuralNetwork* net, float* inputs, float* targets, 
                       int batch_size, float eta) {
    // Здесь должна быть реализация обучения на батче
    // Для упрощения оставляем заглушку
    printf("Training on batch of size %d\n", batch_size);
}

int evaluate(NeuralNetwork* net, float* test_inputs, int* test_labels, int test_size) {
    int correct = 0;
    for (int i = 0; i < test_size; i++) {
        feedforward(net, &test_inputs[i * net->sizes[0]]);
        
        // Получаем результат
        float* h_output = (float*)malloc(net->sizes[net->num_layers - 1] * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_output, net->d_activations[net->num_layers - 1],
                              net->sizes[net->num_layers - 1] * sizeof(float),
                              cudaMemcpyDeviceToHost));
        
        int predicted = 0;
        float max_val = h_output[0];
        for (int j = 1; j < net->sizes[net->num_layers - 1]; j++) {
            if (h_output[j] > max_val) {
                max_val = h_output[j];
                predicted = j;
            }
        }
        
        if (predicted == test_labels[i]) {
            correct++;
        }
        free(h_output);
    }
    return correct;
}

void free_network(NeuralNetwork* net) {
    for (int l = 0; l < net->num_layers - 1; l++) {
        free(net->h_weights[l]);
        free(net->h_biases[l]);
        cudaFree(net->d_weights[l]);
        cudaFree(net->d_biases[l]);
    }
    for (int l = 0; l < net->num_layers; l++) {
        free(net->h_activations[l]);
        cudaFree(net->d_activations[l]);
    }
    for (int l = 0; l < net->num_layers - 1; l++) {
        cudaFree(net->d_zs[l]);
    }
    
    free(net->h_weights);
    free(net->h_biases);
    free(net->d_weights);
    free(net->d_biases);
    free(net->h_activations);
    free(net->d_activations);
    free(net->d_zs);
}

int main() {
    printf("========================================================\n");
    printf("НЕЙРОСЕТЬ ДЛЯ РАСПОЗНАВАНИЯ ЦИФР (CUDA)\n");
    printf("========================================================\n");
    
    // Архитектура сети: 784 входных, 30 скрытых, 10 выходных
    int sizes[] = {784, 30, 10};
    int num_layers = 3;
    
    NeuralNetwork net;
    init_network(&net, sizes, num_layers);
    
    printf("Архитектура сети: %d -> %d -> %d\n", sizes[0], sizes[1], sizes[2]);
    
    // Здесь должна быть загрузка данных MNIST
    printf("\nДля полноценного обучения необходим датасет MNIST\n");
    printf("Размеры весов:\n");
    printf("  Слой 0->1: %d x %d = %d весов\n", sizes[1], sizes[0], sizes[1] * sizes[0]);
    printf("  Слой 1->2: %d x %d = %d весов\n", sizes[2], sizes[1], sizes[2] * sizes[1]);
    
    free_network(&net);
    printf("\nПрограмма завершена\n");
    
    return 0;
}