#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <ctime>

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
    float* d_target;       // целевое значение для текущего примера (device)
} NeuralNetwork;

typedef struct {
    float** d_nabla_w;    // градиенты весов (device)
    float** d_nabla_b;    // градиенты смещений (device)
} Gradients;

__global__ void feedforward_kernel(float* output, const float* input, const float* weights, const float* biases,
                                    int input_size, int output_size);
__global__ void backprop_kernel(float* delta_next, float* delta_curr,
                                 const float* weights_next, const float* z_curr,
                                 int curr_size, int next_size);
__global__ void compute_output_delta_kernel(float* delta, const float* output, 
                                             const float* target, const float* z,
                                             int size);
__global__ void accumulate_weights_gradient_kernel(float* nabla_w, const float* delta,
                                                    const float* activation_prev,
                                                    int input_size, int output_size);
__global__ void accumulate_biases_gradient_kernel(float* nabla_b, const float* delta,
                                                   int size);
__global__ void zero_gradients_kernel(float* data, int size);
__global__ void apply_gradients_kernel(float* weights, float* nabla_w,
                                        int size, float eta, int batch_size);

void init_network(NeuralNetwork* net, int* sizes, int num_layers);
void init_gradients(NeuralNetwork* net, Gradients* grads);
void free_gradients(NeuralNetwork* net, Gradients* grads);
void feedforward(NeuralNetwork* net, const float* input);
void feedforward_stream(NeuralNetwork* net, cudaStream_t stream);
void backprop_single(NeuralNetwork* net, const float* input, const float* target,
                     Gradients* grads, cudaStream_t stream);
void update_mini_batch(NeuralNetwork* net, float* h_inputs, int* h_targets, 
                       int batch_size, float eta);
int evaluate(NeuralNetwork* net, float* test_inputs, int* test_labels, int test_size);
void free_network(NeuralNetwork* net);

float* load_digit_image(const char* filename);
int predict(NeuralNetwork* net, float* image, int image_size); 
int load_model(NeuralNetwork* net, const char* filename);
void save_model(NeuralNetwork* net, const char* filename);

typedef struct {
    float* images;
    int* labels;
    int count;
    int image_size;
} Dataset;

int load_binary_dataset(const char* images_path, const char* labels_path, 
                        Dataset* dataset, int expected_image_size);
void free_dataset(Dataset* dataset);


//------------------------------------------------------------------------------------------------------------
// MAIN
//------------------------------------------------------------------------------------------------------------

int main() {
    printf("========================================================\n");
    printf("NEURONIKITOS: DIGIT EDITION\n");
    printf("========================================================\n");
    
    // Архитектура сети: 784 входных, 30 скрытых, 10 выходных
    int sizes[] = {784, 30, 10};
    int num_layers = 3;
    
    Dataset train, test;
    
    printf("\nЗагрузка данных MNIST...\n");
    
    if (load_binary_dataset("img/train_images.bin", "img/train_labels.bin", &train, 784) != 0) {
        printf("Не удалось загрузить обучающую выборку!\n");
        printf("Запустите convert_mnist.py для конвертации данных\n");
        return -1;
    }
    
    if (load_binary_dataset("img/test_images.bin", "img/test_labels.bin", &test, 784) != 0) {
        printf("Не удалось загрузить тестовую выборку!\n");
        free_dataset(&train);
        return -1;
    }
    
    printf("Обучающая выборка: %d изображений\n", train.count);
    printf("Тестовая выборка: %d изображений\n", test.count);
    printf("Размер изображения: %d пикселей\n", train.image_size);
    
    NeuralNetwork net;
    init_network(&net, sizes, num_layers);
    
    //----------ПАРАМЕТРЫ-------------------
    int epochs = 10;
    int batch_size = 32;
    float learning_rate = 0.1f;
     //-------------------------------------
    
    printf("\nПараметры обучения:\n");
    printf("  Эпох: %d\n", epochs);
    printf("  Размер батча: %d\n", batch_size);
    printf("  Скорость обучения: %.3f\n", learning_rate);
    
    printf("\nНачало обучения...\n");
    printf("----------------------------------------\n");
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        double start_time = get_time();
        
        int num_batches = train.count / batch_size;
        for (int batch = 0; batch < num_batches; batch++) {
            int offset = batch * batch_size;
            update_mini_batch(&net, 
                              &train.images[offset * train.image_size],
                              &train.labels[offset],
                              batch_size, learning_rate);
            
            if ((batch + 1) % 100 == 0) {
                printf("\r  Батч %d/%d", batch + 1, num_batches);
                fflush(stdout);
            }
        }
        
        double end_time = get_time();
        
        int correct = evaluate(&net, test.images, test.labels, test.count);
        float accuracy = 100.0f * correct / test.count;
        
        printf("\rЭпоха %2d: точность = %.2f%% (%d/%d), время = %.2f сек\n",
               epoch + 1, accuracy, correct, test.count, end_time - start_time);
    }
    
    printf("\nОбучение завершено!\n");
    
    int final_accuracy = evaluate(&net, test.images, test.labels, test.count);
    printf("\nФинальная точность на тестовой выборке: %.2f%% (%d/%d)\n",
           100.0f * final_accuracy / test.count, final_accuracy, test.count);

    save_model(&net, "model_weights.bin");
    printf("Модель сохранена в model_weights.bin\n");

    printf("\n========================================================\n");
    printf("ТЕСТИРОВАНИЕ НА СОБСТВЕННЫХ ИЗОБРАЖЕНИЯХ\n");
    printf("========================================================\n");
    printf("Формат: PNG, JPG\n");
    printf("Введите путь к файлу (или 'exit' для выхода):\n");

    char filename[256];
    while (1) {
        printf("\n> ");
        if (scanf("%255s", filename) != 1) break;
    
        if (strcmp(filename, "exit") == 0) break;
        if (strcmp(filename, "quit") == 0) break;
    
        float* img = load_digit_image(filename);
        if (!img) {
            printf("Не удалось загрузить изображение. Попробуйте снова.\n");
            continue;
        }
        predict(&net, img, 28*28);
    
        free(img);
    }

    printf("\nРабота завершена. Спасибо за использование!\n");
    
    free_network(&net);
    free_dataset(&train);
    free_dataset(&test);
    
    return 0;
}


//------------------------------------------------------------------------------------------------------------
// ЯДРА
//------------------------------------------------------------------------------------------------------------

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

__global__ void compute_output_delta_kernel(float* delta, const float* output, 
                                             const float* target, const float* z,
                                             int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= size) return;
    delta[i] = (output[i] - target[i]) * sigmoid_prime(z[i]);
}

__global__ void accumulate_weights_gradient_kernel(float* nabla_w, const float* delta,
                                                    const float* activation_prev,
                                                    int input_size, int output_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int total = output_size * input_size;
    if (idx >= total) return;
    
    int j = idx / input_size;   
    int i = idx % input_size;   
    
    atomicAdd(&nabla_w[idx], delta[j] * activation_prev[i]);
}

__global__ void accumulate_biases_gradient_kernel(float* nabla_b, const float* delta,
                                                   int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= size) return;
    atomicAdd(&nabla_b[i], delta[i]);
}

__global__ void zero_gradients_kernel(float* data, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= size) return;
    data[i] = 0.0f;
}

__global__ void apply_gradients_kernel(float* weights, float* nabla_w,
                                        int size, float eta, int batch_size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= size) return;
    weights[i] -= (eta / batch_size) * nabla_w[i];
}

//------------------------------------------------------------------------------------------------------------
// HOST ФУНКЦИИ
//------------------------------------------------------------------------------------------------------------

void init_network(NeuralNetwork* net, int* sizes, int num_layers) {
    net->sizes = sizes;
    net->num_layers = num_layers;
    net->h_weights = (float**)malloc((num_layers - 1) * sizeof(float*));
    net->h_biases = (float**)malloc((num_layers - 1) * sizeof(float*));
    net->d_weights = (float**)malloc((num_layers - 1) * sizeof(float*));
    net->d_biases = (float**)malloc((num_layers - 1) * sizeof(float*));
    net->h_activations = (float**)malloc(num_layers * sizeof(float*));
    net->d_activations = (float**)malloc(num_layers * sizeof(float*));
    net->d_zs = (float**)malloc((num_layers - 1) * sizeof(float*));
    
    cudaMalloc(&net->d_target, sizes[num_layers - 1] * sizeof(float));
    
    srand(time(NULL));
    for (int l = 0; l < num_layers - 1; l++) {
        int input_size = sizes[l];
        int output_size = sizes[l + 1];
        int weight_size = output_size * input_size;
        
        net->h_weights[l] = (float*)malloc(weight_size * sizeof(float));
        net->h_biases[l] = (float*)malloc(output_size * sizeof(float));
        
        float scale = sqrtf(2.0f / input_size);
        for (int i = 0; i < weight_size; i++) {
            net->h_weights[l][i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
        }
        for (int i = 0; i < output_size; i++) {
            net->h_biases[l][i] = 0.0f;
        }
        cudaMalloc(&net->d_weights[l], weight_size * sizeof(float));
        cudaMalloc(&net->d_biases[l], output_size * sizeof(float));
        cudaMemcpy(net->d_weights[l], net->h_weights[l], weight_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(net->d_biases[l], net->h_biases[l], output_size * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    for (int l = 0; l < num_layers; l++) {
        net->h_activations[l] = (float*)malloc(sizes[l] * sizeof(float));
        cudaMalloc(&net->d_activations[l], sizes[l] * sizeof(float));
    }
    
    for (int l = 0; l < num_layers - 1; l++) {
        cudaMalloc(&net->d_zs[l], sizes[l + 1] * sizeof(float));
    }
}

void init_gradients(NeuralNetwork* net, Gradients* grads) {
    int num_layers = net->num_layers;
    grads->d_nabla_w = (float**)malloc((num_layers - 1) * sizeof(float*));
    grads->d_nabla_b = (float**)malloc((num_layers - 1) * sizeof(float*));
    
    for (int l = 0; l < num_layers - 1; l++) {
        int input_size = net->sizes[l];
        int output_size = net->sizes[l + 1];
        int weight_size = output_size * input_size;
        
        cudaMalloc(&grads->d_nabla_w[l], weight_size * sizeof(float));
        cudaMalloc(&grads->d_nabla_b[l], output_size * sizeof(float));
        
        cudaMemset(grads->d_nabla_w[l], 0, weight_size * sizeof(float));
        cudaMemset(grads->d_nabla_b[l], 0, output_size * sizeof(float));
    }
}

void free_gradients(NeuralNetwork* net, Gradients* grads) {
    for (int l = 0; l < net->num_layers - 1; l++) {
        cudaFree(grads->d_nabla_w[l]);
        cudaFree(grads->d_nabla_b[l]);
    }
    free(grads->d_nabla_w);
    free(grads->d_nabla_b);
}

void feedforward(NeuralNetwork* net, const float* input) {

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

void feedforward_stream(NeuralNetwork* net, cudaStream_t stream) {
    int threads = 256;
    for (int l = 0; l < net->num_layers - 1; l++) {
        int input_size = net->sizes[l];
        int output_size = net->sizes[l + 1];
        int blocks = (output_size + threads - 1) / threads;
        
        feedforward_kernel<<<blocks, threads, 0, stream>>>(
            net->d_activations[l + 1],
            net->d_activations[l],
            net->d_weights[l],
            net->d_biases[l],
            input_size, output_size
        );
    }
}

void backprop_single(NeuralNetwork* net, const float* input, const float* target,
                     Gradients* grads, cudaStream_t stream) {
    int num_layers = net->num_layers;
    int threads = 256;
    
    if (input) {
        cudaMemcpyAsync(net->d_activations[0], input, net->sizes[0] * sizeof(float), 
                        cudaMemcpyHostToDevice, stream);
    }
    if (target) {
        cudaMemcpyAsync(net->d_target, target, net->sizes[num_layers - 1] * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
    }
    
    feedforward_stream(net, stream);
    
    int output_size = net->sizes[num_layers - 1];
    int blocks = (output_size + threads - 1) / threads;
    
    compute_output_delta_kernel<<<blocks, threads, 0, stream>>>(
        net->d_zs[num_layers - 2], 
        net->d_activations[num_layers - 1],
        net->d_target,
        net->d_zs[num_layers - 2],
        output_size
    );
    
    int prev_size = net->sizes[num_layers - 2];
    int weight_size = output_size * prev_size;
    blocks = (weight_size + threads - 1) / threads;
    
    accumulate_weights_gradient_kernel<<<blocks, threads, 0, stream>>>(
        grads->d_nabla_w[num_layers - 2],
        net->d_zs[num_layers - 2],
        net->d_activations[num_layers - 2],
        prev_size, output_size
    );
    
    blocks = (output_size + threads - 1) / threads;
    accumulate_biases_gradient_kernel<<<blocks, threads, 0, stream>>>(
        grads->d_nabla_b[num_layers - 2],
        net->d_zs[num_layers - 2],
        output_size
    );
    
    for (int l = num_layers - 2; l >= 1; l--) {
        int curr_size = net->sizes[l];
        int next_size = net->sizes[l + 1];
        int prev_size_curr = net->sizes[l - 1];
        
        blocks = (curr_size + threads - 1) / threads;
        backprop_kernel<<<blocks, threads, 0, stream>>>(
            net->d_zs[l],           
            net->d_zs[l - 1],      
            net->d_weights[l],     
            net->d_zs[l - 1],       
            curr_size, next_size
        );
        
        weight_size = curr_size * prev_size_curr;
        blocks = (weight_size + threads - 1) / threads;
        
        accumulate_weights_gradient_kernel<<<blocks, threads, 0, stream>>>(
            grads->d_nabla_w[l - 1],
            net->d_zs[l - 1],
            net->d_activations[l - 1],
            prev_size_curr, curr_size
        );
        
        blocks = (curr_size + threads - 1) / threads;
        accumulate_biases_gradient_kernel<<<blocks, threads, 0, stream>>>(
            grads->d_nabla_b[l - 1],
            net->d_zs[l - 1],
            curr_size
        );
    }
}

void update_mini_batch(NeuralNetwork* net, float* h_inputs, int* h_targets, 
                       int batch_size, float eta) {
    static Gradients grads;
    static int initialized = 0;
    static float* d_inputs = NULL;
    static float* d_targets = NULL;
    static cudaStream_t stream;
    
    if (!initialized) {
        init_gradients(net, &grads);
        cudaStreamCreate(&stream);
        initialized = 1;
    }
    
    int input_size = net->sizes[0];
    int output_size = net->sizes[net->num_layers - 1];
    
    if (!d_inputs) {
        cudaMalloc(&d_inputs, batch_size * input_size * sizeof(float));
        cudaMalloc(&d_targets, batch_size * output_size * sizeof(float));
    }
    
    for (int l = 0; l < net->num_layers - 1; l++) {
        int input_sz = net->sizes[l];
        int output_sz = net->sizes[l + 1];
        int weight_size = output_sz * input_sz;
        
        zero_gradients_kernel<<<(weight_size + 255) / 256, 256>>>(
            grads.d_nabla_w[l], weight_size);
        zero_gradients_kernel<<<(output_sz + 255) / 256, 256>>>(
            grads.d_nabla_b[l], output_sz);
    }
    cudaDeviceSynchronize();
    
    float* h_onehot_targets = (float*)malloc(batch_size * output_size * sizeof(float));
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < output_size; j++) {
            h_onehot_targets[i * output_size + j] = (h_targets[i] == j) ? 1.0f : 0.0f;
        }
    }
    
    cudaMemcpy(d_inputs, h_inputs, batch_size * input_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, h_onehot_targets, batch_size * output_size * sizeof(float),
               cudaMemcpyHostToDevice);
    
    for (int i = 0; i < batch_size; i++) {
        backprop_single(net, 
                        &d_inputs[i * input_size],
                        &d_targets[i * output_size],
                        &grads, stream);
    }
    
    cudaStreamSynchronize(stream);
    
    int threads = 256;
    for (int l = 0; l < net->num_layers - 1; l++) {
        int input_sz = net->sizes[l];
        int output_sz = net->sizes[l + 1];
        int weight_size = output_sz * input_sz;
        
        int blocks = (weight_size + threads - 1) / threads;
        apply_gradients_kernel<<<blocks, threads>>>(
            net->d_weights[l], grads.d_nabla_w[l], weight_size, eta, batch_size);
        
        blocks = (output_sz + threads - 1) / threads;
        apply_gradients_kernel<<<blocks, threads>>>(
            net->d_biases[l], grads.d_nabla_b[l], output_sz, eta, batch_size);
    }
    cudaDeviceSynchronize();
    
    for (int l = 0; l < net->num_layers - 1; l++) {
        int input_sz = net->sizes[l];
        int output_sz = net->sizes[l + 1];
        int weight_size = output_sz * input_sz;
        
        cudaMemcpy(net->h_weights[l], net->d_weights[l],
                   weight_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(net->h_biases[l], net->d_biases[l],
                   output_sz * sizeof(float), cudaMemcpyDeviceToHost);
    }
    
    free(h_onehot_targets);
}

int evaluate(NeuralNetwork* net, float* test_inputs, int* test_labels, int test_size) {
    int correct = 0;
    for (int i = 0; i < test_size; i++) {
        feedforward(net, &test_inputs[i * net->sizes[0]]);
        
        float* h_output = (float*)malloc(net->sizes[net->num_layers - 1] * sizeof(float));
        cudaMemcpy(h_output, net->d_activations[net->num_layers - 1],
                   net->sizes[net->num_layers - 1] * sizeof(float),
                   cudaMemcpyDeviceToHost);
        
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

int predict(NeuralNetwork* net, float* image, int image_size) {
    if (image_size != net->sizes[0]) {
        printf("Ошибка: размер изображения %d, ожидается %d\n", 
               image_size, net->sizes[0]);
        return -1;
    }
    
    feedforward(net, image);
    
    float* h_output = (float*)malloc(net->sizes[net->num_layers - 1] * sizeof(float));
    cudaMemcpy(h_output, net->d_activations[net->num_layers - 1],
               net->sizes[net->num_layers - 1] * sizeof(float),
               cudaMemcpyDeviceToHost);
    
    int predicted = 0;
    float max_val = h_output[0];
    for (int j = 1; j < net->sizes[net->num_layers - 1]; j++) {
        if (h_output[j] > max_val) {
            max_val = h_output[j];
            predicted = j;
        }
    }
    
    printf("Предсказание: цифра %d (вероятность: %.2f%%)\n", 
           predicted, max_val * 100);
    
    printf("Вероятности:\n");
    for (int j = 0; j < net->sizes[net->num_layers - 1]; j++) {
        printf("  %d: %.2f%%\n", j, h_output[j] * 100);
    }
    
    free(h_output);
    return predicted;
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
    cudaFree(net->d_target);
    
    free(net->h_weights);
    free(net->h_biases);
    free(net->d_weights);
    free(net->d_biases);
    free(net->h_activations);
    free(net->d_activations);
    free(net->d_zs);
}

void save_model(NeuralNetwork* net, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        printf("Не удалось создать файл %s\n", filename);
        return;
    }
    
    fwrite(&net->num_layers, sizeof(int), 1, f);
    for (int l = 0; l < net->num_layers; l++) {
        fwrite(&net->sizes[l], sizeof(int), 1, f);
    }
    
    for (int l = 0; l < net->num_layers - 1; l++) {
        int input_size = net->sizes[l];
        int output_size = net->sizes[l + 1];
        int weight_size = output_size * input_size;
        
        fwrite(net->h_weights[l], sizeof(float), weight_size, f);
        fwrite(net->h_biases[l], sizeof(float), output_size, f);
    }
    
    fclose(f);
}

int load_model(NeuralNetwork* net, const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        printf("Файл %s не найден\n", filename);
        return -1;
    }
    
    int num_layers;
    fread(&num_layers, sizeof(int), 1, f);
    
    if (num_layers != net->num_layers) {
        printf("Несовместимая архитектура: %d vs %d\n", num_layers, net->num_layers);
        fclose(f);
        return -1;
    }
    
    int* loaded_sizes = (int*)malloc(num_layers * sizeof(int));
    for (int l = 0; l < num_layers; l++) {
        fread(&loaded_sizes[l], sizeof(int), 1, f);
        if (loaded_sizes[l] != net->sizes[l]) {
            printf("Несовместимый размер слоя %d: %d vs %d\n", 
                   l, loaded_sizes[l], net->sizes[l]);
            free(loaded_sizes);
            fclose(f);
            return -1;
        }
    }
    free(loaded_sizes);
    
    for (int l = 0; l < net->num_layers - 1; l++) {
        int input_size = net->sizes[l];
        int output_size = net->sizes[l + 1];
        int weight_size = output_size * input_size;
        
        fread(net->h_weights[l], sizeof(float), weight_size, f);
        fread(net->h_biases[l], sizeof(float), output_size, f);
        
        cudaMemcpy(net->d_weights[l], net->h_weights[l],
                   weight_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(net->d_biases[l], net->h_biases[l],
                   output_size * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    fclose(f);
    printf("Модель загружена из %s\n", filename);
    return 0;
}