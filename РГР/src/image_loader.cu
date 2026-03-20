#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Загрузка и предобработка изображения для MNIST
float* load_and_preprocess_image(const char* filename, int* width, int* height) {
    // Загружаем изображение
    int w, h, channels;
    unsigned char* img = stbi_load(filename, &w, &h, &channels, 1); // 1 =灰度图
    
    if (!img) {
        printf("Не удалось загрузить изображение: %s\n", filename);
        return NULL;
    }
    
    *width = w;
    *height = h;
    
    // Создаём массив для предобработанного изображения (28x28)
    float* processed = (float*)malloc(28 * 28 * sizeof(float));
    
    // Масштабируем изображение до 28x28 (простой билинейный алгоритм)
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            int src_x = x * w / 28;
            int src_y = y * h / 28;
            if (src_x >= w) src_x = w - 1;
            if (src_y >= h) src_y = h - 1;
            
            float pixel = img[src_y * w + src_x] / 255.0f;
            // Инвертируем, так как MNIST использует белый фон (0) и чёрную цифру (1)
            processed[y * 28 + x] = 1.0f - pixel;
        }
    }
    
    stbi_image_free(img);
    
    return processed;
}

float* load_digit_image(const char* filename) {
    int w, h;
    float* img = load_and_preprocess_image(filename, &w, &h);
    
    if (!img) return NULL;
    
    printf("Загружено изображение: %s (%dx%d -> 28x28)\n", filename, w, h);
    
    // Выводим мини-картинку для отладки
    printf("Преобразованное изображение (10x10 preview):\n");
    for (int y = 0; y < 10; y++) {
        for (int x = 0; x < 10; x++) {
            int val = (int)(img[y * 28 + x] * 10);
            printf("%c", val > 5 ? '#' : (val > 2 ? '*' : '.'));
        }
        printf("\n");
    }
    
    return img;
}