#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

float* load_and_preprocess_image(const char* filename, int* width, int* height) {
    int w, h, channels;
    unsigned char* img = stbi_load(filename, &w, &h, &channels, 1);
    
    if (!img) {
        printf("Не удалось загрузить изображение: %s\n", filename);
        return NULL;
    }
    
    *width = w;
    *height = h;
    
    float* processed = (float*)malloc(28 * 28 * sizeof(float));
    
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            int src_x = x * w / 28;
            int src_y = y * h / 28;
            if (src_x >= w) src_x = w - 1;
            if (src_y >= h) src_y = h - 1;
            
            float pixel = img[src_y * w + src_x] / 255.0f;
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
    
    return img;
}