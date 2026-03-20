#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

typedef struct {
    float* images;
    int* labels;
    int count;
    int image_size;
} Dataset;

int load_binary_dataset(const char* images_path, const char* labels_path, 
                        Dataset* dataset, int expected_image_size) {
    FILE* f_images = fopen(images_path, "rb");
    FILE* f_labels = fopen(labels_path, "rb");
    
    if (!f_images || !f_labels) {
        printf("Не удалось открыть файлы: %s or %s\n", images_path, labels_path);
        return -1;
    }
    
    fseek(f_images, 0, SEEK_END);
    long file_size = ftell(f_images);
    fseek(f_images, 0, SEEK_SET);
    
    dataset->image_size = expected_image_size;
    dataset->count = file_size / (dataset->image_size * sizeof(float));
    
    dataset->images = (float*)malloc(file_size);
    dataset->labels = (int*)malloc(dataset->count * sizeof(int));
    
    size_t read = fread(dataset->images, sizeof(float), dataset->count * dataset->image_size, f_images);
    if (read != dataset->count * dataset->image_size) {
        printf("Не удалось прочитать изображения: прочитано %zu, ожидалось %d\n", 
               read, dataset->count * dataset->image_size);
        return -1;
    }
    
    read = fread(dataset->labels, sizeof(int), dataset->count, f_labels);
    if (read != dataset->count) {
        printf("Не удалось прочитать метки: прочитано %zu, ожидалось %d\n", read, dataset->count);
        return -1;
    }
    
    fclose(f_images);
    fclose(f_labels);
    
    printf("Загружено %d изображений, размером %d\n", dataset->count, dataset->image_size);
    return 0;
}

void free_dataset(Dataset* dataset) {
    if (dataset->images) free(dataset->images);
    if (dataset->labels) free(dataset->labels);
    dataset->images = NULL;
    dataset->labels = NULL;
    dataset->count = 0;
}