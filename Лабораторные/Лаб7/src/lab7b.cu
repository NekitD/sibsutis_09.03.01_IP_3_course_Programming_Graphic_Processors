#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define NX 256
#define NY 256
#define NZ 256

typedef struct {
    float min_x, min_y, min_z;
    float step_x, step_y, step_z;
    int nx, ny, nz;
    float R;      
    float R2;     
} GridParams;

float* init_host_data(int nx, int ny, int nz, GridParams* params) {
    int total_size = nx * ny * nz;
    float* h_data = (float*)malloc(total_size * sizeof(float));
    
    params->min_x = -5.0f;
    params->min_y = -5.0f;
    params->min_z = -5.0f;
    float max_x = 5.0f;
    float max_y = 5.0f;
    float max_z = 5.0f;
    
    params->step_x = (max_x - params->min_x) / (nx - 1);
    params->step_y = (max_y - params->min_y) / (ny - 1);
    params->step_z = (max_z - params->min_z) / (nz - 1);
    
    params->nx = nx;
    params->ny = ny;
    params->nz = nz;
    
    for (int i = 0; i < nx; i++) {
        float x = params->min_x + i * params->step_x;
        for (int j = 0; j < ny; j++) {
            float y = params->min_y + j * params->step_y;
            for (int k = 0; k < nz; k++) {
                float z = params->min_z + k * params->step_z;
                int idx = i * ny * nz + j * nz + k;
                h_data[idx] = x*x + y*y + z*z;  
            }
        }
    }
    
    return h_data;
}


__device__ float nearest_neighbor_interp(float x, float y, float z, 
                                         const float* data, const GridParams params) {
    float nx_float = (x - params.min_x) / params.step_x;
    float ny_float = (y - params.min_y) / params.step_y;
    float nz_float = (z - params.min_z) / params.step_z;
    
    int ix = (int)roundf(nx_float);
    int iy = (int)roundf(ny_float);
    int iz = (int)roundf(nz_float);
    
    if (ix < 0 || ix >= params.nx || iy < 0 || iy >= params.ny || iz < 0 || iz >= params.nz) {
        return 0.0f;
    }
    
    int idx = ix * params.ny * params.nz + iy * params.nz + iz;
    return data[idx];
}

__global__ void sphere_integral_nearest_kernel(float* d_result, int n_theta, int n_phi,
                                               const float* d_data, GridParams params) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    __shared__ float shared_sum[256];
    float local_sum = 0.0f;
    
    for (int i = idx; i < n_theta * n_phi; i += total_threads) {
        int theta_idx = i / n_phi;
        int phi_idx = i % n_phi;
        
        float theta = (theta_idx + 0.5f) * M_PI / n_theta;
        float phi = (phi_idx + 0.5f) * 2.0f * M_PI / n_phi;
        
        float sin_theta = sinf(theta);
        float x = params.R * sin_theta * cosf(phi);
        float y = params.R * sin_theta * sinf(phi);
        float z = params.R * cosf(theta);
        
        float f_val = nearest_neighbor_interp(x, y, z, d_data, params);
        
        float dS = params.R2 * sin_theta * (M_PI / n_theta) * (2.0f * M_PI / n_phi);
        local_sum += f_val * dS;
    }
    
    shared_sum[threadIdx.x] = local_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(d_result, shared_sum[0]);
    }
}


__device__ float trilinear_interp(float x, float y, float z,
                                   const float* data, const GridParams params) {

    float fx = (x - params.min_x) / params.step_x;
    float fy = (y - params.min_y) / params.step_y;
    float fz = (z - params.min_z) / params.step_z;
    
    int ix0 = (int)floorf(fx);
    int iy0 = (int)floorf(fy);
    int iz0 = (int)floorf(fz);
    
    if (ix0 < 0 || ix0 >= params.nx - 1 || 
        iy0 < 0 || iy0 >= params.ny - 1 || 
        iz0 < 0 || iz0 >= params.nz - 1) {
        return 0.0f;
    }
    
    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;
    int iz1 = iz0 + 1;
    
    float wx = fx - ix0;
    float wy = fy - iy0;
    float wz = fz - iz0;
    
    int idx000 = ix0 * params.ny * params.nz + iy0 * params.nz + iz0;
    int idx001 = ix0 * params.ny * params.nz + iy0 * params.nz + iz1;
    int idx010 = ix0 * params.ny * params.nz + iy1 * params.nz + iz0;
    int idx011 = ix0 * params.ny * params.nz + iy1 * params.nz + iz1;
    int idx100 = ix1 * params.ny * params.nz + iy0 * params.nz + iz0;
    int idx101 = ix1 * params.ny * params.nz + iy0 * params.nz + iz1;
    int idx110 = ix1 * params.ny * params.nz + iy1 * params.nz + iz0;
    int idx111 = ix1 * params.ny * params.nz + iy1 * params.nz + iz1;
    
    float v000 = data[idx000];
    float v001 = data[idx001];
    float v010 = data[idx010];
    float v011 = data[idx011];
    float v100 = data[idx100];
    float v101 = data[idx101];
    float v110 = data[idx110];
    float v111 = data[idx111];
    
    float v00 = v000 * (1.0f - wx) + v100 * wx;
    float v01 = v001 * (1.0f - wx) + v101 * wx;
    float v10 = v010 * (1.0f - wx) + v110 * wx;
    float v11 = v011 * (1.0f - wx) + v111 * wx;
    
    float v0 = v00 * (1.0f - wy) + v10 * wy;
    float v1 = v01 * (1.0f - wy) + v11 * wy;
    
    return v0 * (1.0f - wz) + v1 * wz;
}

__global__ void sphere_integral_linear_kernel(float* d_result, int n_theta, int n_phi,
                                               const float* d_data, GridParams params) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    __shared__ float shared_sum[256];
    float local_sum = 0.0f;
    
    for (int i = idx; i < n_theta * n_phi; i += total_threads) {
        int theta_idx = i / n_phi;
        int phi_idx = i % n_phi;
        
        float theta = (theta_idx + 0.5f) * M_PI / n_theta;
        float phi = (phi_idx + 0.5f) * 2.0f * M_PI / n_phi;
        
        float sin_theta = sinf(theta);
        float x = params.R * sin_theta * cosf(phi);
        float y = params.R * sin_theta * sinf(phi);
        float z = params.R * cosf(theta);
        
        float f_val = trilinear_interp(x, y, z, d_data, params);
        
        float dS = params.R2 * sin_theta * (M_PI / n_theta) * (2.0f * M_PI / n_phi);
        local_sum += f_val * dS;
    }
    
    shared_sum[threadIdx.x] = local_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(d_result, shared_sum[0]);
    }
}

float compute_integral_nearest(float* h_data, GridParams params, int n_theta, int n_phi) {
    float *d_data, *d_result;
    int total_size = params.nx * params.ny * params.nz;
    
    cudaMalloc(&d_data, total_size * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    
    cudaMemcpy(d_data, h_data, total_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));
    
    int threads_per_block = 256;
    int blocks = (n_theta * n_phi + threads_per_block - 1) / threads_per_block;
    blocks = min(blocks, 1024);
    
    sphere_integral_nearest_kernel<<<blocks, threads_per_block>>>(
        d_result, n_theta, n_phi, d_data, params);
    
    cudaDeviceSynchronize();
    
    float h_result;
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_data);
    cudaFree(d_result);
    
    return h_result;
}

float compute_integral_linear(float* h_data, GridParams params, int n_theta, int n_phi) {
    float *d_data, *d_result;
    int total_size = params.nx * params.ny * params.nz;
    
    cudaMalloc(&d_data, total_size * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    
    cudaMemcpy(d_data, h_data, total_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));
    
    int threads_per_block = 256;
    int blocks = (n_theta * n_phi + threads_per_block - 1) / threads_per_block;
    blocks = min(blocks, 1024);
    
    sphere_integral_linear_kernel<<<blocks, threads_per_block>>>(
        d_result, n_theta, n_phi, d_data, params);
    
    cudaDeviceSynchronize();
    
    float h_result;
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_data);
    cudaFree(d_result);
    
    return h_result;
}

float analytical_solution(float R) {
    return 4.0f * M_PI * R * R * R * R;
}

float cpu_integral_linear(float* h_data, GridParams params, int n_theta, int n_phi) {
    double sum = 0.0;
    
    for (int t = 0; t < n_theta; t++) {
        for (int p = 0; p < n_phi; p++) {
            float theta = (t + 0.5f) * M_PI / n_theta;
            float phi = (p + 0.5f) * 2.0f * M_PI / n_phi;
            
            float sin_theta = sinf(theta);
            float x = params.R * sin_theta * cosf(phi);
            float y = params.R * sin_theta * sinf(phi);
            float z = params.R * cosf(theta);
        
            float fx = (x - params.min_x) / params.step_x;
            float fy = (y - params.min_y) / params.step_y;
            float fz = (z - params.min_z) / params.step_z;
            
            int ix0 = (int)floorf(fx);
            int iy0 = (int)floorf(fy);
            int iz0 = (int)floorf(fz);
            
            if (ix0 >= 0 && ix0 < params.nx - 1 && 
                iy0 >= 0 && iy0 < params.ny - 1 && 
                iz0 >= 0 && iz0 < params.nz - 1) {
                
                int ix1 = ix0 + 1;
                int iy1 = iy0 + 1;
                int iz1 = iz0 + 1;
                
                float wx = fx - ix0;
                float wy = fy - iy0;
                float wz = fz - iz0;
                
                // Индексы
                int idx000 = ix0 * params.ny * params.nz + iy0 * params.nz + iz0;
                int idx001 = ix0 * params.ny * params.nz + iy0 * params.nz + iz1;
                int idx010 = ix0 * params.ny * params.nz + iy1 * params.nz + iz0;
                int idx011 = ix0 * params.ny * params.nz + iy1 * params.nz + iz1;
                int idx100 = ix1 * params.ny * params.nz + iy0 * params.nz + iz0;
                int idx101 = ix1 * params.ny * params.nz + iy0 * params.nz + iz1;
                int idx110 = ix1 * params.ny * params.nz + iy1 * params.nz + iz0;
                int idx111 = ix1 * params.ny * params.nz + iy1 * params.nz + iz1;
                
                float v000 = h_data[idx000];
                float v001 = h_data[idx001];
                float v010 = h_data[idx010];
                float v011 = h_data[idx011];
                float v100 = h_data[idx100];
                float v101 = h_data[idx101];
                float v110 = h_data[idx110];
                float v111 = h_data[idx111];
                
                float v00 = v000 * (1.0f - wx) + v100 * wx;
                float v01 = v001 * (1.0f - wx) + v101 * wx;
                float v10 = v010 * (1.0f - wx) + v110 * wx;
                float v11 = v011 * (1.0f - wx) + v111 * wx;
                
                float v0 = v00 * (1.0f - wy) + v10 * wy;
                float v1 = v01 * (1.0f - wy) + v11 * wy;
                
                float f_val = v0 * (1.0f - wz) + v1 * wz;
                
                float dS = params.R2 * sin_theta * (M_PI / n_theta) * (2.0f * M_PI / n_phi);
                sum += f_val * dS;
            }
        }
    }
    
    return (float)sum;
}

int main() {
    GridParams params;
    params.R = 3.0f;
    params.R2 = params.R * params.R;
    
    int n_theta = 1024;   
    int n_phi = 2048;
    
    printf("========================================================\n");
    printf("ВЫЧИСЛЕНИЕ ИНТЕГРАЛА ПО СФЕРЕ\n");
    printf("БЕЗ ИСПОЛЬЗОВАНИЯ ТЕКСТУРНОЙ И КОНСТАНТНОЙ ПАМЯТИ\n");
    printf("========================================================\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    
    float* h_data = init_host_data(NX, NY, NZ, &params);
    
    printf("\nПАРАМЕТРЫ ЗАДАЧИ:\n");
    printf("----------------------------------------\n");
    printf("Радиус сферы: %.2f\n", params.R);
    printf("Сетка: %d x %d x %d\n", params.nx, params.ny, params.nz);
    printf("Разбиение сферы: %d x %d\n", n_theta, n_phi);
    printf("Всего точек: %d\n", n_theta * n_phi);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    float result_nearest = compute_integral_nearest(h_data, params, n_theta, n_phi);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_nearest;
    cudaEventElapsedTime(&time_nearest, start, stop);
    
    cudaEventRecord(start);
    float result_linear = compute_integral_linear(h_data, params, n_theta, n_phi);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_linear;
    cudaEventElapsedTime(&time_linear, start, stop);

    float analytical = analytical_solution(params.R);
    
    float cpu_result = 0.0f;
    if (NX <= 64 && NY <= 64 && NZ <= 64) {
        cpu_result = cpu_integral_linear(h_data, params, n_theta/4, n_phi/4);
    }
    
    printf("\nРЕЗУЛЬТАТЫ:\n");
    printf("----------------------------------------\n");
    printf("Метод                 | Результат  | Ошибка  | Время (мс)\n");
    printf("----------------------|------------|---------|-----------\n");
    printf("Ступенчатая интерп.   | %.2f | %.4f%% | %.3f\n", 
           result_nearest, 
           fabs(result_nearest - analytical) / analytical * 100.0f,
           time_nearest);
    printf("Линейная интерп.      | %.2f | %.4f%% | %.3f\n", 
           result_linear,
           fabs(result_linear - analytical) / analytical * 100.0f,
           time_linear);
    printf("Аналитическое решение | %.2f | 0.0000%% | -\n", analytical);
    if (cpu_result != 0.0f) {
        printf("CPU (линейная)        | %.2f | %.4f%% | -\n", 
               cpu_result,
               fabs(cpu_result - analytical) / analytical * 100.0f);
    }
    
    free(h_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}