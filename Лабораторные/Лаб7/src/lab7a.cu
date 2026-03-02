#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>


#define NX 256
#define NY 256
#define NZ 256

__constant__ float cR;           
__constant__ float cR2;         
__constant__ float cStepX;       
__constant__ float cStepY;       
__constant__ float cStepZ;     
__constant__ float cMinX;       
__constant__ float cMinY;        
__constant__ float cMinZ;       
__constant__ int cNx;           
__constant__ int cNy;           
__constant__ int cNz;            


cudaTextureObject_t texObject;

float* init_host_data(int nx, int ny, int nz, float* min_x, float* min_y, float* min_z,
                      float* step_x, float* step_y, float* step_z) {
    int total_size = nx * ny * nz;
    float* h_data = (float*)malloc(total_size * sizeof(float));
    
    *min_x = -5.0f;
    *min_y = -5.0f;
    *min_z = -5.0f;
    float max_x = 5.0f;
    float max_y = 5.0f;
    float max_z = 5.0f;
    
    *step_x = (max_x - *min_x) / (nx - 1);
    *step_y = (max_y - *min_y) / (ny - 1);
    *step_z = (max_z - *min_z) / (nz - 1);
    
    for (int i = 0; i < nx; i++) {
        float x = *min_x + i * (*step_x);
        for (int j = 0; j < ny; j++) {
            float y = *min_y + j * (*step_y);
            for (int k = 0; k < nz; k++) {
                float z = *min_z + k * (*step_z);
                int idx = i * ny * nz + j * nz + k;
                h_data[idx] = x*x + y*y + z*z;  
            }
        }
    }
    
    return h_data;
}

__global__ void sphere_integral_kernel(float* d_result, int n_theta, int n_phi, 
                                        cudaTextureObject_t texObj) {
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
        float x = cR * sin_theta * cosf(phi);
        float y = cR * sin_theta * sinf(phi);
        float z = cR * cosf(theta);
        float tex_x = (x - cMinX) / ((cNx - 1) * cStepX);
        float tex_y = (y - cMinY) / ((cNy - 1) * cStepY);
        float tex_z = (z - cMinZ) / ((cNz - 1) * cStepZ);
        
        if (tex_x >= 0.0f && tex_x <= 1.0f && 
            tex_y >= 0.0f && tex_y <= 1.0f && 
            tex_z >= 0.0f && tex_z <= 1.0f) {
            float f_val = tex3D<float>(texObj, tex_x, tex_y, tex_z);
            float dS = cR2 * sin_theta * (M_PI / n_theta) * (2.0f * M_PI / n_phi);
            local_sum += f_val * dS;
        }
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

cudaTextureObject_t createTextureObject(float* d_data, int nx, int ny, int nz) {
    cudaTextureObject_t texObj = 0;
    
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t cuArray;
    
    cudaExtent extent = make_cudaExtent(nx, ny, nz);
    cudaMalloc3DArray(&cuArray, &channelDesc, extent);
    
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr(d_data, nx * sizeof(float), nx, ny);
    copyParams.dstArray = cuArray;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    cudaMemcpy3D(&copyParams);
    
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1; 
    
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
    
    return texObj;
}

float compute_sphere_integral(float* h_data, float R, int nx, int ny, int nz,
                              float min_x, float min_y, float min_z,
                              float step_x, float step_y, float step_z,
                              int n_theta, int n_phi) {
    
    float *d_data, *d_result;
    float h_result = 0.0f;
    int total_size = nx * ny * nz;
    
    cudaMalloc(&d_data, total_size * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    
    cudaMemcpy(d_data, h_data, total_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));
    
    cudaMemcpyToSymbol(cR, &R, sizeof(float));
    float R2 = R * R;
    cudaMemcpyToSymbol(cR2, &R2, sizeof(float));
    cudaMemcpyToSymbol(cStepX, &step_x, sizeof(float));
    cudaMemcpyToSymbol(cStepY, &step_y, sizeof(float));
    cudaMemcpyToSymbol(cStepZ, &step_z, sizeof(float));
    cudaMemcpyToSymbol(cMinX, &min_x, sizeof(float));
    cudaMemcpyToSymbol(cMinY, &min_y, sizeof(float));
    cudaMemcpyToSymbol(cMinZ, &min_z, sizeof(float));
    cudaMemcpyToSymbol(cNx, &nx, sizeof(int));
    cudaMemcpyToSymbol(cNy, &ny, sizeof(int));
    cudaMemcpyToSymbol(cNz, &nz, sizeof(int));
    
    cudaTextureObject_t texObj = createTextureObject(d_data, nx, ny, nz);
    
    int threads_per_block = 256;
    int blocks = (n_theta * n_phi + threads_per_block - 1) / threads_per_block;
    blocks = min(blocks, 1024); 
    
    sphere_integral_kernel<<<blocks, threads_per_block>>>(d_result, n_theta, n_phi, texObj);
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaDestroyTextureObject(texObj);
    cudaFree(d_data);
    cudaFree(d_result);
    
    return h_result;
}

float analytical_solution(float R) {
    return 4.0f * M_PI * R * R * R * R;
}

int main() {
    int nx = NX, ny = NY, nz = NZ;
    float min_x, min_y, min_z;
    float step_x, step_y, step_z;
    
    printf("========================================================\n");
    printf("ВЫЧИСЛЕНИЕ ИНТЕГРАЛА ПО СФЕРЕ\n");
    printf("С ИСПОЛЬЗОВАНИЕМ ТЕКСТУРНОЙ И КОНСТАНТНОЙ ПАМЯТИ\n");
    printf("========================================================\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    
    float* h_data = init_host_data(nx, ny, nz, &min_x, &min_y, &min_z,
                                   &step_x, &step_y, &step_z);
    
    float R = 3.0f; 
    int n_theta = 1024; 
    int n_phi = 2048;  
    
    printf("\nПАРАМЕТРЫ ЗАДАЧИ:\n");
    printf("----------------------------------------\n");
    printf("Радиус сферы: %.2f\n", R);
    printf("Размер сетки: %d x %d x %d\n", nx, ny, nz);
    printf("Диапазон X: [%.2f, %.2f], шаг %.4f\n", min_x, min_x + (nx-1)*step_x, step_x);
    printf("Диапазон Y: [%.2f, %.2f], шаг %.4f\n", min_y, min_y + (ny-1)*step_y, step_y);
    printf("Диапазон Z: [%.2f, %.2f], шаг %.4f\n", min_z, min_z + (nz-1)*step_z, step_z);
    printf("Разбиение сферы: %d x %d точек\n", n_theta, n_phi);
    printf("Общее количество точек: %d\n", n_theta * n_phi);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    float result = compute_sphere_integral(h_data, R, nx, ny, nz,
                                          min_x, min_y, min_z,
                                          step_x, step_y, step_z,
                                          n_theta, n_phi);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    float analytical = analytical_solution(R);
    float error = fabs(result - analytical) / analytical * 100.0f;
    
    printf("\nРЕЗУЛЬТАТЫ:\n");
    printf("----------------------------------------\n");
    printf("Вычисленный интеграл: %.6f\n", result);
    printf("Аналитическое решение: %.6f\n", analytical);
    printf("Относительная ошибка: %.4f%%\n", error);
    printf("Время выполнения: %.3f мс\n", elapsed_time);

    free(h_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}