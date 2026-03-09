import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time

kernel_code = """
__global__ void matrix_mul(float* A, float* B, float* C,
                           int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}
"""

def main():
    M, N, K = 1024, 1024, 1024
    
    h_A = np.ones((M, K), dtype=np.float32)
    h_B = np.ones((K, N), dtype=np.float32)
    h_C = np.zeros((M, N), dtype=np.float32)
    
    mod = SourceModule(kernel_code)
    matrix_mul = mod.get_function("matrix_mul")
    
    d_A = cuda.mem_alloc(h_A.nbytes)
    d_B = cuda.mem_alloc(h_B.nbytes)
    d_C = cuda.mem_alloc(h_C.nbytes)
    
    cuda.memcpy_htod(d_A, h_A)
    cuda.memcpy_htod(d_B, h_B)
    
    block_size = 16
    grid_x = (N + block_size - 1) // block_size
    grid_y = (M + block_size - 1) // block_size
    
    matrix_mul(d_A, d_B, d_C,
               np.int32(M), np.int32(N), np.int32(K),
               block=(block_size, block_size, 1),
               grid=(grid_x, grid_y))
    cuda.Context.synchronize()
    
    start = time.time()
    matrix_mul(d_A, d_B, d_C,
               np.int32(M), np.int32(N), np.int32(K),
               block=(block_size, block_size, 1),
               grid=(grid_x, grid_y))
    cuda.Context.synchronize()
    end = time.time()
    
    cuda.memcpy_dtoh(h_C, d_C)
    
    print("PyCUDA:")
    print(f"  Размер матриц: {M} x {N}")
    print(f"  Время: {end - start:.6f} сек")
    print(f"  Производительность: {2.0 * M * N * K / (end - start) / 1e9:.2f} GFLOPS")
    #print(f"  Проверка: C[0,0] = {h_C[0,0]:.1f} (ожидается {K:.1f})")

if __name__ == "__main__":
    main()