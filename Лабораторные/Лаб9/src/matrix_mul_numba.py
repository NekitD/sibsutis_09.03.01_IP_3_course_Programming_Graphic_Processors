import numpy as np
from numba import cuda
import time

@cuda.jit
def matrix_mul_kernel(A, B, C):
    row, col = cuda.grid(2)
    M, K = A.shape
    N = B.shape[1]
    
    if row < M and col < N:
        sum_val = 0.0
        for i in range(K):
            sum_val += A[row, i] * B[i, col]
        C[row, col] = sum_val

def main():
    
    M, N, K = 1024, 1024, 1024
    
    h_A = np.ones((M, K), dtype=np.float32)
    h_B = np.ones((K, N), dtype=np.float32)
    h_C = np.zeros((M, N), dtype=np.float32)
    
    d_A = cuda.to_device(h_A)
    d_B = cuda.to_device(h_B)
    d_C = cuda.to_device(h_C)
    
    threads_per_block = (16, 16)
    blocks_per_grid_x = (N + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (M + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    matrix_mul_kernel[blocks_per_grid, threads_per_block](d_A, d_B, d_C)
    cuda.synchronize()
    
    start = time.time()
    matrix_mul_kernel[blocks_per_grid, threads_per_block](d_A, d_B, d_C)
    cuda.synchronize()
    end = time.time()
    
    d_C.copy_to_host(h_C)
    
    print("Numba CUDA:")
    print(f"  Размер матриц: {M} x {N}")
    print(f"  Время: {end - start:.6f} сек")
    #print(f"  Производительность: {2.0 * M * N * K / (end - start) / 1e9:.2f} GFLOPS")
    print(f"  Проверка: C[0,0] = {h_C[0,0]:.1f} (ожидалось {K:.1f})")

if __name__ == "__main__":
    main()