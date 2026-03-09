import numpy as np
from ctypes import *
import sys
import time
import os

if 'linux' in sys.platform:
    cuda = CDLL('libcuda.so')
elif 'win' in sys.platform:
    cuda = CDLL('nvcuda.dll')

CUDA_SUCCESS = 0

c_uint_p = POINTER(c_uint)
c_int_p = POINTER(c_int)
c_void_p_p = POINTER(c_void_p)

cuInit = cuda.cuInit
cuInit.argtypes = [c_uint]
cuInit.restype = int

cuDeviceGet = cuda.cuDeviceGet
cuDeviceGet.argtypes = [POINTER(c_int), c_int]
cuDeviceGet.restype = int

cuDeviceGetName = cuda.cuDeviceGetName
cuDeviceGetName.argtypes = [c_char_p, c_int, c_int]
cuDeviceGetName.restype = int

cuCtxCreate = cuda.cuCtxCreate
cuCtxCreate.argtypes = [POINTER(c_void_p), c_uint, c_int]
cuCtxCreate.restype = int

cuCtxSynchronize = cuda.cuCtxSynchronize
cuCtxSynchronize.argtypes = []
cuCtxSynchronize.restype = int

cuCtxDestroy = cuda.cuCtxDestroy
cuCtxDestroy.argtypes = [c_void_p]
cuCtxDestroy.restype = int

cuModuleLoad = cuda.cuModuleLoad
cuModuleLoad.argtypes = [POINTER(c_void_p), c_char_p]
cuModuleLoad.restype = int

cuModuleUnload = cuda.cuModuleUnload
cuModuleUnload.argtypes = [c_void_p]
cuModuleUnload.restype = int

cuModuleGetFunction = cuda.cuModuleGetFunction
cuModuleGetFunction.argtypes = [POINTER(c_void_p), c_void_p, c_char_p]
cuModuleGetFunction.restype = int

cuMemAlloc = cuda.cuMemAlloc
cuMemAlloc.argtypes = [POINTER(c_void_p), c_size_t]
cuMemAlloc.restype = int

cuMemcpyHtoD = cuda.cuMemcpyHtoD
cuMemcpyHtoD.argtypes = [c_void_p, c_void_p, c_size_t]
cuMemcpyHtoD.restype = int

cuMemcpyDtoH = cuda.cuMemcpyDtoH
cuMemcpyDtoH.argtypes = [c_void_p, c_void_p, c_size_t]
cuMemcpyDtoH.restype = int

cuMemFree = cuda.cuMemFree
cuMemFree.argtypes = [c_void_p]
cuMemFree.restype = int

cuLaunchKernel = cuda.cuLaunchKernel
cuLaunchKernel.argtypes = [c_void_p, c_uint, c_uint, c_uint, 
                           c_uint, c_uint, c_uint, 
                           c_uint, c_void_p, POINTER(c_void_p), POINTER(c_void_p)]
cuLaunchKernel.restype = int

def check_cuda(err, msg):
    if err != CUDA_SUCCESS:
        print(f"CUDA Error {err} in {msg}")
        sys.exit(1)

def main():
    M, N, K = 1024, 1024, 1024
    if len(sys.argv) >= 4:
        M = int(sys.argv[1])
        N = int(sys.argv[2])
        K = int(sys.argv[3])
    
    print("========================================================")
    print("CUDA Driver API (Python) - Matrix Multiplication")
    print("========================================================")
    print(f"Размер матриц: M={M}, N={N}, K={K}")
    print(f"Память: A={M*K*4/1024/1024:.2f} MB, B={K*N*4/1024/1024:.2f} MB, C={M*N*4/1024/1024:.2f} MB")
    
    check_cuda(cuInit(0), "cuInit")
    
    device = c_int()
    check_cuda(cuDeviceGet(byref(device), 0), "cuDeviceGet")
    
    device_name = create_string_buffer(256)
    cuDeviceGetName(device_name, 256, device)
    print(f"Устройство: {device_name.value.decode()}")
    
    context = c_void_p()
    check_cuda(cuCtxCreate(byref(context), 0, device), "cuCtxCreate")
    
    module = c_void_p()
    
    cubin_path = b"matrix_mul_kernel.sm_86.cubin"
    if os.path.exists(cubin_path):
        #print(f"Loading cubin: {cubin_path}")
        check_cuda(cuModuleLoad(byref(module), cubin_path), "cuModuleLoad cubin")
    else:
        ptx_path = b"matrix_mul_kernel.ptx"
        #print(f"Cubin not found, loading PTX: {ptx_path}")
        check_cuda(cuModuleLoad(byref(module), ptx_path), "cuModuleLoad PTX")
    
    kernel = c_void_p()
    check_cuda(cuModuleGetFunction(byref(kernel), module, b"matrix_mul_kernel"), "cuModuleGetFunction")
    
    h_A = np.ones((M, K), dtype=np.float32)
    h_B = np.ones((K, N), dtype=np.float32)
    h_C = np.zeros((M, N), dtype=np.float32)
    
    d_A = c_void_p()
    d_B = c_void_p()
    d_C = c_void_p()
    check_cuda(cuMemAlloc(byref(d_A), M * K * 4), "cuMemAlloc A")
    check_cuda(cuMemAlloc(byref(d_B), K * N * 4), "cuMemAlloc B")
    check_cuda(cuMemAlloc(byref(d_C), M * N * 4), "cuMemAlloc C")
    
    check_cuda(cuMemcpyHtoD(d_A, h_A.ctypes.data_as(c_void_p), M * K * 4), "cuMemcpyHtoD A")
    check_cuda(cuMemcpyHtoD(d_B, h_B.ctypes.data_as(c_void_p), K * N * 4), "cuMemcpyHtoD B")
    
    block_size = 16
    grid_x = (N + block_size - 1) // block_size
    grid_y = (M + block_size - 1) // block_size
    
    #print(f"\nKernel launch configuration:")
    #print(f"  Grid: {grid_x} x {grid_y} x 1")
    #print(f"  Block: {block_size} x {block_size} x 1")
    
    m_val = c_int(M)
    n_val = c_int(N)
    k_val = c_int(K)
    
    kernel_args = [
        pointer(d_A),
        pointer(d_B),
        pointer(d_C),
        pointer(m_val),
        pointer(n_val),
        pointer(k_val)
    ]
    
    args_array = (c_void_p * len(kernel_args))()
    for i, arg in enumerate(kernel_args):
        args_array[i] = cast(arg, c_void_p)
    
    check_cuda(cuLaunchKernel(kernel, grid_x, grid_y, 1,
                              block_size, block_size, 1,
                              0, None, args_array, None), "cuLaunchKernel warmup")
    check_cuda(cuCtxSynchronize(), "cuCtxSynchronize warmup")
    
    start = time.time()
    check_cuda(cuLaunchKernel(kernel, grid_x, grid_y, 1,
                              block_size, block_size, 1,
                              0, None, args_array, None), "cuLaunchKernel")
    check_cuda(cuCtxSynchronize(), "cuCtxSynchronize")
    end = time.time()
    
    check_cuda(cuMemcpyDtoH(h_C.ctypes.data_as(c_void_p), d_C, M * N * 4), "cuMemcpyDtoH")
    
    elapsed = end - start
    gflops = 2.0 * M * N * K / elapsed / 1e9
    
    print(f"  Время: {elapsed:.6f} сек")
    print(f"  Производительность: {gflops:.2f} GFLOPS")
    #print(f"  Verification: C[0][0] = {h_C[0,0]:.2f} (expected {K:.2f})")
    
    cuMemFree(d_A)
    cuMemFree(d_B)
    cuMemFree(d_C)
    cuModuleUnload(module)
    cuCtxDestroy(context)

if __name__ == "__main__":
    main()