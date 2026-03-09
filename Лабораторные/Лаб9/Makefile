CC = nvcc
ARCH = -arch=sm_86
CFLAGS = $(ARCH) -O2

SRCDIR = src
OBJDIR = obj
BINDIR = .

RUNTIME_SRC = $(SRCDIR)/matrix_mul_runtime.cu
DRIVER_SRC = $(SRCDIR)/matrix_mul_driver.cpp
KERNEL_SRC = $(SRCDIR)/matrix_mul_kernel.cu

RUNTIME_TARGET = $(BINDIR)/matrix_mul_runtime
DRIVER_TARGET = $(BINDIR)/matrix_mul_driver

KERNEL_CUBIN = $(OBJDIR)/matrix_mul_kernel.sm_86.cubin
KERNEL_PTX = $(OBJDIR)/matrix_mul_kernel.ptx

VENV = $(CURDIR)/cuda_env
PYTHON = $(VENV)/bin/python3

DRIVER_PY = $(SRCDIR)/matrix_mul_driver.py
PYCUDA_PY = $(SRCDIR)/matrix_mul_pycuda.py
NUMBA_PY = $(SRCDIR)/matrix_mul_numba.py

M = 1024
N = 1024
K = 1024

.PHONY: all clean runtime driver cubin python run-cpp run-python run benchmark setup

setup:
	@mkdir -p $(OBJDIR)

all: setup cubin runtime driver

cubin: $(KERNEL_CUBIN) $(KERNEL_PTX)

$(KERNEL_CUBIN): $(KERNEL_SRC) | setup
	@ $(CC) $(ARCH) -cubin $(KERNEL_SRC) -o $(KERNEL_CUBIN)

$(KERNEL_PTX): $(KERNEL_SRC) | setup
	@ $(CC) $(ARCH) -ptx $(KERNEL_SRC) -o $(KERNEL_PTX)

runtime: $(RUNTIME_TARGET)

$(RUNTIME_TARGET): $(RUNTIME_SRC) | setup
	@ $(CC) $(CFLAGS) $(RUNTIME_SRC) -o $(RUNTIME_TARGET)

driver: $(DRIVER_TARGET)

$(DRIVER_TARGET): $(DRIVER_SRC) | cubin
	@ $(CC) $(CFLAGS) $(DRIVER_SRC) -o $(DRIVER_TARGET) -lcuda

# ==================== ЗАПУСК (с sudo) ====================


run-cpp: runtime driver
	@echo "---------------"
	@echo "C++"
	@echo "---------------"
	@echo "\n--- CUDA Runtime API (C++) ---"
	@CUDA_VISIBLE_DEVICES=0 ./$(RUNTIME_TARGET) $(M) $(N) $(K)
	@echo "\n--- CUDA Driver API (C++) ---"
	@CUDA_VISIBLE_DEVICES=0 ./$(DRIVER_TARGET) $(M) $(N) $(K)

link-python: cubin
	@ln -sf ../$(KERNEL_CUBIN) $(SRCDIR)/matrix_mul_kernel.sm_86.cubin
	@ln -sf ../$(KERNEL_PTX) $(SRCDIR)/matrix_mul_kernel.ptx

run-python: link-python
	@echo "---------------"
	@echo "PYTHON"
	@echo "---------------"
	@echo "\n--- CUDA Driver API (Python) ---"
	@cd $(SRCDIR) && CUDA_VISIBLE_DEVICES=0 $(PYTHON) matrix_mul_driver.py $(M) $(N) $(K)
	@echo "\n--- PyCUDA ---"
	@cd $(SRCDIR) && CUDA_VISIBLE_DEVICES=0 $(PYTHON) matrix_mul_pycuda.py $(M) $(N) $(K)
	@echo "\n--- Numba CUDA ---"
	@cd $(SRCDIR) && CUDA_VISIBLE_DEVICES=0 $(PYTHON) matrix_mul_numba.py $(M) $(N) $(K)

run: run-cpp run-python

# ==================== ТЕСТИРОВАНИЕ ====================
benchmark: runtime driver
	@echo "========================================================="
	@echo "СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ (РАЗНЫЕ РАЗМЕРЫ)"
	@echo "========================================================="
	@for size in 256 512 1024 2048; do \
		echo "\n--- Размер матрицы: $$size x $$size ---"; \
		echo "CUDA Runtime API:"; \
		sudo ./$(RUNTIME_TARGET) $$size $$size $$size; \
		echo "CUDA Driver API:"; \
		sudo ./$(DRIVER_TARGET) $$size $$size $$size; \
	done

clean:
	@rm -rf $(OBJDIR) $(RUNTIME_TARGET) $(DRIVER_TARGET) *.pyc __pycache__
