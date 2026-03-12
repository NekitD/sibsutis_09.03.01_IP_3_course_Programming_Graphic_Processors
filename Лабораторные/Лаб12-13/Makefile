CC = nvcc
ARCH = -arch=sm_86
CFLAGS = $(ARCH) -O2 -lineinfo -lcublas

SRCDIR = src
BINDIR = .

TARGETS = matrix_mul_cuda matrix_mul_cublas matrix_mul_wmma

.PHONY: all clean run

all: $(TARGETS)

matrix_mul_cuda: $(SRCDIR)/matrix_mul_cuda.cu
	$(CC) $(ARCH) -O2 $< -o $@

matrix_mul_cublas: $(SRCDIR)/matrix_mul_cublas.cu
	$(CC) $(ARCH) -O2 $< -o $@ -lcublas

matrix_mul_wmma: $(SRCDIR)/matrix_mul_wmma.cu
	$(CC) $(ARCH) -O2 $< -o $@

run: all
	./matrix_mul_cuda 512 512 512
	./matrix_mul_cuda 1024 1024 1024
	./matrix_mul_cuda 2048 2048 2048
	
	./matrix_mul_cublas 512 512 512
	./matrix_mul_cublas 1024 1024 1024
	./matrix_mul_cublas 2048 2048 2048
	
	./matrix_mul_wmma 512 512 512
	./matrix_mul_wmma 1024 1024 1024
	./matrix_mul_wmma 2048 2048 2048

clean:
	rm -f $(TARGETS)

.PHONY: all clean run