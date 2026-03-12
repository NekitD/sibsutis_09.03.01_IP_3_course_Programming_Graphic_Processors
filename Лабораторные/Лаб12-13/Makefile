CC = nvcc
ARCH = -arch=sm_86
CFLAGS = $(ARCH) -O2 -lineinfo -lcublas

SRCDIR = src
OBJDIR = obj
BINDIR = .

TARGETS = $(OBJDIR)/matrix_mul_cuda $(OBJDIR)/matrix_mul_cublas $(OBJDIR)/matrix_mul_wmma

.PHONY: all clean run

all: $(TARGETS)

$(OBJDIR)/matrix_mul_cuda: $(SRCDIR)/matrix_mul_cuda.cu
	@ $(CC) $(ARCH) -O2 $< -o $@

$(OBJDIR)/matrix_mul_cublas: $(SRCDIR)/matrix_mul_cublas.cu
	@ $(CC) $(ARCH) -O2 $< -o $@ -lcublas

$(OBJDIR)/matrix_mul_wmma: $(SRCDIR)/matrix_mul_wmma.cu
	@ $(CC) $(ARCH) -O2 $< -o $@

run: all
	@ $(OBJDIR)/matrix_mul_cuda 512 512 512
	@ $(OBJDIR)/matrix_mul_cuda 1024 1024 1024
	@ $(OBJDIR)/matrix_mul_cuda 2048 2048 2048
	
	@ $(OBJDIR)/matrix_mul_cublas 512 512 512
	@ $(OBJDIR)/matrix_mul_cublas 1024 1024 1024
	@ $(OBJDIR)/matrix_mul_cublas 2048 2048 2048
	
	@ $(OBJDIR)/matrix_mul_wmma 512 512 512
	@ $(OBJDIR)/matrix_mul_wmma 1024 1024 1024
	@ $(OBJDIR)/matrix_mul_wmma 2048 2048 2048

clean:
	@ rm -f $(TARGETS)

.PHONY: all clean run