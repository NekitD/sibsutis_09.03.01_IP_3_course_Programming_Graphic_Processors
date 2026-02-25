CC = nvcc
ARCH = -arch=sm_86
CFLAGS = $(ARCH) -O2 -lineinfo

TARGET = obj/lab5
SOURCE = src/lab5.cu

NCU = /opt/nvidia/nsight-compute/2024.3.2/ncu
NCU_METRICS_BASIC = --metrics gpu__time_duration.avg
NCU_METRICS_MEMORY = --metrics dram__bytes.sum.per_second,lts__t_bytes.sum.per_second,l1tex__t_bytes.sum.per_second
NCU_METRICS_LOCAL = --metrics l1tex__t_sectors_pipe_lsu_mem_local_op.sum,l1tex__t_bytes_pipe_lsu_mem_local_op.sum

all: $(TARGET)

$(TARGET): $(SOURCE)
	@ $(CC) $(CFLAGS) $< -o $@

run:
	@ ./$(TARGET)
pro-run:
	@ sudo $(NCU) $(NCU_METRICS_BASIC) ./$(TARGET)

pass:
	@ sudo $(NCU) $(NCU_METRICS_MEMORY) ./$(TARGET)

loc:
	@ sudo $(NCU) $(NCU_METRICS_LOCAL)  ./$(TARGET)

clean:
	@ rm -rf obj/*

.PHONY: all run pro-run pass loc clean