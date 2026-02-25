CC = nvcc
ARCH = -arch=sm_86
CFLAGS = $(ARCH) -O2 -lineinfo

TARGET = obj/lab5
SOURCE = src/lab5.cu

NCU = /opt/nvidia/nsight-compute/2024.3.2/ncu
NCU_METRICS_BASIC = --metrics gpu__time_duration.avg
NCU_METRICS_MEMORY = --metrics dram__bytes.sum.per_second,lts__t_bytes.sum.per_second,l1tex__t_bytes.sum.per_second
NCU_METRICS_LOCAL = --metrics sm__sass_data_bytes_mem_local.sum,sm__sass_data_bytes_mem_local_op_ld.sum,sm__sass_data_bytes_mem_local_op_st.sum,smsp__sass_inst_executed_op_local.sum

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