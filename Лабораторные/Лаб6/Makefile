CC = nvcc
ARCH = -arch=sm_86
CFLAGS = $(ARCH) -O2 -lineinfo

TARGET = obj/lab6
SOURCE = src/lab6.cu

NCU = /opt/nvidia/nsight-compute/2024.3.2/ncu

NCU_METRICS_BASIC = --metrics gpu__time_duration.avg
NCU_METRICS_SHARED = --metrics smsp__sass_average_data_bytes_per_wavefront_mem_shared.pct
NCU_METRICS_BANK = --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
NCU_METRICS_SHARED_LD = --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum.per_second
NCU_METRICS_ALL = --metrics smsp__sass_average_data_bytes_per_wavefront_mem_shared.pct,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum.per_second,gpu__time_duration.avg

all: $(TARGET)

$(TARGET): $(SOURCE)
	@ mkdir -p obj
	@ $(CC) $(CFLAGS) $< -o $@

run:
	@ echo "=== Транспонирование матрицы 256x32 ==="
	@ ./$(TARGET) 256 32
	@ echo "\n=== Транспонирование матрицы 512x32 ==="
	@ ./$(TARGET) 512 32
	@ echo "\n=== Транспонирование матрицы 1024x32 ==="
	@ ./$(TARGET) 1024 32

pro-run:
	@ sudo $(NCU) $(NCU_METRICS_BASIC) ./$(TARGET) 256 32

pro-shared:
	@ sudo $(NCU) $(NCU_METRICS_SHARED) ./$(TARGET) 256 32

pro-bank:
	@ sudo $(NCU) $(NCU_METRICS_BANK) ./$(TARGET) 256 32

pro-shared-ld:
	@ sudo $(NCU) $(NCU_METRICS_SHARED_LD) ./$(TARGET) 256 32

pro-all:
	@ sudo $(NCU) $(NCU_METRICS_ALL) ./$(TARGET) 256 32

compare:
	@ echo "=== СРАВНЕНИЕ ВАРИАНТОВ ТРАНСПОНИРОВАНИЯ (матрица 256x32) ==="
	@ echo "\n1. gTranspose1 (без разделяемой памяти):"
	@ sudo $(NCU) --metrics gpu__time_duration.avg ./$(TARGET) 256 32 | grep -A 2 "gTranspose1"
	@ echo "\n2. gTransposeSM (с разделяемой памятью, конфликты):"
	@ sudo $(NCU) --metrics gpu__time_duration.avg ./$(TARGET) 256 32 | grep -A 2 "gTransposeSM"
	@ echo "\n3. gTransposeSM_WC (с разделяемой памятью, без конфликтов):"
	@ sudo $(NCU) --metrics gpu__time_duration.avg ./$(TARGET) 256 32 | grep -A 2 "gTransposeSM_WC"
	@ echo "\n4. gInit (инициализация):"
	@ sudo $(NCU) --metrics gpu__time_duration.avg ./$(TARGET) 256 32 | grep -A 2 "gInit"

shared-efficiency:
	@ echo "=== ЭФФЕКТИВНОСТЬ РАЗДЕЛЯЕМОЙ ПАМЯТИ ==="
	@ echo "\n1. gTransposeSM (с разделяемой памятью, конфликты):"
	@ sudo $(NCU) --metrics smsp__sass_average_data_bytes_per_wavefront_mem_shared.pct ./$(TARGET) 256 32 | grep -A 3 "gTransposeSM" | grep "Metric Value"
	@ echo "\n2. gTransposeSM_WC (с разделяемой памятью, без конфликтов):"
	@ sudo $(NCU) --metrics smsp__sass_average_data_bytes_per_wavefront_mem_shared.pct ./$(TARGET) 256 32 | grep -A 3 "gTransposeSM_WC" | grep "Metric Value"

bank-conflicts:
	@ echo "=== КОНФЛИКТЫ БАНКОВ ==="
	@ echo "\n1. gTransposeSM (с разделяемой памятью, конфликты):"
	@ sudo $(NCU) --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./$(TARGET) 256 32 | grep -A 3 "gTransposeSM" | grep "Metric Value"
	@ echo "\n2. gTransposeSM_WC (с разделяемой памятью, без конфликтов):"
	@ sudo $(NCU) --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./$(TARGET) 256 32 | grep -A 3 "gTransposeSM_WC" | grep "Metric Value"

test-sizes:
	@ for size in 128 256 512 1024 2048; do \
		echo "\n=== РАЗМЕР МАТРИЦЫ: $$size x 32 ==="; \
		sudo $(NCU) --metrics gpu__time_duration.avg ./$(TARGET) $$size 32 | grep -E "gTranspose1|gTransposeSM|gTransposeSM_WC" | grep -A 1 "Metric Value"; \
	done

clean:
	@ rm -rf obj/*

.PHONY: all run pro-run pro-shared pro-bank pro-shared-ld pro-all compare shared-efficiency bank-conflicts test-sizes clean