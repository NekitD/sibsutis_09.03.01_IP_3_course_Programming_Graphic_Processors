CC = nvcc
ARCH = -arch=sm_86
CFLAGS = $(ARCH) -O2 -lineinfo

SRCDIR = src
OBJDIR = obj
BINDIR = .

# Исполняемые файлы
DOT_RAW = $(BINDIR)/dot_product_raw
DOT_THRUST = $(BINDIR)/dot_product_thrust
TRANS_RAW = $(BINDIR)/transpose_raw
TRANS_THRUST = $(BINDIR)/transpose_thrust

.PHONY: all clean dot transpose run benchmark

all: dot transpose

# Скалярное произведение
dot: $(DOT_RAW) $(DOT_THRUST)

$(DOT_RAW): $(SRCDIR)/dot_product_raw.cu
	$(CC) $(CFLAGS) $< -o $@

$(DOT_THRUST): $(SRCDIR)/dot_product_thrust.cu
	$(CC) $(CFLAGS) $< -o $@

# Транспонирование
transpose: $(TRANS_RAW) $(TRANS_THRUST)

$(TRANS_RAW): $(SRCDIR)/transpose_raw.cu
	$(CC) $(CFLAGS) $< -o $@

$(TRANS_THRUST): $(SRCDIR)/transpose_thrust.cu
	$(CC) $(CFLAGS) $< -o $@

# Запуск с разными размерами
run-dot:
	@echo "========================================================="
	@echo "СКАЛЯРНОЕ ПРОИЗВЕДЕНИЕ"
	@echo "========================================================="
	@for size in 1024 4096 16384 65536 262144 1048576 4194304; do \
		echo "\n--- Размер: $$size ---"; \
		echo "\nСырой CUDA:"; \
		./$(DOT_RAW) $$size; \
		echo "\nThrust:"; \
		./$(DOT_THRUST) $$size; \
	done

run-transpose:
	@echo "========================================================="
	@echo "ТРАНСПОНИРОВАНИЕ"
	@echo "========================================================="
	@for size in 128 256 512 1024 2048 4096; do \
		echo "\n--- Размер матрицы: $$size x $$size ---"; \
		echo "\nСырой CUDA:"; \
		./$(TRANS_RAW) $$size $$size; \
		echo "\nThrust:"; \
		./$(TRANS_THRUST) $$size $$size; \
	done

run: run-dot run-transpose

clean:
	rm -f $(DOT_RAW) $(DOT_THRUST) $(TRANS_RAW) $(TRANS_THRUST)

.PHONY: all clean dot transpose run run-dot run-transpose benchmark