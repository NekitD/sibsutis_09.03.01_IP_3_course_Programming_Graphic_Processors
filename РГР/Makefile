CC = nvcc
ARCH = -arch=sm_86
CFLAGS = $(ARCH) -O2 -lineinfo -I$(SRCDIR)

SRCDIR = src
OBJDIR = obj
BINDIR = .
IMGDIR = img

TARGET = $(BINDIR)/neural_network

SOURCES = $(SRCDIR)/neural_network.cu
UTILS = $(SRCDIR)/utils.h
DATALOADER = $(SRCDIR)/data_loader.cu

OBJECTS = $(OBJDIR)/neural_network.o

.PHONY: all clean run convert check-data

all: check-data $(TARGET)

check-data:
	@if [ ! -f $(IMGDIR)/train_images.bin ] || [ ! -f $(IMGDIR)/train_labels.bin ] || \
	   [ ! -f $(IMGDIR)/test_images.bin ] || [ ! -f $(IMGDIR)/test_labels.bin ]; then \
		echo "========================================================="; \
		echo "Бинарные файлы MNIST не найдены!"; \
		echo "========================================================="; \
		echo "Сначала необходимо сконвертировать mnist.pkl в бинарные файлы."; \
		echo "Запустите конвертацию с помощью скрипта:"; \
		echo "  python3 convert_mnist.py"; \
		echo ""; \
		echo "Или создайте бинарные файлы вручную:"; \
		echo "  train_images.bin, train_labels.bin,"; \
		echo "  test_images.bin, test_labels.bin"; \
		echo "========================================================="; \
		exit 1; \
	else \
		echo "Данные MNIST найдены:"; \
		ls -lh $(IMGDIR)/*.bin; \
	fi

$(TARGET): $(SOURCES) $(UTILS) $(DATALOADER)
	@mkdir -p $(OBJDIR)
	@echo "========================================================="
	@echo "Компиляция нейросети для распознавания цифр"
	@echo "========================================================="
	$(CC) $(CFLAGS) $< -o $@
	@echo "Создан исполняемый файл: $(TARGET)"


run: $(TARGET)
	@echo "========================================================="
	@echo "ЗАПУСК НЕЙРОСЕТИ"
	@echo "========================================================="
	./$(TARGET)

clean:
	@echo "========================================================="
	@echo "ОЧИСТКА"
	@echo "========================================================="
	rm -rf $(OBJDIR) $(TARGET)
	@echo "Очистка завершена"

clean-all: clean
	@echo "Удаление бинарных данных..."
	rm -f $(IMGDIR)/*.bin $(IMGDIR)/dims.txt
	@echo "Полная очистка завершена"

info:
	@echo "========================================================="
	@echo "ИНФОРМАЦИЯ О ДАННЫХ"
	@echo "========================================================="
	@if [ -f $(IMGDIR)/train_images.bin ]; then \
		echo "train_images.bin: $$(ls -lh $(IMGDIR)/train_images.bin | awk '{print $$5}')"; \
		echo "train_labels.bin: $$(ls -lh $(IMGDIR)/train_labels.bin | awk '{print $$5}')"; \
		echo "test_images.bin:  $$(ls -lh $(IMGDIR)/test_images.bin | awk '{print $$5}')"; \
		echo "test_labels.bin:  $$(ls -lh $(IMGDIR)/test_labels.bin | awk '{print $$5}')"; \
	else \
		echo "Бинарные данные не найдены"; \
	fi

check-data-size:
	@echo "========================================================="
	@echo "РАЗМЕРЫ ДАННЫХ"
	@echo "========================================================="
	@if [ -f $(IMGDIR)/train_images.bin ]; then \
		echo "train_images.bin: $$(stat -c%s $(IMGDIR)/train_images.bin) байт"; \
		echo "train_labels.bin: $$(stat -c%s $(IMGDIR)/train_labels.bin) байт"; \
		echo "test_images.bin:  $$(stat -c%s $(IMGDIR)/test_images.bin) байт"; \
		echo "test_labels.bin:  $$(stat -c%s $(IMGDIR)/test_labels.bin) байт"; \
		echo ""; \
		echo "Ожидаемые размеры для MNIST:"; \
		echo "  train_images.bin: 50000 * 784 * 4 = 156,800,000 байт (~149.5 MB)"; \
		echo "  train_labels.bin: 50000 * 4 = 200,000 байт (~195 KB)"; \
		echo "  test_images.bin:  10000 * 784 * 4 = 31,360,000 байт (~29.9 MB)"; \
		echo "  test_labels.bin:  10000 * 4 = 40,000 байт (~39 KB)"; \
	fi

.PHONY: all clean clean-all run info check-data check-data-size