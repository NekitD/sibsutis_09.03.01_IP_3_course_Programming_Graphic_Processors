CC = nvcc
ARCH = -arch=sm_86
CFLAGS = $(ARCH) -O2 -lineinfo -I$(SRCDIR) -D_STB_IMAGE_IMPLEMENTATION

SRCDIR = src
OBJDIR = obj
BINDIR = .
IMGDIR = img

TARGET = $(BINDIR)/neural_network

SOURCES = $(SRCDIR)/neural_network.cu $(SRCDIR)/data_loader.cu $(SRCDIR)/image_loader.cu

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

$(TARGET): $(SOURCES)
	@mkdir -p $(OBJDIR)
	@ $(CC) $(CFLAGS) $^ -o $@
	@echo "Создан исполняемый файл: $(TARGET)"

run: $(TARGET)
	@echo "ЗАПУСК НЕЙРОСЕТИ"
	./$(TARGET)

clean:
	@ rm -rf $(OBJDIR) $(TARGET)
	@echo "Очистка завершена"

clean-all: clean
	@ rm -f $(IMGDIR)/*.bin $(IMGDIR)/dims.txt
	@echo "Очистка завершена"

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

help:
	@echo "========================================================="
	@echo "ДОСТУПНЫЕ КОМАНДЫ"
	@echo "========================================================="
	@echo "make               - проверить данные и собрать программу"
	@echo "make run           - собрать и запустить нейросеть"
	@echo "make clean         - удалить объектные файлы и исполняемый файл"
	@echo "make clean-all     - полная очистка (включая бинарные данные)"
	@echo "make info          - показать информацию о бинарных данных"
	@echo "make help          - показать эту справку"
	@echo ""
	@echo "ПРИМЕЧАНИЕ:"
	@echo "  Перед первой компиляцией необходимо иметь бинарные файлы в img/:"
	@echo "    train_images.bin, train_labels.bin"
	@echo "    test_images.bin, test_labels.bin"
	@echo ""
	@echo "  Для конвертации из mnist.pkl используйте скрипт convert_mnist.py"

.PHONY: all clean clean-all run info check-data help