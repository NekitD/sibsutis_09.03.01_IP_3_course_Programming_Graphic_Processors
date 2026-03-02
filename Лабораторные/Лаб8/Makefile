CC = nvcc
GENCODE = -gencode arch=compute_86,code=sm_86
OPT = -O2
# Раздельная компиляция
RDC = -rdc=true

CFLAGS = $(GENCODE) $(OPT) $(RDC)

TARGET = vector_add

KERNEL_SRC = src/kernel.cu
MAIN_SRC = src/main.cu

# Создаем директорию obj
OBJDIR = obj
$(shell mkdir -p $(OBJDIR))

# Промежуточные файлы
KERNEL_PTX = $(OBJDIR)/kernel.ptx
KERNEL_CUBIN = $(OBJDIR)/kernel.sm_86.cubin
KERNEL_O = $(OBJDIR)/kernel.o
MAIN_O = $(OBJDIR)/main.o
FATBIN = $(OBJDIR)/kernel.fatbin
FATBIN_C = $(OBJDIR)/kernel.fatbin.c
FATBIN_O = $(OBJDIR)/kernel_fatbin.o

.PHONY: all clean ptx cubin kernel_obj fatbin main_obj link final run \
        show-ptx disassemble show-fatbin symbols list simple

all: ptx cubin kernel_obj fatbin main_obj final
	@echo "========================================================="
	@echo "ВСЕ ЭТАПЫ УСПЕШНО ЗАВЕРШЕНЫ"
	@echo "========================================================="
	@ls -la $(OBJDIR)/

# ШАГ 1: Компиляция в PTX
ptx: $(KERNEL_PTX)

$(KERNEL_PTX): $(KERNEL_SRC)
	@echo "========================================================="
	@echo "ШАГ 1: Компиляция в PTX"
	@echo "========================================================="
	$(CC) $(GENCODE) -ptx $< -o $@
	@echo "Создан PTX файл: $@"
	@echo ""

# ШАГ 2: Компиляция в CUDA Binary (cubin)
cubin: $(KERNEL_CUBIN)

$(KERNEL_CUBIN): $(KERNEL_PTX)
	@echo "========================================================="
	@echo "ШАГ 2: Компиляция в CUDA Binary (cubin)"
	@echo "========================================================="
	ptxas -arch=sm_86 $< -o $@
	@echo "Создан cubin файл: $@"
	@echo ""

# ШАГ 3: Компиляция ядра в объектный файл с раздельной компиляцией
kernel_obj: $(KERNEL_O)

$(KERNEL_O): $(KERNEL_SRC)
	@echo "========================================================="
	@echo "ШАГ 3: Компиляция ядра в объектный файл"
	@echo "========================================================="
	$(CC) $(CFLAGS) -c $< -o $@
	@echo "Создан объектный файл ядра: $@"
	@echo ""

# ШАГ 4: Создание fatbinary
fatbin: $(FATBIN) $(FATBIN_C)

$(FATBIN) $(FATBIN_C): $(KERNEL_CUBIN)
	@echo "========================================================="
	@echo "ШАГ 4: Создание fatbinary"
	@echo "========================================================="
	fatbinary --create=$(FATBIN) -64 \
	          --image3=kind=elf,sm=86,file=$< \
	          --embedded-fatbin=$(FATBIN_C)
	@echo "Создан fatbin файл: $(FATBIN)"
	@echo "Создан fatbin.c файл: $(FATBIN_C)"
	@echo ""

# ШАГ 5: Компиляция основного файла с раздельной компиляцией
main_obj: $(MAIN_O)

$(MAIN_O): $(MAIN_SRC)
	@echo "========================================================="
	@echo "ШАГ 5: Компиляция основного файла"
	@echo "========================================================="
	$(CC) $(CFLAGS) -c $< -o $@
	@echo "Создан объектный файл: $@"
	@echo ""

# ШАГ 6: Компиляция fatbin.c в объектный файл
fatbin_obj: $(FATBIN_O)

$(FATBIN_O): $(FATBIN_C)
	@echo "========================================================="
	@echo "ШАГ 6: Компиляция fatbin.c в объектный файл"
	@echo "========================================================="
	gcc -c $< -o $@
	@echo "Создан объектный файл fatbin: $@"
	@echo ""

# ШАГ 7: Финальная компоновка (БЕЗ -dlink!)
final: $(TARGET)

$(TARGET): $(MAIN_O) $(KERNEL_O) $(FATBIN_O)
	@echo "========================================================="
	@echo "ШАГ 7: Финальная компоновка"
	@echo "========================================================="
	# Компоновка всех объектных файлов напрямую
	$(CC) $^ -o $@ -lcudart -lcudadevrt
	@echo "Создан исполняемый файл: $@"
	@echo ""

# Запуск программы
run: $(TARGET)
	@echo "========================================================="
	@echo "ЗАПУСК ПРОГРАММЫ"
	@echo "========================================================="
	./$(TARGET)

# Показать содержимое PTX файла
show-ptx: $(KERNEL_PTX)
	@echo "========================================================="
	@echo "СОДЕРЖИМОЕ PTX ФАЙЛА"
	@echo "========================================================="
	cat $<

# Дизассемблирование cubin
disassemble: $(KERNEL_CUBIN)
	@echo "========================================================="
	@echo "ДИЗАССЕМБЛИРОВАНИЕ CUBIN"
	@echo "========================================================="
	cuobjdump -sass $<

# Показать информацию о fatbinary
show-fatbin: $(FATBIN)
	@echo "========================================================="
	@echo "ИНФОРМАЦИЯ О FATBINARY"
	@echo "========================================================="
	cuobjdump $<

# Показать символы в объектных файлах
symbols: $(KERNEL_O) $(MAIN_O)
	@echo "========================================================="
	@echo "СИМВОЛЫ В ОБЪЕКТНЫХ ФАЙЛАХ"
	@echo "========================================================="
	@echo "Символы в kernel.o (с RDC):"
	nm $(KERNEL_O) | grep vector_add || true
	@echo ""
	@echo "Неопределённые символы в main.o (с RDC):"
	nm -u $(MAIN_O) | grep vector_add || true

# Очистка
clean:
	@echo "========================================================="
	@echo "ОЧИСТКА"
	@echo "========================================================="
	rm -f $(OBJDIR)/* $(TARGET) *.ii *.cudafe1* *.module_id
	@echo "Очистка завершена."

# Показать все созданные файлы
list:
	@echo "========================================================="
	@echo "СОЗДАННЫЕ ФАЙЛЫ"
	@echo "========================================================="
	ls -la $(OBJDIR)/

# Упрощённая рабочая версия (для проверки)
simple:
	@echo "========================================================="
	@echo "УПРОЩЁННАЯ КОМПИЛЯЦИЯ (для проверки)"
	@echo "========================================================="
	$(CC) $(GENCODE) src/kernel.cu src/main.cu -o $(TARGET)
	./$(TARGET)

.PHONY: all clean ptx cubin kernel_obj fatbin main_obj fatbin_obj final run \
        show-ptx disassemble show-fatbin symbols list simple