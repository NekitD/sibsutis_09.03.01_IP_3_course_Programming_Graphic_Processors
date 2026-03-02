CC = nvcc
ARCH = -arch=sm_86
CFLAGS = $(ARCH) -O2 -lineinfo

TARGET_A = obj/lab7a
TARGET_B = obj/lab7b
SOURCE_A = src/lab7a.cu
SOURCE_B = src/lab7b.cu


all: $(TARGET_A) $(TARGET_B)

$(TARGET_A): $(SOURCE_A)
	@ mkdir -p obj
	@ $(CC) $(CFLAGS) $< -o $@

$(TARGET_B): $(SOURCE_B)
	@ mkdir -p obj
	@ $(CC) $(CFLAGS) $< -o $@

run-a:
	@ ./$(TARGET_A)

run-b:
	@ ./$(TARGET_B)

clean:
	@ rm -rf obj/*

.PHONY: all run clean