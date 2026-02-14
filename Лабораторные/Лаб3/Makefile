CC = gcc
GC = nvcc

CFLAGS = -arch=sm_86 -O2 -g -G

LAB3_SOURCE = src/lab3.cu

LAB3_TARGET = obj/lab3


.PHONY: all run-3 clean

all: $(LAB3_TARGET)


$(LAB3_TARGET): $(LAB3_SOURCE)
	@ $(GC) $(CFLAGS) $< -o $@

run-3:
	@ ./$(LAB3_TARGET)

check-3:
	@ cuda-gdb ./$(LAB3_TARGET)

clean:
	@ rm obj/*
