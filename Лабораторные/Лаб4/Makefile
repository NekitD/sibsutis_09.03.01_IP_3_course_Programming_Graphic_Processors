CC = gcc
GC = nvcc

CFLAGS = -arch=sm_86 -O2 -g -G

LAB4_SOURCE = src/lab4.cu

LAB4_TARGET = obj/lab4


.PHONY: all run-4 clean

all: $(LAB4_TARGET)


$(LAB4_TARGET): $(LAB4_SOURCE)
	@ $(GC) $(CFLAGS) $< -o $@

run-4:
	@ ./$(LAB4_TARGET)

clean:
	@ rm obj/*
