CC = gcc
GC = nvcc

CFLAGS = -arch=sm_86 -O2

LAB2A_SOURCE = src/lab2a.c
LAB2B_SOURCE = src/lab2b.cu

LAB2A_TARGET = obj/lab2a
LAB2B_TARGET = obj/lab2b


.PHONY: all run-2a run-2b clean

all: $(LAB2A_TARGET) $(LAB2B_TARGET)

$(LAB2A_TARGET): $(LAB2A_SOURCE)
	@ $(CC) $< -o $@ -pthread 

$(LAB2B_TARGET): $(LAB2B_SOURCE)
	@ $(GC) $(CFLAGS) $< -o $@

run-2a:
	@ ./$(LAB2A_TARGET)

run-2b:
	@ ./$(LAB2B_TARGET)

clean:
	@ rm obj/*
