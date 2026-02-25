CC = gcc
GC = nvcc

CFLAGS = -arch=sm_86 -O2 -g -G

LAB_SOURCE = src/lab5.cu

LAB_TARGET = obj/lab5


.PHONY: all run pass loc clean

all: $(LAB_TARGET)


$(LAB_TARGET): $(LAB_SOURCE)
	@ $(GC) $(CFLAGS) $< -o $@

run:
	@ ./$(LAB_TARGET)

pass:
	@ nvprof --metrics gld_throughput,gst_throughput ./$(LAB_TARGET)

loc:
	@ nvprof --metrics local_load_throughput,local_store_throughput ./$(LAB_TARGET)

clean:
	@ rm obj/*
