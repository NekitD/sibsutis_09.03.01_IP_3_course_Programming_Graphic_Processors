CC = gcc
GC = nvcc

LAB2A_SOURCE = lab2a
LAB2B_SOURCE = lab2b

LAB2A_TARGET = obj/lab2a
LAB2B_TARGET = obj/lab2b


.PHONY: all run-2a run-2b clean

all: $(LAB2A_TARGET) $(LAB2B_TARGET)

$(LAB2A_TARGET): $(LAB2A_SOURCE)
	@ $(CC) $< -o $@ -pthread 

$(LAB2B_TARGET): $(LAB2B_SOURCE)
	@ $(GC) $< -o $@

run-2a:
	./$(LAB2A_TARGET)

run-2b:
	./$(LAB2B_TARGET)

clean:
	rm obj/*
