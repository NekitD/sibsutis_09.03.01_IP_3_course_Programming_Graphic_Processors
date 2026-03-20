CC = nvcc
ARCH = -arch=sm_86
CFLAGS = $(ARCH) -O2 -lineinfo

SRCDIR = src
OBJDIR = obj
BINDIR = .

TARGET = $(BINDIR)/neural_network
SOURCES = $(SRCDIR)/neural_network.cu

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SOURCES)
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(OBJDIR) $(TARGET)

.PHONY: all clean run