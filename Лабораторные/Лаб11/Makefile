CC = nvcc
ARCH = -arch=sm_86
CFLAGS = $(ARCH) -O2 -lineinfo

SRCDIR = src
OBJDIR = obj
BINDIR = .

TARGETS = $(OBJDIR)/memory_compare $(OBJDIR)/vector_add_streams $(OBJDIR)/dot_product_streams

.PHONY: all clean run

all: $(TARGETS)

$(OBJDIR)/memory_compare: $(SRCDIR)/memory_compare.cu
	@ $(CC) $(CFLAGS) $< -o $@

$(OBJDIR)/vector_add_streams: $(SRCDIR)/vector_add_streams.cu
	@ $(CC) $(CFLAGS) $< -o $@

$(OBJDIR)/dot_product_streams: $(SRCDIR)/dot_product_streams.cu
	@ $(CC) $(CFLAGS) $< -o $@

run: all
	@ $(OBJDIR)/memory_compare
	@ $(OBJDIR)/vector_add_streams
	@ $(OBJDIR)/dot_product_streams

clean:
	@ rm -f $(TARGETS)

.PHONY: all clean run