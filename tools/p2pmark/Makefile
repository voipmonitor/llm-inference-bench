CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(or $(shell which nvcc 2>/dev/null),$(wildcard $(CUDA_HOME)/bin/nvcc),nvcc)

CFLAGS ?= -O2 -std=c++20 -arch=native
LDFLAGS ?= -lcudart -lnccl

all: llm_p2pmark

llm_p2pmark: llm_p2pmark.cu
	$(NVCC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f llm_p2pmark

.PHONY: all clean
