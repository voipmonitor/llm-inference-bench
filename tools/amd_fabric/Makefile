CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -march=native -Wall -Wextra
LDFLAGS ?= -lnuma -pthread

all: llm_amd_fabric

llm_amd_fabric: llm_amd_fabric.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f llm_amd_fabric

.PHONY: all clean
