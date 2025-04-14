# Set LIBTORCH to your libtorch installation directory.
# You can override this variable externally, e.g., `make LIBTORCH=/your/path/to/libtorch`
LIBTORCH ?= /home/kaust/libtorch

# Compiler and flags
CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall -D_GLIBCXX_USE_CXX11_ABI=1 \
           -I$(LIBTORCH)/include \
           -I$(LIBTORCH)/include/torch/csrc/api/include \
           -I/usr/local/cuda/include

# Linker flags: use RPATH so that the shared libraries are found at runtime.
# The --no-as-needed / --as-needed flags ensure that all the provided libraries are retained.
LDFLAGS  = -L$(LIBTORCH)/lib -Wl,-rpath,$(LIBTORCH)/lib \
           -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -Wl,--as-needed \
           -L/usr/local/cuda/lib64 -lcudart

# Targets
all: test_blocks test_model test_compare

test_blocks: test_blocks.cpp model.h
	$(CXX) $(CXXFLAGS) test_blocks.cpp -o test_blocks $(LDFLAGS)

test_model: test_model.cpp model.h
	$(CXX) $(CXXFLAGS) test_model.cpp -o test_model $(LDFLAGS)

test_compare: test_compare.cpp model.h
	$(CXX) $(CXXFLAGS) test_compare.cpp -o test_compare $(LDFLAGS)

clean:
	rm -f test_blocks test_model test_compare
