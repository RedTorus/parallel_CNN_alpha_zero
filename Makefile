# -------- CONFIGURATION --------
# Set paths to LibTorch and CUDA
LIBTORCH   ?= /afs/.ece.cmu.edu/usr/kaustabp/open_spiel/open_spiel/libtorch/libtorch
CUDA_PATH  ?= /usr/local/cuda

# Compilers
CXX        = g++
NVCC       = nvcc

# Include dirs (for both C++ and CUDA) and Python includes
INCLUDES   = -I$(LIBTORCH)/include \
	         -I$(LIBTORCH)/include/torch/csrc/api/include \
	         -I$(CUDA_PATH)/include \
	         $(shell python3-config --includes)

# Flags
CXXFLAGS   = -O3 -std=c++17 $(INCLUDES) \
	            -I$(LIBTORCH)/include/pybind11_ \
	            -fPIC
NVCCFLAGS  = -O3 -std=c++17 -Xcompiler -fPIC \
	         -I$(LIBTORCH)/include \
	         -I$(LIBTORCH)/include/torch/csrc/api/include \
	         -I$(LIBTORCH)/include/pybind11_ \
	         -I$(CUDA_PATH)/include \
	         $(shell python3-config --includes)


# Linker flags
LDFLAGS    = -L$(LIBTORCH)/lib \
	         -L$(CUDA_PATH)/targets/x86_64-linux/lib \
	         -Wl,-rpath,$(LIBTORCH)/lib \
	         -ltorch -ltorch_cpu -lc10 \
	         -ltorch_cuda -lc10_cuda -lcudart \
	         -ldl -lpthread -lrt

.PHONY: all clean
all: test_kernel

# Compile CUDA kernel
input_conv.o: input_conv.cu input_conv.h
	$(NVCC) $(NVCCFLAGS) -c input_conv.cu -o input_conv.o

test_blocks.o: test_blocks.cpp test_blocks.h
	$(CXX) $(CXXFLAGS) -c test_blocks.cpp -o test_blocks.o

# Compile C++ test harness
test_kernel.o: test_kernel.cpp input_conv.h test_blocks.h
	$(CXX) $(CXXFLAGS) -c test_kernel.cpp -o test_kernel.o

# Link objects using g++ so -Wl flags are understood
test_kernel: input_conv.o test_blocks.o test_kernel.o
	$(CXX) input_conv.o test_blocks.o test_kernel.o -o test_kernel $(LDFLAGS)

test_blocks: test_blocks.cpp model.h
	$(CXX) $(CXXFLAGS) test_blocks.cpp -o test_blocks $(LDFLAGS)

clean:
	rm -f *.o test_kernel
