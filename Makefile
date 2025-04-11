# Update this path to your actual libtorch installation directory
TORCH_PATH ?= /afs/.ece.cmu.edu/usr/hanlux/Private/open_spiel/open_spiel/libtorch/libtorch

CXX = g++
NVCC = nvcc

# Flags for the libtorch-based program
LIBTORCH_CXXFLAGS = -O3 -std=c++17 -I$(TORCH_PATH)/include -I$(TORCH_PATH)/include/torch/csrc/api/include
LIBTORCH_LDFLAGS  = -L$(TORCH_PATH)/lib -L/usr/local/cuda/targets/x86_64-linux/lib -Wl,-rpath,$(TORCH_PATH)/lib -ltorch_cpu -ltorch -lc10 -lcudart

# NVCC flags for CUDA files (use -fPIC for position-independent code)
NVCCFLAGS = -O3 -std=c++17 -Xcompiler -fPIC -I$(TORCH_PATH)/include -I$(TORCH_PATH)/include/torch/csrc/api/include $(shell python3-config --includes)

# Source files
LIBTORCH_SRCS = semi_alpha_zero.cc
PARALLEL_SRCS    = semi_alpha_zero_par.cc torso_conv_cuda.cu output_conv_cuda.cu
COMPARE_SRCS    = semi_compare.cc torso_conv_cuda.cu output_conv_cuda.cu

# Object files
LIBTORCH_OBJS = semi_alpha_zero.o
PARALLEL_OBJS    = semi_alpha_zero_par.o torso_conv_cuda.o output_conv_cuda.o
COMPARE_OBJS    = semi_compare.o torso_conv_cuda.o output_conv_cuda.o

# Target executables
LIBTORCH_TARGET = semi_alpha_zero
PARALLEL_TARGET    = parallel_program
COMPARE_TARGET    = compare_program

.PHONY: all clean run_semi run_parallel

all: $(LIBTORCH_TARGET) $(PARALLEL_TARGET) $(COMPARE_TARGET)

# Build the libtorch executable
$(LIBTORCH_TARGET): $(LIBTORCH_OBJS)
	$(CXX) $(LIBTORCH_CXXFLAGS) -o $@ $^ $(LIBTORCH_LDFLAGS)

# Build the parallel executable
$(PARALLEL_TARGET): $(PARALLEL_OBJS)
	$(CXX) $(LIBTORCH_CXXFLAGS) -o $@ $^ $(LIBTORCH_LDFLAGS)

# Build the semi CNN compare executable
$(COMPARE_TARGET): $(COMPARE_OBJS)
	$(CXX) $(LIBTORCH_CXXFLAGS) -o $@ $^ $(LIBTORCH_LDFLAGS)

# Compile rule for the libtorch-based source file
semi_alpha_zero.o: semi_alpha_zero.cc
	$(CXX) $(LIBTORCH_CXXFLAGS) -c $< -o $@

# Compile rule for the parallel source file
semi_alpha_zero_par.o: semi_alpha_zero_par.cc
	$(CXX) $(LIBTORCH_CXXFLAGS) -c $< -o $@

# Compile rule for the pure CNN compare source file
semi_compare.o: semi_compare.cc
	$(CXX) $(LIBTORCH_CXXFLAGS) -c $< -o $@

# Compile rule for torso_conv_cuda.cu
torso_conv_cuda.o: torso_conv_cuda.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile rule for output_conv_cuda.cu
output_conv_cuda.o: output_conv_cuda.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Target to run the libtorch-based executable
run_semi: $(LIBTORCH_TARGET)
	./$(LIBTORCH_TARGET)

# Target to run the parallel executable
run_parallel: $(PARALLEL_TARGET)
	./$(PARALLEL_TARGET)

# Run the comparing program on pure CNNs
run_semi_compare_cnn: $(COMPARE_TARGET)
	./$(COMPARE_TARGET)

# Compare the outputs of the two programs
compare: run_semi run_parallel
	@echo "Comparing output_libtorch.txt and output_par.txt..."
	@diff -u output_libtorch.txt output_par.txt && echo "Outputs match!" || echo "Outputs differ!"

compare_cnn: run_semi_compare_cnn

clean:
	rm -f $(LIBTORCH_OBJS) $(PARALLEL_OBJS) $(COMPARE_OBJS) $(LIBTORCH_TARGET) $(PARALLEL_TARGET) $(COMPARE_TARGET) *.txt

