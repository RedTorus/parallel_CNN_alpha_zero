# Update this path to your actual libtorch installation directory
TORCH_PATH ?= /afs/.ece.cmu.edu/usr/hanlux/Private/open_spiel/open_spiel/libtorch/libtorch

CXX = g++
NVCC = nvcc

# Flags for the libtorch-based program
LIBTORCH_CXXFLAGS = -O3 -std=c++17 -I$(TORCH_PATH)/include -I$(TORCH_PATH)/include/torch/csrc/api/include -I/usr/local/cuda/include
LIBTORCH_LDFLAGS  = -L$(TORCH_PATH)/lib -L/usr/local/cuda/targets/x86_64-linux/lib -Wl,-rpath,$(TORCH_PATH)/lib -ltorch_cpu -ltorch -lc10 -lcudart

# NVCC flags for CUDA files (use -fPIC for position-independent code)
NVCCFLAGS = -O3 -std=c++17 -Xcompiler -fPIC -I$(TORCH_PATH)/include -I$(TORCH_PATH)/include/torch/csrc/api/include $(shell python3-config --includes)

# Source files
COMPARE_SRCS    = semi_compare.cc torso_conv_cuda.cu output_conv_cuda.cu
COMPARE_MODEL_SRCS    = semi_compare_model.cc torso_conv_cuda.cu output_conv_cuda.cu

# Object files
COMPARE_OBJS    = semi_compare.o torso_conv_cuda.o output_conv_cuda.o
COMPARE_MODEL_OBJS    = semi_compare_model.o torso_conv_cuda.o output_conv_cuda.o

# Target executables
COMPARE_TARGET    = compare_program
COMPARE_MODEL_TARGET    = compare_model_program

all: $(COMPARE_TARGET) $(COMPARE_MODEL_TARGET)

# Build the semi CNN compare program
$(COMPARE_TARGET): $(COMPARE_OBJS)
	$(CXX) $(LIBTORCH_CXXFLAGS) -o $@ $^ $(LIBTORCH_LDFLAGS)

# Build the semi CNN compare executable
$(COMPARE_MODEL_TARGET): $(COMPARE_MODEL_OBJS)
	$(CXX) $(LIBTORCH_CXXFLAGS) -o $@ $^ $(LIBTORCH_LDFLAGS)

# Compile rule for the pure CNN compare source file
semi_compare.o: semi_compare.cc
	$(CXX) $(LIBTORCH_CXXFLAGS) -c $< -o $@

# Compile rule for the model compare source file
semi_compare_model.o: semi_compare_model.cc
	$(CXX) $(LIBTORCH_CXXFLAGS) -c $< -o $@

# Compile rule for torso_conv_cuda.cu
torso_conv_cuda.o: torso_conv_cuda.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile rule for output_conv_cuda.cu
output_conv_cuda.o: output_conv_cuda.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Run the comparing program on pure CNNs
run_semi_compare_cnn: $(COMPARE_TARGET)
	./$(COMPARE_TARGET)

# Run the comparing program on semi model
run_semi_compare_model: $(COMPARE_MODEL_TARGET)
	./$(COMPARE_MODEL_TARGET)

compare_cnn: run_semi_compare_cnn

compare_model: run_semi_compare_model

clean:
	rm -f $(COMPARE_OBJS) $(COMPARE_MODEL_OBJS) $(COMPARE_TARGET) $(COMPARE_MODEL_TARGET) *.txt

