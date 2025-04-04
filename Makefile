# Update this path to your actual libtorch installation directory
TORCH_PATH ?= /Users/hanluxu/Downloads/libtorch

CXX = g++

# Flags for the libtorch-based program
LIBTORCH_CXXFLAGS = -O3 -std=c++17 -I$(TORCH_PATH)/include -I$(TORCH_PATH)/include/torch/csrc/api/include
LIBTORCH_LDFLAGS  = -L$(TORCH_PATH)/lib -Wl,-rpath,$(TORCH_PATH)/lib -ltorch_cpu -ltorch -lc10

# Flags for the plain program (which does not require libtorch)
PLAIN_CXXFLAGS = -O3 -std=c++17

# Source files
LIBTORCH_SRCS = semi_alpha_zero.cc
PLAIN_SRCS    = semi_alpha_zero_without_torch.cc

# Object files
LIBTORCH_OBJS = semi_alpha_zero.o
PLAIN_OBJS    = semi_alpha_zero_without_torch.o

# Target executables
LIBTORCH_TARGET = semi_alpha_zero
PLAIN_TARGET    = plain_program

.PHONY: all clean run_semi run_plain

all: $(LIBTORCH_TARGET) $(PLAIN_TARGET)

# Build the libtorch executable
$(LIBTORCH_TARGET): $(LIBTORCH_OBJS)
	$(CXX) $(LIBTORCH_CXXFLAGS) -o $@ $^ $(LIBTORCH_LDFLAGS)

# Build the plain executable
$(PLAIN_TARGET): $(PLAIN_OBJS)
	$(CXX) $(PLAIN_CXXFLAGS) -o $@ $^

# Compile rule for the libtorch-based source file
semi_alpha_zero.o: semi_alpha_zero.cc
	$(CXX) $(LIBTORCH_CXXFLAGS) -c $< -o $@

# Compile rule for the plain source file
semi_alpha_zero_without_torch.o: semi_alpha_zero_without_torch.cc
	$(CXX) $(PLAIN_CXXFLAGS) -c $< -o $@

# Target to run the libtorch-based executable
run_semi: $(LIBTORCH_TARGET)
	./$(LIBTORCH_TARGET)

# Target to run the plain executable
run_plain: $(PLAIN_TARGET)
	./$(PLAIN_TARGET)

clean:
	rm -f $(LIBTORCH_OBJS) $(PLAIN_OBJS) $(LIBTORCH_TARGET) $(PLAIN_TARGET) *.txt

