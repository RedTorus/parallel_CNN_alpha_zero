// test_model.cpp
#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include "model.h"

int main() {
    // Create a dummy input tensor: batch size 1, 3 channels, 224x224 image.
    torch::Tensor input = torch::rand({1, 5, 8, 8});

    // Create a full Model object with 3 input channels and 10 output classes.
    Model model(5, 2);
    model->to(torch::kCUDA);
    model->eval();  // Set the model to evaluation mode.
    std::cout << "CUDA available? " << torch::cuda::is_available() << std::endl;
    std::cout << "Model created with input channels: 5, output classes: 2" << std::endl;

    // Measure time for forward pass.
    auto start = std::chrono::steady_clock::now();
    torch::Tensor output = model->forward(input);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Model output size: " << output.sizes() 
              << " (Forward pass took " << elapsed.count() << " seconds)" << std::endl;

    return 0;
}

