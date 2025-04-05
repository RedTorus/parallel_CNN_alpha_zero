// test_model.cpp
#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include "model.h"

int main() {
    // Create a dummy input tensor: batch size 1, 3 channels, 224x224 image.
    torch::Tensor input = torch::rand({1, 3, 224, 224});

    // Create a full Model object with 3 input channels and 10 output classes.
    Model model(3, 10);

    // Measure time for forward pass.
    auto start = std::chrono::steady_clock::now();
    torch::Tensor output = model->forward(input);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Model output size: " << output.sizes() 
              << " (Forward pass took " << elapsed.count() << " seconds)" << std::endl;

    return 0;
}

