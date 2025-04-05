// test_compare.cpp
#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include "model.h"

// Helper function to check if two outputs are identical within a tolerance.
bool outputs_identical(const torch::Tensor& out1, const torch::Tensor& out2, double tol = 1e-5) {
    return torch::allclose(out1, out2, tol);
}

int main() {
    // Create a full Model object with 3 input channels and 10 output classes.
    Model model(3, 10);
    model->eval(); // Set model to evaluation mode.

    // Create a dummy input tensor: batch size 1, 3 channels, 224x224 image.
    torch::Tensor input = torch::rand({1, 3, 224, 224});

    // Run forward pass the first time and measure time.
    auto start = std::chrono::steady_clock::now();
    torch::Tensor output1 = model->forward(input);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;
    std::cout << "First forward pass took " << elapsed1.count() << " seconds." << std::endl;

    // Run forward pass the second time and measure time.
    start = std::chrono::steady_clock::now();
    torch::Tensor output2 = model->forward(input);
    end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed2 = end - start;
    std::cout << "Second forward pass took " << elapsed2.count() << " seconds." << std::endl;

    // Check if the two outputs are identical.
    if (outputs_identical(output1, output2)) {
        std::cout << "The network outputs are identical." << std::endl;
    } else {
        std::cout << "The network outputs are different." << std::endl;
    }

    return 0;
}

