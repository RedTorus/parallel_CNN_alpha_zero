// test_blocks.cpp
#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include "model.h"

int main() {
    // Create a dummy input tensor: batch size 1, 3 channels, 224x224 image.
    torch::Tensor input = torch::rand({1, 3, 224, 224});

    //---------- Test Conv2dBlock ----------
    std::cout << "Testing Conv2dBlock:" << std::endl;
    Conv2dBlock convBlock(3, 16, 3, 1, 1);
    auto start = std::chrono::steady_clock::now();
    torch::Tensor output = convBlock->forward(input);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Conv2dBlock output size: " << output.sizes()
              << " (Forward pass took " << elapsed.count() << " seconds)" << std::endl;

    //---------- Test ResInputBlock ----------
    std::cout << "\nTesting ResInputBlock:" << std::endl;
    ResInputBlock resInput(3, 64, 3, 1, 1);
    start = std::chrono::steady_clock::now();
    output = resInput->forward(input);
    end = std::chrono::steady_clock::now();
    elapsed = end - start;
    std::cout << "ResInputBlock output size: " << output.sizes()
              << " (Forward pass took " << elapsed.count() << " seconds)" << std::endl;

    //---------- Test ResTorsoBlock ----------
    std::cout << "\nTesting ResTorsoBlock:" << std::endl;
    // Simulate a feature map with 64 channels and spatial dimensions 56x56.
    torch::Tensor featureMap = torch::rand({1, 64, 56, 56});
    ResTorsoBlock resTorso(64, 3, 1, 1);
    start = std::chrono::steady_clock::now();
    output = resTorso->forward(featureMap);
    end = std::chrono::steady_clock::now();
    elapsed = end - start;
    std::cout << "ResTorsoBlock output size: " << output.sizes()
              << " (Forward pass took " << elapsed.count() << " seconds)" << std::endl;

    //---------- Test ResOutputBlock ----------
    std::cout << "\nTesting ResOutputBlock:" << std::endl;
    // Assuming the feature map is 64 channels with spatial dimensions 56x56.
    int flattened_features = 64 * 56 * 56;
    // Reuse the featureMap tensor from above.
    ResOutputBlock resOutput(flattened_features, 10);
    start = std::chrono::steady_clock::now();
    output = resOutput->forward(featureMap);
    end = std::chrono::steady_clock::now();
    elapsed = end - start;
    std::cout << "ResOutputBlock output size: " << output.sizes()
              << " (Forward pass took " << elapsed.count() << " seconds)" << std::endl;

    //---------- Test MLPOutputBlock ----------
    std::cout << "\nTesting MLPOutputBlock:" << std::endl;
    MLPOutputBlock mlpOutput(flattened_features, 128, 10);
    start = std::chrono::steady_clock::now();
    output = mlpOutput->forward(featureMap);
    end = std::chrono::steady_clock::now();
    elapsed = end - start;
    std::cout << "MLPOutputBlock output size: " << output.sizes()
              << " (Forward pass took " << elapsed.count() << " seconds)" << std::endl;

    return 0;
}

