// test_blocks.cpp
#include <torch/torch.h>
#include <iostream>
#include <cuda_runtime.h>
#include "model.h"

// Global device variable: change torch::kCUDA to torch::kCPU if needed.
const torch::Device device(torch::kCUDA);

bool outputs_identical(const torch::Tensor& out1, const torch::Tensor& out2, double tol = 1e-4) {
    return torch::allclose(out1, out2, tol);
}

torch::Tensor computeAverageAbsoluteDifference(const torch::Tensor& tensor1, const torch::Tensor& tensor2) {
    // Ensure both tensors are on the same device
    TORCH_CHECK(tensor1.device() == tensor2.device(), "Tensors must be on the same device");

    // Compute elementwise absolute difference
    torch::Tensor abs_diff = torch::abs(tensor1 - tensor2);

    // Compute the average of the absolute differences
    torch::Tensor avg_diff = abs_diff.mean();

    return avg_diff;
}

torch::Tensor computeAbsoluteDifferenceSum(const torch::Tensor& tensor1, const torch::Tensor& tensor2) {
    // Ensure both tensors are on the same device
    TORCH_CHECK(tensor1.device() == tensor2.device(), "Tensors must be on the same device");

    // Compute elementwise absolute difference
    torch::Tensor abs_diff = torch::abs(tensor1 - tensor2);

    // Sum all elements
    torch::Tensor sum_diff = abs_diff.sum();

    return sum_diff;
}

void testConv2dBlock(const torch::Tensor& input) {
    std::cout << "Testing Conv2dBlock:" << std::endl;
    Conv2dBlock convBlock(5, 128, 3, 1, 1);
    // Move module to the global device.
    convBlock->to(device);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);
    torch::Tensor output = convBlock->forward(input);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    std::cout << "Conv2dBlock output size: " << output.sizes()
              << " (Forward pass took " << elapsedTime / 1000.0f << " seconds)" << std::endl;

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

void testConv2dBlock2(const torch::Tensor& input, const torch::Tensor& conv_weights) {
    std::cout << "Testing Conv2dBlock:" << std::endl;
    Conv2dSimple convBlock(5, 128, 3, 1, 1);
    convBlock->conv->weight = conv_weights;
    convBlock->conv->bias = torch::zeros({128}, torch::TensorOptions().device(device));
    // Move module to the global device.
    convBlock->to(device);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);
    torch::Tensor output = convBlock->forward(input);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    std::cout << "Conv2dBlock output size: " << output.sizes()
              << " (Forward pass took " << elapsedTime / 1000.0f << " seconds)" << std::endl;

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

torch::Tensor testConv2dBlock3(const torch::Tensor& input, const torch::Tensor& conv_weights) {
    std::cout << "Testing Conv2dBlock:" << std::endl;
    Conv2dSimple convBlock(5, 128, 3, 1, 1);
    convBlock->conv->weight = conv_weights;
    convBlock->conv->bias = torch::zeros({128}, torch::TensorOptions().device(device));
    // Move module to the global device.
    convBlock->to(device);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);
    torch::Tensor output = convBlock->forward(input);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    std::cout << "Conv2dBlock output size: " << output.sizes()
              << " (Forward pass took " << elapsedTime / 1000.0f << " seconds)" << std::endl;

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    return output;
}

torch::Tensor testConv2dBlock4(const torch::Tensor& input, const torch::Tensor& conv_weights) {
    std::cout << "Testing Conv2dBlock:" << std::endl;
    Conv2dBlock convBlock(5, 128, 3, 1, 1);
    convBlock->conv->weight = conv_weights;
    convBlock->conv->bias = torch::zeros({128}, torch::TensorOptions().device(device));
    // Move module to the global device.
    convBlock->to(device);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);
    torch::Tensor output = convBlock->forward(input);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    std::cout << "Conv2dBlock output size: " << output.sizes()
              << " (Forward pass took " << elapsedTime / 1000.0f << " seconds)" << std::endl;

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    return output;
}

void testResInputBlock(const torch::Tensor& input) {
    std::cout << "\nTesting ResInputBlock:" << std::endl;
    ResInputBlock resInput(3, 64, 3, 1, 1);
    resInput->to(device);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);
    torch::Tensor output = resInput->forward(input);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    std::cout << "ResInputBlock output size: " << output.sizes()
              << " (Forward pass took " << elapsedTime / 1000.0f << " seconds)" << std::endl;

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

void testResTorsoBlock(const torch::Tensor& featureMap) {
    std::cout << "\nTesting ResTorsoBlock:" << std::endl;
    ResTorsoBlock resTorso(64, 3, 1, 1);
    resTorso->to(device);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);
    torch::Tensor output = resTorso->forward(featureMap);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    std::cout << "ResTorsoBlock output size: " << output.sizes()
              << " (Forward pass took " << elapsedTime / 1000.0f << " seconds)" << std::endl;

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

void testResOutputBlock(const torch::Tensor& featureMap, int flattened_features) {
    std::cout << "\nTesting ResOutputBlock:" << std::endl;
    ResOutputBlock resOutput(flattened_features, 10);
    resOutput->to(device);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);
    torch::Tensor output = resOutput->forward(featureMap);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    std::cout << "ResOutputBlock output size: " << output.sizes()
              << " (Forward pass took " << elapsedTime / 1000.0f << " seconds)" << std::endl;

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

void testMLPOutputBlock(const torch::Tensor& featureMap, int flattened_features) {
    std::cout << "\nTesting MLPOutputBlock:" << std::endl;
    MLPOutputBlock mlpOutput(flattened_features, 128, 10);
    mlpOutput->to(device);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);
    torch::Tensor output = mlpOutput->forward(featureMap);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    std::cout << "MLPOutputBlock output size: " << output.sizes()
              << " (Forward pass took " << elapsedTime / 1000.0f << " seconds)" << std::endl;

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

// int main() {
//     std::cout << "CUDA available? " << torch::cuda::is_available() << std::endl;

//     // Create a dummy input tensor on the global device.
//     torch::Tensor input = torch::rand({1, 5, 8, 8}, torch::TensorOptions().device(device));
//     torch::Tensor conv_weights = torch::rand({128, 5, 3, 3}, torch::TensorOptions().device(device));
//     //torch::rand({1, 3, 224, 224}, torch::TensorOptions().device(device));
//     testConv2dBlock2(input, conv_weights);
//     //testResInputBlock(input);

//     // Create a dummy feature map on the global device.
//     torch::Tensor featureMap = torch::rand({1, 64, 56, 56}, torch::TensorOptions().device(device));
//     testResTorsoBlock(featureMap);

//     int flattened_features = 64 * 56 * 56;
//     testResOutputBlock(featureMap, flattened_features);
//     testMLPOutputBlock(featureMap, flattened_features);

//     return 0;
// }
