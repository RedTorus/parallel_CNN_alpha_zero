#include <torch/torch.h>
#include <iostream>
#include <cuda_runtime.h>
#include "input_conv.h"
#include "test_blocks.h"
#include "model.h"

int main() {
    // Define the input tensor
    int batch_size = 1;
    int input_channels = 5;
    int input_height = 8;
    int input_width = 8;
    torch::Tensor input = torch::rand({batch_size, input_channels, input_height, input_width}).cuda();

    // Define the convolution weights
    int output_channels = 128;
    int kernel_size = 3;
    torch::Tensor conv_weights = torch::rand({output_channels, input_channels, kernel_size, kernel_size}).cuda();

    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);

    // Perform the forward pass
    torch::Tensor output = input_conv_forward(input, conv_weights);

    // Record the stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print the output shape and time taken
    std::cout << "Output shape: " << output.sizes() << std::endl;
    std::cout << "Time taken: " << milliseconds / 1000.0 << " seconds" << std::endl;

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Test Conv2dBlock
    //time
    torch::Tensor out2 = testConv2dBlock3(input, conv_weights);
    std::cout << "Here we are comparing conv2d block with input_conv_forward" << std::endl;
    bool identical = outputs_identical(output, out2);
    if (identical) {
        std::cout << "Outputs are identical." << std::endl;
    } else {
        std::cout << "Outputs are different." << std::endl;
    }

    // Test Conv2dKernelBlock

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    Conv2dKernelBlock conv2d_kernel_block(input_channels, output_channels, kernel_size);
    conv2d_kernel_block->weight = conv_weights;
    cudaEventRecord(start, 0);
    torch::Tensor output_kernel = conv2d_kernel_block->forward(input);
    //std::cout << "---Output kernel shape: " << output_kernel.sizes() << std::endl;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float kernel_block_time = 0;
    cudaEventElapsedTime(&kernel_block_time, start, stop);
    std::cout << "Output shape: " << output_kernel.sizes() << std::endl;
    std::cout << "Time taken for Conv2dKernelBlock: " << kernel_block_time / 1000.0 << " seconds" << std::endl;


    torch::Tensor output_desired = testConv2dBlock4(input, conv_weights);

    std::cout << "Here we are comparing conv2d kernel block with conv2d block" << std::endl;
    identical = outputs_identical(output_kernel, output_desired, 1e-3);
    if (identical) {
        std::cout << "Outputs are identical." << std::endl;
    } else {
        std::cout << "Outputs are different." << std::endl;
    }

    torch::Tensor diffsum = computeAbsoluteDifferenceSum(output_kernel, output_desired);
    std::cout << "Sum of absolute differences: " << diffsum.item<float>() << std::endl;
    torch::Tensor avgdiff = computeAverageAbsoluteDifference(output_kernel, output_desired);
    std::cout << "Average absolute difference: " << avgdiff.item<float>() << std::endl;

    return 0;
}
