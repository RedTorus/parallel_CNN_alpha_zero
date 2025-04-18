#include <torch/torch.h>
#include <iostream>
#include <cuda_runtime.h>
#include "input_conv.h"
#include "test_blocks.h"

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
    bool identical = outputs_identical(output, out2);
    if (identical) {
        std::cout << "Outputs are identical." << std::endl;
    } else {
        std::cout << "Outputs are different." << std::endl;
    }
    return 0;
}
