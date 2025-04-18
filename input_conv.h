#ifndef INPUT_CONV_H
#define INPUT_CONV_H

#include <torch/extension.h>

// Forward declaration of the CUDA convolution launcher
// Defined in input_conv.cu

/**
 * input_conv_forward
 * @param input         A 4D CUDA tensor of shape [1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH]
 * @param conv_weights  A 4D CUDA tensor of shape [OUTPUT_CHANNELS, INPUT_CHANNELS, KERNEL_SIZE, KERNEL_SIZE]
 * @returns             A CUDA tensor of shape [OUTPUT_CHANNELS, OUTPUT_HEIGHT, OUTPUT_WIDTH]
 */
torch::Tensor input_conv_forward(
    torch::Tensor input,
    torch::Tensor conv_weights
);

#endif // INPUT_CONV_H
