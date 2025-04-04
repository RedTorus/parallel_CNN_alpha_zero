#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>

// Helper functions for indexing a 3D tensor (channels, height, width)
// and a 4D tensor (out_channels, in_channels, kernel_height, kernel_width).
inline int idx3d(int c, int h, int w, int H, int W) {
    return c * H * W + h * W + w;
}
inline int idx4d(int oc, int ic, int kh, int kw, int in_channels, int kernel_size) {
    return oc * (in_channels * kernel_size * kernel_size) + ic * (kernel_size * kernel_size) + kh * kernel_size + kw;
}

// ReLU activation
inline float relu(float x) {
    return (x > 0) ? x : 0;
}

// 2D convolution for a single batch.
// Assumes stride=1. For a 3x3 kernel, padding=1 gives same output dimensions.
void conv2d(const float* input, int in_channels, int in_height, int in_width,
            const float* filter, int out_channels, int kernel_size, int padding,
            float* output) {
    int out_height = in_height; // assuming same padding
    int out_width = in_width;
    for (int oc = 0; oc < out_channels; oc++) {
        for (int h = 0; h < out_height; h++) {
            for (int w = 0; w < out_width; w++) {
                float sum = 0.0f;
                for (int ic = 0; ic < in_channels; ic++) {
                    for (int kh = 0; kh < kernel_size; kh++) {
                        for (int kw = 0; kw < kernel_size; kw++) {
                            int in_h = h + kh - padding;
                            int in_w = w + kw - padding;
                            if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                                int input_index = idx3d(ic, in_h, in_w, in_height, in_width);
                                int filter_index = idx4d(oc, ic, kh, kw, in_channels, kernel_size);
                                sum += input[input_index] * filter[filter_index];
                            }
                        }
                    }
                }
                // Here we apply ReLU activation after the convolution.
                output[idx3d(oc, h, w, out_height, out_width)] = relu(sum);
            }
        }
    }
}

// Fully connected (linear) layer.
// Computes: output[j] = sum(input[i] * weight[j * in_features + i])
void fully_connected(const float* input, const float* weight, int in_features, int out_features, float* output) {
    for (int o = 0; o < out_features; o++) {
        float sum = 0.0f;
        for (int i = 0; i < in_features; i++) {
            sum += input[i] * weight[o * in_features + i];
        }
        output[o] = sum;
    }
}

// Projection: reshape the input [1,320] to a tensor of size [1,5,8,8].
// (Since 5*8*8 == 320, a simple copy is sufficient.)
void project_input(const float* input, float* output) {
    for (int i = 0; i < 320; i++) {
        output[i] = input[i];
    }
}

// Input layer:
// 1. Projects the 2-D input [1,320] to a tensor [1,5,8,8].
// 2. Applies a 3x3 convolution (input channels = 5, output channels = 128, padding=1)
//    to produce an output tensor of shape [1,128,8,8].
void input_layer(const float* input_2d, float* output_tensor, const float* conv_filter) {
    float projected[320]; // same number of elements as [1,5,8,8]
    project_input(input_2d, projected);
    // Apply convolution: input channels = 5, output channels = 128, kernel=3, padding=1.
    conv2d(projected, 5, 8, 8, conv_filter, 128, 3, 1, output_tensor);
}

// A torso block consists of two convolution operations with a residual connection.
// Input and output are tensors of shape [128,8,8].
void torso_block(float* tensor, const float* conv_filter1, const float* conv_filter2) {
    const int channels = 128, height = 8, width = 8;
    int tensor_size = channels * height * width;
    std::vector<float> conv1_out(tensor_size, 0.0f);
    std::vector<float> conv2_out(tensor_size, 0.0f);

    // First convolution in the block.
    conv2d(tensor, channels, height, width, conv_filter1, channels, 3, 1, conv1_out.data());
    // Second convolution.
    conv2d(conv1_out.data(), channels, height, width, conv_filter2, channels, 3, 1, conv2_out.data());
    // Add the residual connection and apply ReLU.
    for (int i = 0; i < tensor_size; i++) {
        tensor[i] = relu(tensor[i] + conv2_out[i]);
    }
}

// Torso layer: Contains 5 torso blocks.
// Each block uses its own pair of convolution filters.
void torso_layer(float* tensor,
                 const std::vector<const float*>& torso_filters1,
                 const std::vector<const float*>& torso_filters2) {
    for (int block = 0; block < 5; block++) {
        torso_block(tensor, torso_filters1[block], torso_filters2[block]);
    }
}

// Output layer:
// 1. Value head: applies a 1x1 convolution (input channels = 128, output channels = 1)
//    to yield [1,1,8,8], then flattens (64 elements) and projects to [1,1] via a fully connected layer.
//    A tanh activation is applied to the final value.
// 2. Policy head: applies a 1x1 convolution (input channels = 128, output channels = 2)
//    to yield [1,2,8,8], then flattens (128 elements) and projects to [1,512] via a fully connected layer.
void output_layer(const float* tensor,
                  const float* value_conv_filter, const float* policy_conv_filter,
                  const float* value_fc_weight, const float* policy_fc_weight,
                  float* value_output, float* policy_output) {
    const int in_channels = 128, height = 8, width = 8;
    
    // --- Value head ---
    const int value_out_channels = 1; // after 1x1 conv: [1,1,8,8]
    int value_conv_size = value_out_channels * height * width; // 1*8*8 = 64
    std::vector<float> value_conv_out(value_conv_size, 0.0f);
    // For 1x1 convolution, no padding is needed.
    conv2d(tensor, in_channels, height, width, value_conv_filter, value_out_channels, 1, 0, value_conv_out.data());
    // Fully connected layer: maps 64 -> 1.
    float value_fc_out[1] = {0.0f};
    fully_connected(value_conv_out.data(), value_fc_weight, value_conv_size, 1, value_fc_out);
    // Apply tanh activation.
    value_output[0] = std::tanh(value_fc_out[0]);
    
    // --- Policy head ---
    const int policy_out_channels = 2; // after 1x1 conv: [1,2,8,8]
    int policy_conv_size = policy_out_channels * height * width; // 2*8*8 = 128
    std::vector<float> policy_conv_out(policy_conv_size, 0.0f);
    conv2d(tensor, in_channels, height, width, policy_conv_filter, policy_out_channels, 1, 0, policy_conv_out.data());
    // Fully connected layer: maps 128 -> 512.
    fully_connected(policy_conv_out.data(), policy_fc_weight, policy_conv_size, 512, policy_output);
}

int main() {
    // ----- Simulated Input -----
    // Create a 2-D input of size [1,320]. Here we simply use normalized values.
    float input_2d[320];
    for (int i = 0; i < 320; i++) {
        input_2d[i] = static_cast<float>(i) / 320.0f;
    }
    
    // ----- Input Layer Setup -----
    // Initialize the filter for the input layer convolution.
    // Filter shape: [128, 5, 3, 3]
    const int input_conv_filter_size = 128 * 5 * 3 * 3;
    std::vector<float> input_conv_filter(input_conv_filter_size, 0.01f);  // constant value for testing
    
    // Output tensor of the input layer: shape [128,8,8]
    const int input_layer_output_size = 128 * 8 * 8;
    std::vector<float> input_layer_output(input_layer_output_size, 0.0f);
    input_layer(input_2d, input_layer_output.data(), input_conv_filter.data());
    
    // ----- Torso Layer Setup -----
    // For each of the 5 torso blocks, we initialize two convolution filters.
    // Each filter has shape: [128, 128, 3, 3]
    const int torso_filter_size = 128 * 128 * 3 * 3;
    std::vector< std::vector<float> > torso_filters1(5, std::vector<float>(torso_filter_size, 0.01f));
    std::vector< std::vector<float> > torso_filters2(5, std::vector<float>(torso_filter_size, 0.01f));
    // Prepare pointers for ease of use.
    std::vector<const float*> torso_filters1_ptr(5), torso_filters2_ptr(5);
    for (int b = 0; b < 5; b++) {
        torso_filters1_ptr[b] = torso_filters1[b].data();
        torso_filters2_ptr[b] = torso_filters2[b].data();
    }
    // Process the torso layer.
    torso_layer(input_layer_output.data(), torso_filters1_ptr, torso_filters2_ptr);
    
    // ----- Output Layer Setup -----
    // Value head:
    // Convolution filter: shape [1, 128, 1, 1]
    const int value_conv_filter_size = 1 * 128 * 1 * 1;
    std::vector<float> value_conv_filter(value_conv_filter_size, 0.01f);
    // Fully connected layer weight: shape [1, 64] (maps flattened [1,1,8,8] -> [1,1])
    const int value_fc_weight_size = 1 * 64;
    std::vector<float> value_fc_weight(value_fc_weight_size, 0.01f);
    
    // Policy head:
    // Convolution filter: shape [2, 128, 1, 1]
    const int policy_conv_filter_size = 2 * 128 * 1 * 1;
    std::vector<float> policy_conv_filter(policy_conv_filter_size, 0.01f);
    // Fully connected layer weight: shape [512, 128] (maps flattened [1,2,8,8] -> [1,512])
    const int policy_fc_weight_size = 512 * 128;
    std::vector<float> policy_fc_weight(policy_fc_weight_size, 0.01f);
    
    // Output containers.
    float value_output[1] = {0.0f};
    std::vector<float> policy_output(512, 0.0f);
    
    // Run the output layer.
    output_layer(input_layer_output.data(),
                 value_conv_filter.data(), policy_conv_filter.data(),
                 value_fc_weight.data(), policy_fc_weight.data(),
                 value_output, policy_output.data());
    
    // ----- Write Final Results to File -----
    std::ofstream outfile("output.txt");
    if (!outfile.is_open()) {
        std::cerr << "Error opening output file!" << std::endl;
        return 1;
    }
    outfile << "Value Output (1x1):" << std::endl;
    outfile << value_output[0] << std::endl;
    outfile << "Policy Output (1x512):" << std::endl;
    for (int i = 0; i < 512; i++) {
        outfile << policy_output[i] << " ";
    }
    outfile << std::endl;
    outfile.close();
    
    std::cout << "Computation complete. Results written to output.txt" << std::endl;
    return 0;
}
