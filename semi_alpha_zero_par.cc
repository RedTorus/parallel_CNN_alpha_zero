#include <torch/torch.h>
#include <iostream>
#include <fstream>

// Declarations for custom CUDA operators.
torch::Tensor torso_conv_forward(torch::Tensor input, torch::Tensor filter);
torch::Tensor output_conv_forward(torch::Tensor input, torch::Tensor filter);

// Unified output function: writes the value and policy outputs with fixed precision.
void write_output(const torch::Tensor& value_tensor, const torch::Tensor& policy_tensor, const std::string& filename) {
    std::ofstream outfile(filename);
    if (!outfile) {
        std::cerr << "Failed to open output file: " << filename << std::endl;
        return;
    }
    // Use scientific notation with 6 digits of precision.
    outfile << std::scientific << std::setprecision(6);
    
    // Write value output.
    auto value_flat = value_tensor.view({-1});
    outfile << "Value Output (1x" << value_flat.size(0) << "):\n";
    for (int i = 0; i < value_flat.size(0); ++i) {
        outfile << value_flat[i].item<float>() << " ";
    }
    outfile << "\n";
    
    // Write policy output.
    auto policy_flat = policy_tensor.view({-1});
    outfile << "Policy Output (1x" << policy_flat.size(0) << "):\n";
    for (int i = 0; i < policy_flat.size(0); ++i) {
        outfile << policy_flat[i].item<float>() << " ";
    }
    outfile << "\n";
    outfile.close();
}

int main() {
    // Set device to CUDA.
    torch::Device device(torch::kCUDA);

    // --------------------------------------------------------------------
    // 1. Input Layer (Non-Parallel)
    // --------------------------------------------------------------------
    // Simulated 2-D input: [1, 320].
    // Create a 2-D input tensor of size [1,320] with linearly spaced values.
    auto input_tensor = torch::linspace(0, 1, 320).view({1, 320});
    // x is a 2-D tensor of shape [1,320]. Reshape it to [1,5,8,8].
    input_tensor = input_tensor.view({1, 5, 8, 8}).to(device);

    // Input CNN: A non-parallel convolution: Conv2d mapping 5 -> 128 channels with 3x3 kernel and padding=1.
    auto input_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(5, 128, 3).stride(1).padding(1));
    input_conv->weight.data().fill_(0.01);
    input_conv->bias.data().fill_(0.0);
    input_conv->to(device);
    
    auto input_layer_output = torch::relu(input_conv->forward(input_tensor)); // Shape: [1, 128, 8, 8].

    // --------------------------------------------------------------------
    // 2. Torso Layer (Parallel using custom CUDA op)
    // --------------------------------------------------------------------
    // The torso layer has 5 blocks; each block has 2 CNNs.
    // Prepare a torso filter for all CNNs: shape [128, 128, 3, 3], constant value 0.01.
    auto torso_filter = torch::full({128, 128, 3, 3}, 0.01, torch::TensorOptions().dtype(torch::kFloat).device(device));

    // Initialize torso input.
    torch::Tensor x = input_layer_output; // Shape: [1, 128, 8, 8].

    // BatchNorm2d to be used
    torch::nn::BatchNorm2d batch_norm(128);
    batch_norm->to(device);

    // For each torso block:
    for (int i = 0; i < 5; i++) {
        auto residual = x.clone();

        // First CNN in block.
        auto y = torso_conv_forward(x, torso_filter); // Returns shape [128, 8, 8] (batch dropped).
        y = y.unsqueeze(0);                           // Restore batch dim: [1, 128, 8, 8].
        y = torch::relu(batch_norm(y));
        // Second CNN in block.
        auto z = torso_conv_forward(y, torso_filter);  // Returns shape [128, 8, 8].
        z = z.unsqueeze(0);                           // Update x for next block: [1, 128, 8, 8].
        z = batch_norm(z);

        x = torch::relu(residual + z);
    }
    // Final torso output: x of shape [1, 128, 8, 8].

    // --------------------------------------------------------------------
    // 3. Output Layer (Parallel + Projection)
    // --------------------------------------------------------------------
    // 3a. Value convolution.
    // Prepare value_conv filter: shape [1, 128, 1, 1], constant value 0.01.
    auto value_filter = torch::full({1, 128, 1, 1}, 0.01, torch::TensorOptions().dtype(torch::kFloat).device(device));
    auto value_conv_out = output_conv_forward(x, value_filter); // Returns shape [1, 8, 8].
    // Restore batch dimension.
    value_conv_out = value_conv_out.unsqueeze(0); // Shape: [1, 1, 8, 8].
    // Flatten to [1, 64].
    auto value_final = torch::relu(value_conv_out);
    value_final = value_final.view({1, -1}); // 1*8*8 = 64.
    /*
    // Linear projection: from 64 -> 1.
    torch::nn::Linear value_linear(8 * 8, 1);
    value_linear->to(device);
    value_linear->weight.data().fill_(0.01);
    value_linear->bias.data().fill_(0.0);
    
    value_final = torch::relu(value_linear->forward(value_final)); // Shape: [1, 1].
    value_final = torch::tanh(value_linear->forward(value_final));
    */

    // 3b. Policy convolution.
    // Prepare policy_conv filter: shape [2, 128, 1, 1], constant value 0.01.
    auto policy_filter = torch::full({2, 128, 1, 1}, 0.01, torch::TensorOptions().dtype(torch::kFloat).device(device));
    auto policy_conv_out = output_conv_forward(x, policy_filter); // Returns shape [2, 8, 8].
    policy_conv_out = policy_conv_out.unsqueeze(0); // Shape: [1, 2, 8, 8].
    // Flatten to [1, 128] (since 2*8*8 = 128).
    auto policy_final = torch::relu(policy_conv_out);
    policy_final = policy_final.view({1, -1});
    /*
    // Linear projection: from 128 -> 512.
    torch::nn::Linear policy_linear(2 * 8 * 8, 512);
    policy_linear->to(device);
    policy_linear->weight.data().fill_(0.01);
    policy_linear->bias.data().fill_(0.0);
    
    policy_final = policy_linear->forward(policy_final); // Shape: [1, 512].
    */

    // --------------------------------------------------------------------
    // 4. Write final outputs to file.
    // --------------------------------------------------------------------
    // Write the outputs to file with unified formatting.
    write_output(value_final, policy_final, "output_par.txt");
    std::cout << "Processing complete. Final outputs written to output_par.txt" << std::endl;
    return 0;
}
