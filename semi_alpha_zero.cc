#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <utility>

// Define a custom model that implements the three layers.
struct SemiProgramModelImpl : torch::nn::Module {
    // Input layer: a 3x3 convolution (from 5 to 128 channels)
    torch::nn::Conv2d input_conv{nullptr};

    // Torso layer: 5 torso blocks; each block is a sequential module of:
    // Conv2d (3x3) -> BatchNorm2d -> ReLU -> Conv2d (3x3) -> BatchNorm2d.
    // The residual addition and final ReLU are applied in the forward().
    std::vector<torch::nn::Sequential> torso_blocks;

    // Output layer: two branches (value and policy).
    // Value head: 1x1 conv (128->1) and a linear layer mapping flattened [1,1,8,8] (64 dims) to [1,1].
    torch::nn::Conv2d value_conv{nullptr};
    torch::nn::Linear value_fc{nullptr};

    // Policy head: 1x1 conv (128->2) and a linear layer mapping flattened [1,2,8,8] (128 dims) to [1,512].
    torch::nn::Conv2d policy_conv{nullptr};
    torch::nn::Linear policy_fc{nullptr};

    SemiProgramModelImpl() {
        // ----- Input Layer -----
        // Create a conv layer with kernel size 3, padding=1.
        input_conv = register_module("input_conv", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(5, 128, /*kernel_size=*/3).padding(1)));
        input_conv->weight.data().fill_(0.01);
        input_conv->bias.data().fill_(0.0);

        // ----- Torso Layer -----
        // Create 5 torso blocks.
        for (int i = 0; i < 5; i++) {
            torch::nn::Sequential block(
                // First convolution + BN + ReLU.
                torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, /*kernel_size=*/3).padding(1)),
                torch::nn::BatchNorm2d(128),
                torch::nn::ReLU(),
                // Second convolution + BN.
                torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, /*kernel_size=*/3).padding(1)),
                torch::nn::BatchNorm2d(128)
            );
            // Initialize convolution weights in the block.
            for (const auto& module : block->children()) {
                if (auto conv = dynamic_cast<torch::nn::Conv2dImpl*>(module.get())) {
                    conv->weight.data().fill_(0.01);
                    conv->bias.data().fill_(0.0);
                }
            }
            torso_blocks.push_back(register_module("torso_block_" + std::to_string(i), block));
        }

        // ----- Output Layer -----
        // Value head: 1x1 conv to get [1,1,8,8]
        value_conv = register_module("value_conv", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(128, 1, /*kernel_size=*/1)));
        value_conv->weight.data().fill_(0.01);
        value_conv->bias.data().fill_(0.0);
        // Linear layer: flatten [1,1,8,8] to a vector of 64 elements and map to 1 output.
        value_fc = register_module("value_fc", torch::nn::Linear(8 * 8, 1));
        value_fc->weight.data().fill_(0.01);
        value_fc->bias.data().fill_(0.0);

        // Policy head: 1x1 conv to get [1,2,8,8]
        policy_conv = register_module("policy_conv", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(128, 2, /*kernel_size=*/1)));
        policy_conv->weight.data().fill_(0.01);
        policy_conv->bias.data().fill_(0.0);
        // Linear layer: flatten [1,2,8,8] to a vector of 128 elements and map to 512 outputs.
        policy_fc = register_module("policy_fc", torch::nn::Linear(2 * 8 * 8, 512));
        policy_fc->weight.data().fill_(0.01);
        policy_fc->bias.data().fill_(0.0);
    }

    // Forward pass returns a pair: (value output, policy output)
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        // ----- Input Layer -----
        // x is a 2-D tensor of shape [1,320]. Reshape it to [1,5,8,8].
        x = x.view({1, 5, 8, 8});
        // Apply the input convolution and ReLU activation.
        auto out = torch::relu(input_conv->forward(x));

        // ----- Torso Layer -----
        // Process through 5 torso blocks.
        for (auto& block : torso_blocks) {
            auto residual = out.clone();
            auto block_out = block->forward(out);
            // Add the residual connection and apply ReLU.
            out = torch::relu(residual + block_out);
        }

        // ----- Output Layer -----
        // Value Head:
        auto v = torch::relu(value_conv->forward(out));  // [1,1,8,8]
        v = v.view({1, -1});                              // Flatten to [1,64]
        v = value_fc->forward(v);                         // Fully connected to [1,1]
        v = torch::tanh(v);                               // Apply tanh

        // Policy Head:
        auto p = torch::relu(policy_conv->forward(out));  // [1,2,8,8]
        p = p.view({1, -1});                              // Flatten to [1,128]
        p = policy_fc->forward(p);                        // Fully connected to [1,512]

        return std::make_pair(v, p);
    }
};
TORCH_MODULE(SemiProgramModel);

int main() {
    // ----- Device Setup -----
    // Use CUDA if available; otherwise, fall back to CPU.
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;

    // ----- Instantiate the Model -----
    auto model = SemiProgramModel();
    model->to(device);
    model->eval();  // Set to evaluation mode

    // ----- Simulated Input -----
    // Create a 2-D input tensor of size [1,320] with linearly spaced values.
    auto input_tensor = torch::linspace(0, 1, 320).to(device).view({1, 320});

    // ----- Forward Pass -----
    auto outputs = model->forward(input_tensor);
    auto value_output = outputs.first;   // Tensor of shape [1,1]
    auto policy_output = outputs.second; // Tensor of shape [1,512]

    // Move results back to CPU for output.
    value_output = value_output.to(torch::kCPU);
    policy_output = policy_output.to(torch::kCPU);

    // ----- Write Final Results to File -----
    std::ofstream outfile("output_libtorch.txt");
    if (!outfile.is_open()) {
        std::cerr << "Error opening output file!" << std::endl;
        return 1;
    }
    outfile << "Value Output (1x1): " << value_output.item<float>() << "\n";
    outfile << "Policy Output (1x512): " << policy_output << "\n";
    outfile.close();

    std::cout << "Computation complete. Results written to output_libtorch.txt" << std::endl;
    return 0;
}
