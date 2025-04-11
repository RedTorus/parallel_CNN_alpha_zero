#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

// Declarations for custom CUDA operators.
torch::Tensor torso_conv_forward(torch::Tensor input, torch::Tensor filter);
torch::Tensor output_conv_forward(torch::Tensor input, torch::Tensor filter);

// Helper function to check if two outputs are identical within a tolerance.
bool outputs_identical(const torch::Tensor& out1, const torch::Tensor& out2, double tol = 1e-5) {
    return torch::allclose(out1, out2, tol);
}

bool outputs_same(const torch::Tensor& out1, const torch::Tensor& out2) {
    return torch::equal(out1, out2);
}

// Unified output function to write outputs to a file.
void write_output(const torch::Tensor& output_tensor, const std::string& filename) {
    std::ofstream outfile(filename);
    if (!outfile) {
        std::cerr << "Failed to open output file: " << filename << std::endl;
        return;
    }
    // Use scientific notation with 6 digits of precision.
    outfile << std::scientific << std::setprecision(6);
    
    // Write output.
    auto output_flat = output_tensor.view({-1});
    outfile << "Tensor Output (1x" << output_flat.size(0) << "):\n";
    for (int i = 0; i < output_flat.size(0); ++i) {
        outfile << output_flat[i].item<float>() << " ";
    }
    outfile << "\n";
    outfile.close();
}

int main() {
    // Set device
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;

    // Create cuda event for timinf
    cudaEvent_t start_, stop_;
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    float elapsed_gpu_;

    // Create a dummy input tensor: batch size 1, channels, image.
    torch::Tensor input = torch::rand({1, 128, 8, 8});
    input = input.to(device);

    // Run forward pass of libtorch CNN and measure time.
    torch::nn::Conv2d conv_layer(torch::nn::Conv2dOptions(128, 128, 3)
                                 .stride(1)
                                 .padding(1));
    conv_layer->weight.data().fill_(0.01);
    conv_layer->bias.data().fill_(0.0);
    conv_layer->to(device);
    cudaEventRecord(start_, 0);
    torch::Tensor output1 = conv_layer(input);
    cudaEventRecord(stop_, 0);
    cudaEventSynchronize(stop_);
    cudaEventElapsedTime(&elapsed_gpu_, start_, stop_);
    std::cout << "Libtorch CNN took " << elapsed_gpu_ << " ms." << std::endl;

    // Run forward pass of parallelized CNN and measure time.
    auto customized_filter = torch::full({128, 128, 3, 3}, 0.01, torch::TensorOptions().dtype(torch::kFloat).device(device));
    cudaEventRecord(start_, 0);
    torch::Tensor output2 = torso_conv_forward(input, customized_filter); // Returns shape [128, 8, 8] (batch dropped).
    output2 = output2.unsqueeze(0);                                  // Restore batch dim: [1, 128, 8, 8].
    cudaEventRecord(stop_, 0);
    cudaEventSynchronize(stop_);
    cudaEventElapsedTime(&elapsed_gpu_, start_, stop_);
    std::cout << "Parallelized CNN took " << elapsed_gpu_ << " ms." << std::endl;

    write_output(output1, "cnn_baseline.txt");
    write_output(output2, "cnn_parallel.txt");

    // Check if the two outputs are identical.
    if (outputs_identical(output1, output2)) {
        std::cout << "The network outputs are identical." << std::endl;
        if (outputs_same(output1, output2)) {
            std::cout << "The network outputs are the same." << std::endl;
        }
    } else {
        std::cout << "The network outputs are different." << std::endl;
    }

    return 0;
}