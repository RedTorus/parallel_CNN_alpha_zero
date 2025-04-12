#include <torch/torch.h>
#include <iostream>
#include <cuda_runtime.h>
#include "model.h"
#include "model_par.h"

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

    // Create a full Model object with 5 input channels and 2 output channels (policy logits tensor).
    Model model(5, 2);
    model->to(device);
    model->eval();

    // Create a customized full Model object with 5 input channels and 2 output channels (policy logits tensor).
    ModelPar model_par(5, 2);
    model_par->to(device);
    model_par->eval();

    // Create a dummy input tensor: batch size 1, channels, image.
    torch::Tensor input = torch::rand({1, 5, 8, 8});
    input = input.to(device);

    // Create cuda event for timinf
    cudaEvent_t start_, stop_;
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    float elapsed_gpu_;

    // Run forward pass of default model and measure time.
    cudaEventRecord(start_, 0);
    torch::Tensor output1 = model->forward(input);
    cudaEventRecord(stop_, 0);
    cudaEventSynchronize(stop_);
    cudaEventElapsedTime(&elapsed_gpu_, start_, stop_);
    std::cout << "Default model with Libtorch CNN took " << elapsed_gpu_ << " ms." << std::endl;

    // Run forward pass of model with customized parallelized CNN and measure time.
    cudaEventRecord(start_, 0);
    torch::Tensor output2 = model_par->forward(input);
    cudaEventRecord(stop_, 0);
    cudaEventSynchronize(stop_);
    cudaEventElapsedTime(&elapsed_gpu_, start_, stop_);
    std::cout << "Model with parallelized CNN took " << elapsed_gpu_ << " ms." << std::endl;

    write_output(output1, "model_baseline.txt");
    write_output(output2, "model_parallel.txt");

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