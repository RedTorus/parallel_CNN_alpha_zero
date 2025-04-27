#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include "model.h"
#include "model_par.h"
//#include <ATen/ATen.h>

// Helper function to check if two outputs are identical within a tolerance.
bool outputs_identical(const torch::Tensor& out1, const torch::Tensor& out2, double tol = 1e-2) {
    return torch::allclose(out1, out2, tol);
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

// Check mean error of all elements in two tensors
float check_mean_error(const torch::Tensor& out1, const torch::Tensor& out2) {
    torch::Tensor diff = torch::abs(out2 - out1);
    float sum_err = diff.sum().item<float>();
    return sum_err / static_cast<float>(diff.numel());
}

int main() {
    // Set device

    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;

    // Create a full Model object (on CPU) with 5 input channels and 2 output channels (policy logits tensor).
    Model model_cpu(5, 2);
    model_cpu->eval();

    // Create a full Model object (on GPU) with 5 input channels and 2 output channels (policy logits tensor).
    Model model(5, 2);
    model->to(device);
    model->eval();

    // Create a customized full Model object with 5 input channels and 2 output channels (policy logits tensor).
    ModelPar model_par(5, 2);
    model_par->to(device);
    model_par->eval();

    // Create a dummy input tensor: batch size 1, channels, image.
    torch::Tensor input = torch::rand({1, 5, 8, 8});
    torch::Tensor input_cpu = input.clone();
    input = input.to(device);

    // Open files to output results
    std::ofstream time_file("output/model_execution_times.txt");
    if (!time_file) {
        std::cerr << "Failed to open model_execution_times.txt for writing" << std::endl;
        return -1;
    }
    std::ofstream error_file("output/model_mean_errors.txt");
    if (!error_file) {
        std::cerr << "Failed to open model_mean_errors.txt for writing" << std::endl;
        return -1;
    }

    // Create cuda event for timinf
    cudaEvent_t start_, stop_;
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    float gpu_baseline_time, gpu_par_time;

    int num_runs = 20;
    for (int run = 0; run < num_runs; ++run) {
        // Run forward pass of default model on CPU and measure time.
        auto cpu_start = std::chrono::high_resolution_clock::now();
        torch::Tensor output1 = model_cpu->forward(input_cpu);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        float cpu_exec_time = std::chrono::duration<float,std::milli>(cpu_end - cpu_start).count();
        std::cout << "Default model with Libtorch CNN on CPU took " << cpu_exec_time << " ms." << std::endl;

        // Run forward pass of default model on GPU and measure time.
        cudaEventRecord(start_, 0);
        torch::Tensor output2 = model->forward(input);
        cudaEventRecord(stop_, 0);
        cudaEventSynchronize(stop_);
        cudaEventElapsedTime(&gpu_baseline_time, start_, stop_);
        std::cout << "Default model with Libtorch CNN on GPU took " << gpu_baseline_time << " ms." << std::endl;

        // Run forward pass of model with customized parallelized CNN on GPU and measure time.
        cudaEventRecord(start_, 0);
        torch::Tensor output3 = model_par->forward(input);
        cudaEventRecord(stop_, 0);
        cudaEventSynchronize(stop_);
        cudaEventElapsedTime(&gpu_par_time, start_, stop_);
        std::cout << "Model with parallelized CNN took " << gpu_par_time << " ms." << std::endl;

        // Move back to CPU for comparison and output to files
        output2 = output2.to(torch::kCPU).contiguous();
        output3 = output3.to(torch::kCPU).contiguous();
        write_output(output1, "model_baseline_cpu.txt");
        write_output(output2, "model_baseline_gpu.txt");
        write_output(output3, "model_parallel.txt");

        // Check if the two outputs are identical.
        if (outputs_identical(output1, output3)) {
            std::cout << "The model outputs between baseline on CPU and parallel implementation are identical." << std::endl;
        } else {
            std::cout << "The model outputs between baseline on CPU and parallel implementation are different." << std::endl;
        }

        if (outputs_identical(output2, output3)) {
            std::cout << "The model outputs between baseline on GPU and parallel implementation are identical." << std::endl;
        } else {
            std::cout << "The model outputs between baseline on GPU and parallel implementation are different." << std::endl;
        }

        // Compute mean error between output3 and output2
        float mean_error = check_mean_error(output2, output1);
        std::cout << "Run " << run+1 << ": Mean error = " << mean_error << std::endl;

        // Write results to files
        time_file << cpu_exec_time << " " << gpu_baseline_time << " " << gpu_par_time << "\n";
        error_file << mean_error << "\n";
    }

    time_file.close();
    error_file.close();
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);

    return 0;
}