#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

// Declarations for custom CUDA operators.
torch::Tensor input_conv_forward(torch::Tensor input, torch::Tensor conv_weights);
torch::Tensor torso_conv_forward(torch::Tensor input, torch::Tensor filter);
torch::Tensor output_conv_forward(torch::Tensor input, torch::Tensor filter);

// Helper function to check if two outputs are identical within a tolerance.
bool outputs_identical(const torch::Tensor& out1, const torch::Tensor& out2, double tol = 1e-5) {
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

float check_mean_error(const torch::Tensor& out1, const torch::Tensor& out2) {
    torch::Tensor diff = torch::abs(out1 - out2);
    torch::Tensor mean_err_tensor = diff.mean();
    float mean_error = mean_err_tensor.item<float>();
    return mean_error;
}

int main() {
    // Set device
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;

    // Open files to output results
    std::ofstream time_file("output/cnn_out_execution_times.txt");
    if (!time_file) {
        std::cerr << "Failed to open cnn_out_execution_times.txt for writing" << std::endl;
        return -1;
    }
    std::ofstream error_file("output/cnn_out_mean_errors.txt");
    if (!error_file) {
        std::cerr << "Failed to open cnn_out_mean_errors.txt for writing" << std::endl;
        return -1;
    }

    // Create a dummy input tensor: batch size 1, channels, image.
    torch::Tensor input = torch::rand({1, 128, 8, 8});
    torch::Tensor input_cpu = input.clone();
    input = input.to(device);

    // Create cuda event for timinf
    cudaEvent_t start_, stop_;
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    float gpu_baseline_time, gpu_par_time;

    int num_runs = 20;
    for (int run = 0; run < num_runs; ++run) {
        // Run forward pass of libtorch CNN on CPU and measure time.
        torch::nn::Conv2d conv_layer_cpu(torch::nn::Conv2dOptions(128, 2, 1)
                                 .stride(1)
                                 .padding(0));
        conv_layer_cpu->weight.data().fill_(0.01);
        conv_layer_cpu->bias.data().fill_(0.0);
        auto cpu_start = std::chrono::high_resolution_clock::now();
        torch::Tensor output1 = conv_layer_cpu(input_cpu);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        float cpu_exec_time = std::chrono::duration<float,std::milli>(cpu_end - cpu_start).count();
        std::cout << "Libtorch CNN on CPU took " << cpu_exec_time << " ms." << std::endl;

        // Run forward pass of libtorch CNN on GPU and measure time.
        torch::nn::Conv2d conv_layer_gpu(torch::nn::Conv2dOptions(128, 2, 1)
                                 .stride(1)
                                 .padding(0));
        conv_layer_gpu->weight.data().fill_(0.01);
        conv_layer_gpu->bias.data().fill_(0.0);
        conv_layer_gpu->to(device);
        cudaEventRecord(start_, 0);
        torch::Tensor output2 = conv_layer_gpu(input);
        cudaEventRecord(stop_, 0);
        cudaEventSynchronize(stop_);
        cudaEventElapsedTime(&gpu_baseline_time, start_, stop_);
        std::cout << "Libtorch CNN on GPU took " << gpu_baseline_time << " ms." << std::endl;

        // Run forward pass of parallelized CNN on GPU and measure time.
        auto customized_filter = torch::full({2, 128, 1, 1}, 0.01, torch::TensorOptions().dtype(torch::kFloat).device(device));
        cudaEventRecord(start_, 0);
        torch::Tensor output3 = output_conv_forward(input, customized_filter); // Returns shape [1, 128, 8, 8]
        cudaEventRecord(stop_, 0);
        cudaEventSynchronize(stop_);
        cudaEventElapsedTime(&gpu_par_time, start_, stop_);
        std::cout << "Parallelized CNN on GPU took " << gpu_par_time << " ms." << std::endl;

        // Move back to CPU for comparison and output to files
        output2 = output2.to(torch::kCPU).contiguous();
        output3 = output3.to(torch::kCPU).contiguous();
        write_output(output1, "cnn_baseline_cpu.txt");
        write_output(output2, "cnn_baseline_gpu.txt");
        write_output(output3, "cnn_parallel.txt");

        // Check if the two outputs are identical.
        if (outputs_identical(output1, output3)) {
            std::cout << "The network outputs between baseline on CPU and parallel implementation are identical." << std::endl;
        } else {
            std::cout << "The network outputs between baseline on CPU and parallel implementation are different." << std::endl;
        }

        if (outputs_identical(output2, output3)) {
            std::cout << "The network outputs between baseline on GPU and parallel implementation are identical." << std::endl;
        } else {
            std::cout << "The network outputs between baseline on GPU and parallel implementation are different." << std::endl;
        }

        // Compute mean errors
        float mean_err_cpu_gpu = check_mean_error(output1, output2);
        float mean_err_cpu_custom = check_mean_error(output1, output3);
        float mean_err_gpu_custom = check_mean_error(output2, output3);

        // Write all results to files
        time_file << cpu_exec_time << " " << gpu_baseline_time << " " << gpu_par_time << "\n";
        error_file << mean_err_cpu_gpu << " " << mean_err_cpu_custom << " " << mean_err_gpu_custom << "\n";
    }

    time_file.close();
    error_file.close();
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);

    return 0;
}