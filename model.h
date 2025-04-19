#ifndef MODEL_H
#define MODEL_H

#include <torch/torch.h>
#include <string>
#include <iostream>
#include "input_conv.h"

//Conv2dKernelBlock 

struct Conv2dKernelBlockImpl : public torch::nn::Module {
    //torch::nn::Conv2d conv{nullptr};

    torch::Tensor weight;
    // BatchNorm layer
    torch::nn::BatchNorm2d bn{nullptr};
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    Conv2dKernelBlockImpl(int in_channels_, int out_channels_, int kernel_size_ = 3,
                          int stride_ = 1, int padding_ = 1)
        : in_channels(in_channels_), out_channels(out_channels_), kernel_size(kernel_size_), stride(stride_), padding(padding_) {
        
        // Register custom weight tensor
        auto options = torch::TensorOptions()
                           .dtype(torch::kFloat32)
                           .device(torch::kCUDA);

        // Define weight shape as IntArrayRef
        std::vector<int64_t> shape = {
            int64_t(out_channels_),
            int64_t(in_channels_),
            int64_t(kernel_size_),
            int64_t(kernel_size_)
        };

        // Register and initialize weight
        weight = register_parameter(
            "weight",
            torch::zeros(c10::IntArrayRef(shape), options)
        );


        // Initialize weights (Kaiming Normal)
        torch::nn::init::kaiming_normal_(weight, /*a=*/0, torch::kFanOut, torch::kReLU);
        
        // BatchNorm
        bn = register_module(
            "bn",
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels))
        );
        bn->to(torch::kCUDA);
    }

    torch::Tensor forward(torch::Tensor x) {
        // Expect x of shape [B, C, H, W]
        TORCH_CHECK(x.dim() == 4 && x.size(1) == in_channels,
                    "Conv2dKernelBlockImpl expected input shape [B, ", in_channels,
                    ", H, W]");
        auto batch_size = x.size(0);

        // Call custom CUDA forward (expects batch size 1)
        //std::cout << "x shape: " << x.sizes() << "device: " << x.device() << std::endl;
        //std::cout << "weight shape: " << weight.sizes() << "device: " << weight.device() << std::endl;
        torch::Tensor conv_out = input_conv_forward(x, weight);
        // conv_out shape: [B, out_channels, H, W]
        //std::cout << "conv_out shape: " << conv_out.sizes() << " device: " << conv_out.device() << std::endl;
        // Apply BatchNorm and ReLU
        torch::Tensor y = bn(conv_out);
        //std::cout << "y shape: " << y.sizes() << " device: " << y.device() << std::endl;
        return torch::relu(y);
    }
};
TORCH_MODULE(Conv2dKernelBlock);
//---------------------------------------
// Conv2dBlockImpl: A simple conv2d block with BatchNorm and ReLU.
//---------------------------------------
struct Conv2dSimpleImpl : public torch::nn::Module {
    torch::nn::Conv2d conv{nullptr};

    // Constructor: in/out channels with optional kernel size, stride, and padding.
    Conv2dSimpleImpl(int in_channels, int out_channels, int kernel_size = 3, int stride = 1, int padding = 1) {
        conv = register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                                                         .stride(stride).padding(padding).bias(false)));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = conv(x);
        return x;
    }
};
TORCH_MODULE(Conv2dSimple);

//---------------------------------------
// Conv2dBlockImpl: A simple conv2d block with BatchNorm and ReLU.
//---------------------------------------
struct Conv2dBlockImpl : public torch::nn::Module {
    torch::nn::Conv2d conv{nullptr};
    torch::nn::BatchNorm2d bn{nullptr};

    // Constructor: in/out channels with optional kernel size, stride, and padding.
    Conv2dBlockImpl(int in_channels, int out_channels, int kernel_size = 3, int stride = 1, int padding = 1) {
        conv = register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                                                         .stride(stride).padding(padding)));
        bn = register_module("bn", torch::nn::BatchNorm2d(out_channels));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = conv(x);
        x = bn(x);
        return torch::relu(x);
    }
};
TORCH_MODULE(Conv2dBlock);

//---------------------------------------
// ResInputBlockImpl: initial block using Conv2dBlock.
//---------------------------------------
struct ResInputBlockImpl : public torch::nn::Module {
    Conv2dBlock conv_block{nullptr};

    // Constructor: in_channels, out_channels, kernel parameters.
    ResInputBlockImpl(int in_channels, int out_channels, int kernel_size = 3, int stride = 1, int padding = 1) {
        conv_block = register_module("conv_block", Conv2dBlock(in_channels, out_channels, kernel_size, stride, padding));
    }

    torch::Tensor forward(torch::Tensor x) {
        return conv_block->forward(x);
    }
};
TORCH_MODULE(ResInputBlock);

//---------------------------------------
// ResTorsoBlockImpl: intermediate block with residual connection,
// built using two Conv2dBlock layers.
//---------------------------------------
struct ResTorsoBlockImpl : public torch::nn::Module {
    Conv2dBlock conv_block1{nullptr};
    Conv2dBlock conv_block2{nullptr};

    ResTorsoBlockImpl(int channels, int kernel_size = 3, int stride = 1, int padding = 1) {
        conv_block1 = register_module("conv_block1", Conv2dBlock(channels, channels, kernel_size, stride, padding));
        conv_block2 = register_module("conv_block2", Conv2dBlock(channels, channels, kernel_size, stride, padding));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto residual = x.clone();
        x = conv_block1->forward(x);
        // Note: For the second block, we do not apply an extra ReLU since Conv2dBlock already applies it.
        x = conv_block2->conv(x);
        x = conv_block2->bn(x);
        // Add residual connection and then apply ReLU.
        x += residual;
        return torch::relu(x);
    }
};
TORCH_MODULE(ResTorsoBlock);

//---------------------------------------
// ResOutputBlockImpl: produces output from a residual branch.
//---------------------------------------
struct ResOutputBlockImpl : public torch::nn::Module {
    torch::nn::Linear fc{nullptr};

    ResOutputBlockImpl(int in_features, int num_classes) {
        fc = register_module("fc", torch::nn::Linear(in_features, num_classes));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = x.view({x.size(0), -1});
        return fc(x);
    }
};
TORCH_MODULE(ResOutputBlock);

//---------------------------------------
// MLPOutputBlockImpl: alternative MLP-based output block.
//---------------------------------------
struct MLPOutputBlockImpl : public torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    MLPOutputBlockImpl(int in_features, int hidden_features, int num_classes) {
        fc1 = register_module("fc1", torch::nn::Linear(in_features, hidden_features));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_features, num_classes));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = x.view({x.size(0), -1});
        x = torch::relu(fc1(x));
        return fc2(x);
    }
};
TORCH_MODULE(MLPOutputBlock);

//---------------------------------------
// ModelImpl: full model combining the above blocks.
//---------------------------------------
struct ModelImpl : public torch::nn::Module {
    ResInputBlock res_input{nullptr};
    ResTorsoBlock res_torso{nullptr};
    ResOutputBlock res_output{nullptr};
    // Alternatively, you could use:
    // MLPOutputBlock mlp_output{nullptr};

    ModelImpl(int in_channels, int num_classes) {
        // Initial block: from input channels to 64 feature maps.
        res_input = register_module("res_input", ResInputBlock(in_channels, 64, 3, 1, 1));
        // Torso block operating on 64 channels.
        res_torso = register_module("res_torso", ResTorsoBlock(64, 3, 1, 1));
        // Output block assumes feature map is downsampled to 64*56*56.
        res_output = register_module("res_output", ResOutputBlock(64 * 56 * 56, num_classes));

        // To use the MLP output block instead:
        // mlp_output = register_module("mlp_output", MLPOutputBlock(64 * 56 * 56, 128, num_classes));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = res_input->forward(x);
        x = res_torso->forward(x);
        x = res_output->forward(x);
        // Alternatively, if using MLP output:
        // x = mlp_output->forward(x);
        return x;
    }
};
TORCH_MODULE(Model);

//---------------------------------------
// Utility function: load checkpoint into the model.
//---------------------------------------
inline bool load_checkpoint(Model& model, const std::string& checkpoint_path) {
    try {
        torch::load(model, checkpoint_path);
        return true;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the checkpoint: " << e.msg() << std::endl;
        return false;
    }
}

#endif // MODEL_H

