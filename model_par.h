#ifndef MODEL_PAR_H
#define MODEL_PAR_H

#include <torch/torch.h>
#include <string>
#include <iostream>

// Declarations for custom CUDA operators.
torch::Tensor torso_conv_forward(torch::Tensor input, torch::Tensor filter);
torch::Tensor output_conv_forward(torch::Tensor input, torch::Tensor filter);

//---------------------------------------
// Conv2dBlockParImpl: A simple conv2d block with BatchNorm and ReLU.
//---------------------------------------
struct Conv2dBlockParImpl : public torch::nn::Module {
    torch::nn::Conv2d conv{nullptr};
    torch::nn::BatchNorm2d bn{nullptr};

    // Constructor: in/out channels with optional kernel size, stride, and padding.
    Conv2dBlockParImpl(int in_channels, int out_channels, int kernel_size = 3, int stride = 1, int padding = 1) {
        conv = register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                                                         .stride(stride).padding(padding)));
        bn = register_module("bn", torch::nn::BatchNorm2d(out_channels));

        // Override the default weight initialization:
        // Fill all elements of the convolution filter with 0.01.
        conv->weight.data().fill_(0.01);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = conv(x);
        x = bn(x);
        return torch::relu(x);
    }
};
TORCH_MODULE(Conv2dBlockPar);

//---------------------------------------
// ResInputBlockParImpl: initial block using Conv2dBlockPar.
//---------------------------------------
struct ResInputBlockParImpl : public torch::nn::Module {
    Conv2dBlockPar conv_block{nullptr};

    // Constructor: in_channels, out_channels, kernel parameters.
    ResInputBlockParImpl(int in_channels, int out_channels, int kernel_size = 3, int stride = 1, int padding = 1) {
        conv_block = register_module("conv_block", Conv2dBlockPar(in_channels, out_channels, kernel_size, stride, padding));
    }

    torch::Tensor forward(torch::Tensor x) {
        return conv_block->forward(x);
    }
};
TORCH_MODULE(ResInputBlockPar);

//---------------------------------------
// ResTorsoBlockParImpl: intermediate block with residual connection,
// built using two Conv2dBlockPar layers.
//---------------------------------------
struct ResTorsoBlockParImpl : public torch::nn::Module {
    Conv2dBlockPar conv_block{nullptr};
    torch::Tensor filter;

    ResTorsoBlockParImpl(int channels, int kernel_size = 3, int stride = 1, int padding = 1) {
        conv_block = register_module("conv_block", Conv2dBlockPar(channels, channels, kernel_size, stride, padding));
        filter = register_parameter("filter", torch::full({channels, channels, kernel_size, kernel_size}, 0.01, torch::TensorOptions().dtype(torch::kFloat)));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto residual = x.clone();
        x = torso_conv_forward(x, filter);
        //x = x.unsqueeze(0); // Restore batch dim
        x = torch::relu(conv_block->bn(x));
        // Note: For the second block, we do not apply an extra ReLU since Conv2dBlockPar already applies it.
        x = torso_conv_forward(x, filter);
        //x = x.unsqueeze(0); // Restore batch dim
        x = conv_block->bn(x);
        // Add residual connection and then apply ReLU.
        x += residual;
        return torch::relu(x);
    }
};
TORCH_MODULE(ResTorsoBlockPar);

//---------------------------------------
// ResOutputBlockParImpl: produces output from a residual branch.
//---------------------------------------
struct ResOutputBlockParImpl : public torch::nn::Module {
    Conv2dBlockPar conv_block_output{nullptr};
    torch::Tensor filter;

    ResOutputBlockParImpl(int in_channels, int out_channels, int kernel_size = 1, int stride = 1, int padding = 0) {
        conv_block_output = register_module("conv_block_output", Conv2dBlockPar(in_channels, out_channels, kernel_size, stride, padding));
        filter = register_parameter("filter", torch::full({out_channels, in_channels, kernel_size, kernel_size}, 0.01, torch::TensorOptions().dtype(torch::kFloat)));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = output_conv_forward(x, filter);
        //x = x.unsqueeze(0); // Restore batch dim
        x = torch::relu(conv_block_output->bn(x));
        x = x.view({x.size(0), -1});
        return x;
    }
};
TORCH_MODULE(ResOutputBlockPar);

//---------------------------------------
// ModelParImpl: full model combining the above blocks.
//---------------------------------------
struct ModelParImpl : public torch::nn::Module {
    ResInputBlockPar res_input{nullptr};
    ResTorsoBlockPar res_torso{nullptr};
    ResOutputBlockPar res_output{nullptr};

    ModelParImpl(int in_channels, int out_channels) {
        // Initial block: from input channels to 64 feature maps.
        res_input = register_module("res_input", ResInputBlockPar(in_channels, 128, 3, 1, 1));
        // Torso block operating on 64 channels.
        res_torso = register_module("res_torso", ResTorsoBlockPar(128, 3, 1, 1));
        // Output block assumes feature map is downsampled to 64*56*56.
        res_output = register_module("res_output", ResOutputBlockPar(128, out_channels, 1, 1, 0));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = res_input->forward(x);
        x = res_torso->forward(x);
        x = res_output->forward(x);
        return x;
    }
};
TORCH_MODULE(ModelPar);

#endif // MODEL_H
