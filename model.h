#ifndef MODEL_H
#define MODEL_H

#include <torch/torch.h>
#include <string>
#include <iostream>

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

        // Override the default weight and bias initialization:
        // Fill all elements of the convolution filter with 0.01 and set bias to false.
        conv->weight.data().fill_(0.01);
        conv->bias.data().fill_(0.0);
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
    Conv2dBlock conv_block_output{nullptr};

    ResOutputBlockImpl(int in_channels, int out_channels, int kernel_size = 1, int stride = 1, int padding = 0) {
        conv_block_output = register_module("conv_block_output", Conv2dBlock(in_channels, out_channels, kernel_size, stride, padding));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = conv_block_output->forward(x);
        x = x.view({x.size(0), -1});
        return x;
    }
};
TORCH_MODULE(ResOutputBlock);

//---------------------------------------
// ModelImpl: full model combining the above blocks.
//---------------------------------------
struct ModelImpl : public torch::nn::Module {
    ResInputBlock res_input{nullptr};
    ResTorsoBlock res_torso{nullptr};
    ResOutputBlock res_output{nullptr};

    ModelImpl(int in_channels, int out_channels) {
        // Initial block: from input channels to 64 feature maps.
        res_input = register_module("res_input", ResInputBlock(in_channels, 128, 3, 1, 1));
        // Torso block operating on 64 channels.
        res_torso = register_module("res_torso", ResTorsoBlock(128, 3, 1, 1));
        // Output block assumes feature map is downsampled to 64*56*56.
        res_output = register_module("res_output", ResOutputBlock(128, out_channels, 1, 1, 0));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = res_input->forward(x);
        x = res_torso->forward(x);
        x = res_output->forward(x);
        return x;
    }
};
TORCH_MODULE(Model);

#endif // MODEL_H
