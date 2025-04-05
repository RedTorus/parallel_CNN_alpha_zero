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

