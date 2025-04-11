#ifndef CONV_FUNCTIONS_H
#define CONV_FUNCTIONS_H

#include <torch/torch.h>

// Declaration of the forward functions
at::Tensor torso_conv_forward(const at::Tensor& input, const at::Tensor& weight);
at::Tensor output_conv_forward(const at::Tensor& input, const at::Tensor& weight, int some_param);

#endif // CONV_FUNCTIONS_H