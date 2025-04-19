#ifndef TEST_BLOCKS_H
#define TEST_BLOCKS_H

#include <torch/torch.h>

// Declare the function
bool outputs_identical(const torch::Tensor& out1, const torch::Tensor& out2, double tol = 1e-5);

torch::Tensor computeAbsoluteDifferenceSum(const torch::Tensor& tensor1, const torch::Tensor& tensor2);

torch::Tensor computeAverageAbsoluteDifference(const torch::Tensor& tensor1, const torch::Tensor& tensor2);

torch::Tensor testConv2dBlock3(const torch::Tensor& input, const torch::Tensor& conv_weights);

torch::Tensor testConv2dBlock4(const torch::Tensor& input, const torch::Tensor& conv_weights);

#endif // TEST_BLOCKS_H