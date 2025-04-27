# parallel_CNN_alpha_zero

## File Structure

- semi_compare.cc: Source code of the semi program to compare outputs of pure CNNs (Libtorch built-in and parallelized version).
- semi_compare_bn_relu.cc: Source code of the semi program to compare outputs of layers wrapped up with CNNs (Libtorch built-in and parallelized version), batch normalization and ReLU.
- semi_compare_model.cc: Source code of the semi program to compare outputs of complete models simulating the real one used in Alpha Zero (using Libtorch built-in CNNs and parallelized version CNNs).
- model.h: Model simulating the real one used in Alpha Zero using Libtorch built-in CNNs. Some transformations might be left out, such as torch::tanh() on the final output in ResOutputBlock.
- model_par.h: Model simulating the real one used in Alpha Zero using parallelized version CNNs. Some transformations might be left out, such as torch::tanh() on the final output in ResOutputBlock.
- input_conv_cuda.cu: Parallelized version CNNs used in ResInputBlock in Alpha Zero with customized kernel call.
- torso_conv_cuda.cu: Parallelized version CNNs used in ResTorsoBlock in Alpha Zero with customized kernel call.
- output_conv_cuda.cu: Parallelized version CNNs used in ResOutputBlock in Alpha Zero with customized kernel call.
- plot.py: Plot performance results.
- output: Folder storing correctness check and performance results.
- plot: Folder storing all plots.

## How to Run

- Compare outputs of pure CNNs:
  ```
  make compare_cnn
  ```
  You can modify the int variable `mode` in line 62 in `semi_compare.cc` to change the type of custom kernel used. Specifically, 0 is for input kernel, 1 is for torso kernel and 2 is for output kernel.
- Compare outputs of layers:
  ```
  make compare_bn_relu
  ```
- Compare outputs of complete models:
  ```
  make compare_model
  ```
- Clean up:
  ```
  make clean
  ```
- Plot performance results:
  ```
  python plot.py
  ```

## Machine used
ece022.ece.local.cmu.edu
