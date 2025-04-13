# parallel_CNN_alpha_zero

## File Structure

- semi_compare.cc: Source code of the semi program to compare outputs of pure CNNs (Libtorch built-in and parallelized version).
- semi_compare_model.cc: Source code of the semi program to compare outputs of complete models simulating the real one used in Alpha Zero (using Libtorch built-in CNNs and parallelized version CNNs).
- model.h: Model simulating the real one used in Alpha Zero using Libtorch built-in CNNs. Some transformations might be left out, such as torch::tanh() on the final output in ResOutputBlock.
- model_par.h: Model simulating the real one used in Alpha Zero using parallelized version CNNs. Some transformations might be left out, such as torch::tanh() on the final output in ResOutputBlock.
- torso_conv_cuda.cu: Parallelized version CNNs used in ResTorsoBlock in Alpha Zero with customized kernel call.
- output_conv_cuda.cu: Parallelized version CNNs used in ResOutputBlock in Alpha Zero with customized kernel call.

## How to Run

- Compare outputs of pure CNNs:
  ```
  make compare_cnn
  ```
- Compare outputs of complete models:
  ```
  make compare_model
  ```
- Clean up:
  ```
  make clean
  ```
