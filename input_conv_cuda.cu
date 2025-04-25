#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define LENGTH            320
#define BATCH_SIZE        1
#define INPUT_HEIGHT      8
#define INPUT_WIDTH       8
#define INPUT_CHANNELS    5
#define OUTPUT_HEIGHT     8
#define OUTPUT_WIDTH      8
#define OUTPUT_CHANNELS   128
#define KERNEL_SIZE       3
#define STRIDE           1
#define PADDING          1
 
constexpr int SHARED_COLS = 32;
constexpr int SHARED_ROWS = 10; //(TOTAL_INPUT + SHARED_COLS - 1) / SHARED_COLS;
constexpr int TOTAL_INPUT = INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH; // 5*8*8 = 320
constexpr int WEIGHT_COUNT   = INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE; 

// CUDA kernel for the convolution operation
__global__ void input_conv_kernel(const float* __restrict__ d_input,
    const float* __restrict__ d_weight,
    float* __restrict__ d_output) {

        // For numT = 40
        int numT = blockDim.x;
        int TBid = blockIdx.x;
        int tid = threadIdx.x;

        // Idea: store input and weights that are needed by thread in local memory instead of pulling them from global memory

        // 1) Stage the input volume into shared memory
        //    Flatten (inCh, y, x) -> flatIndex, then flatIndex -> (row, col) in [256][32]
        // (c,y,x) -> N = c*H*W + y*W + x
        // N -> (r,c) = (N/32, N%32)

        __shared__ float sharedInput[SHARED_ROWS][SHARED_COLS];
        for (int flatIndex = tid; flatIndex < TOTAL_INPUT; flatIndex += numT) {
        
            int sharedRow = flatIndex / SHARED_COLS;  // 0..255
            int sharedCol = flatIndex % SHARED_COLS;  // 0..31
            sharedInput[sharedRow][sharedCol] = d_input[flatIndex];
        }

        __syncthreads();

        int outc = TBid;
        int inc = tid / OUTPUT_WIDTH;
        int outRow = tid % OUTPUT_WIDTH;

        // convolution
        for (int outCol = 0; outCol < OUTPUT_HEIGHT; outCol++) {
            float sum = 0.0f;
            for (int krow = 0; krow < KERNEL_SIZE; krow++) {
                for (int kcol = 0; kcol < KERNEL_SIZE; kcol++) {
                    int inRow = outRow + krow - PADDING;
                    int inCol = outCol + kcol - PADDING;
                    float inputVal = 0.0f;
                    if (inRow >= 0 && inRow < INPUT_HEIGHT && inCol >= 0 && inCol < INPUT_WIDTH) {
                        int flatIndex = inc * INPUT_HEIGHT * INPUT_WIDTH + inRow * INPUT_WIDTH + inCol;
                        int srow = flatIndex / SHARED_COLS;
                        int scol = flatIndex % SHARED_COLS;
                        inputVal = sharedInput[srow][scol];
                    }
                    int weightIndex = outc * WEIGHT_COUNT + inc * KERNEL_SIZE * KERNEL_SIZE + krow * KERNEL_SIZE + kcol;
                    float weightVal = d_weight[weightIndex];
                    sum += inputVal * weightVal;
                }
            }
            int outIndex = outc * OUTPUT_HEIGHT * OUTPUT_WIDTH + outRow * OUTPUT_WIDTH + outCol;
            atomicAdd(&d_output[outIndex], sum);
            __syncthreads();
        }
        __syncthreads();
    
    }


__global__ void input_conv_kernelV2(const float* __restrict__ d_input,
        const float* __restrict__ d_weight,
        float* __restrict__ d_output) {
    
            // For numT = 40
            int numT = blockDim.x;
            int TBid = blockIdx.x;
            int tid = threadIdx.x;
    
            // Idea: store input and weights that are needed by thread in local memory instead of pulling them from global memory
    
            // 1) Stage the input volume into shared memory
            //    Flatten (inCh, y, x) -> flatIndex, then flatIndex -> (row, col) in [256][32]
            // (c,y,x) -> N = c*H*W + y*W + x
            // N -> (r,c) = (N/32, N%32)
    
            __shared__ float sharedInput[SHARED_ROWS][SHARED_COLS];
            for (int flatIndex = tid; flatIndex < TOTAL_INPUT; flatIndex += numT) {
            
                int sharedRow = flatIndex / SHARED_COLS;  // 0..255
                int sharedCol = flatIndex % SHARED_COLS;  // 0..31
                sharedInput[sharedRow][sharedCol] = d_input[flatIndex];
            }

            __shared__ float sharedWeight[INPUT_CHANNELS*KERNEL_SIZE*KERNEL_SIZE];
            for (int flatIndex = tid; flatIndex < INPUT_CHANNELS*KERNEL_SIZE*KERNEL_SIZE; flatIndex += numT) {
                sharedWeight[flatIndex] = d_weight[TBid*INPUT_CHANNELS*KERNEL_SIZE*KERNEL_SIZE + flatIndex];
            }
    
            __syncthreads();
    
            int outc = TBid;
            int inc = tid / OUTPUT_WIDTH;
            int outRow = tid % OUTPUT_WIDTH;
    
            // convolution
            for (int outCol = 0; outCol < OUTPUT_HEIGHT; outCol++) {
                float sum = 0.0f;
                for (int krow = 0; krow < KERNEL_SIZE; krow++) {
                    for (int kcol = 0; kcol < KERNEL_SIZE; kcol++) {
                        int inRow = outRow + krow - PADDING;
                        int inCol = outCol + kcol - PADDING;
                        float inputVal = 0.0f;
                        if (inRow >= 0 && inRow < INPUT_HEIGHT && inCol >= 0 && inCol < INPUT_WIDTH) {
                            int flatIndex = inc * INPUT_HEIGHT * INPUT_WIDTH + inRow * INPUT_WIDTH + inCol;
                            int srow = flatIndex / SHARED_COLS;
                            int scol = flatIndex % SHARED_COLS;
                            inputVal = sharedInput[srow][scol];
                        }
                        int weightIndex = inc * KERNEL_SIZE * KERNEL_SIZE + krow * KERNEL_SIZE + kcol;
                        float weightVal = sharedWeight[weightIndex];
                        sum += inputVal * weightVal;
                    }
                }
                int outIndex = outc * OUTPUT_HEIGHT * OUTPUT_WIDTH + outRow * OUTPUT_WIDTH + outCol;
                atomicAdd(&d_output[outIndex], sum);
                __syncthreads();
            }
            __syncthreads();
        
        }

torch::Tensor input_conv_forward(torch::Tensor input, torch::Tensor conv_weights) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(conv_weights.is_cuda(), "conv_weights must be a CUDA tensor");

    if (!input.is_contiguous()) {
        input = input.contiguous();
    }
    if (!conv_weights.is_contiguous()) {
        conv_weights = conv_weights.contiguous();
    }

    //input must be either or shape [1, 320] or [1, 5, 8, 8]
    TORCH_CHECK(input.dim() == 2 || input.dim() == 4, "input must have 2 or 4 dimensions");
    if (input.dim() == 2 && input.size(0) == BATCH_SIZE && input.size(1) == LENGTH) {
        input = input.view({BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH});
    } else if (!(input.dim() == 4 && input.size(0) == BATCH_SIZE && input.size(1) == INPUT_CHANNELS && input.size(2) == INPUT_HEIGHT && input.size(3) == INPUT_WIDTH)) {
        throw std::invalid_argument("input must be of shape [ " + std::to_string(BATCH_SIZE) + ", " + 
        std::to_string(INPUT_CHANNELS) + ", " + std::to_string(INPUT_HEIGHT) + ", " + 
        std::to_string(INPUT_WIDTH) + "] or [ " + std::to_string(BATCH_SIZE) + ", " + 
        std::to_string(LENGTH) + "]");
    }

    // Convolution weights must be of shape [128, 5, 3, 3]
    TORCH_CHECK(conv_weights.dim() == 4, "conv_weights must have 4 dimensions");
    if (!(conv_weights.size(0) == OUTPUT_CHANNELS && conv_weights.size(1) == INPUT_CHANNELS && conv_weights.size(2) == KERNEL_SIZE && conv_weights.size(3) == KERNEL_SIZE)) {
        throw std::invalid_argument("conv_weights must be of shape [" + std::to_string(OUTPUT_CHANNELS) + ", " + 
        std::to_string(INPUT_CHANNELS) + ", " + std::to_string(KERNEL_SIZE) + ", " + 
        std::to_string(KERNEL_SIZE) + "]");
    }

    float *input_ptr = input.data_ptr<float>();
    float *conv_weights_ptr = conv_weights.data_ptr<float>();

    const int threads = 40;
    const int blocks  = OUTPUT_CHANNELS;
    auto output = torch::zeros({1, OUTPUT_CHANNELS, OUTPUT_HEIGHT, OUTPUT_WIDTH}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

    input_conv_kernelV2<<<blocks, threads>>>(
        input_ptr,
        conv_weights_ptr,
        output.data_ptr<float>()
    );

    

    cudaDeviceSynchronize();
    return output;

}