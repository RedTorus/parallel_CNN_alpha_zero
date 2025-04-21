#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Block dimensions for the torso kernel.
#define BLOCK_DIM_X 128
#define BLOCK_DIM_Y 8

// CNN parameters for torso layer.
#define TORUS_INPUT_CHANNELS    128    // Number of input channels.
#define TORUS_INPUT_HEIGHT       8
#define TORUS_INPUT_WIDTH        8
#define TORUS_FILTER_SIZE        3      // 3x3 filter.
#define TORUS_OUTPUT_HEIGHT      8
#define TORUS_OUTPUT_WIDTH       8

// CUDA kernel for torso convolution.
// Each block computes one output channel.
__global__ void torsoConvKernel(const float* __restrict__ d_input,
                                const float* __restrict__ d_filter,
                                float* __restrict__ d_output,
                                int in_channels, int in_height, int in_width,
                                int filter_size, int out_height, int out_width) {
    // Get the output channel to be computed by this thread block.
    int out_channel_idx = blockIdx.x;

    // Shared memory tile:
    // Load all input channels (each 8x8 = 64 elements) into a tiled array.
    // We group 128 channels in groups of 32 → 4 groups; total rows = 4*64 = 256, columns = 32.
    __shared__ float s_input[256][32];

    int channel_size = in_height * in_width; // 8*8 = 64.
    int total_elements = in_channels * channel_size; // 128*64 = 8192.
    int block_size = blockDim.x;         // TB size
    int tid = threadIdx.x; // thread id in this TB

    // Load input data from global memory into shared memory.
    for (int idx = tid; idx < total_elements; idx += block_size) {
        // Determine the channel and index within that channel.
        int channel = idx / channel_size;        // channel index [0, 127]
        int within_channel = idx % channel_size;   // index within 8x8 channel [0, 63]

        // Compute the shared memory row and column for this element.
        int smem_col = channel % 32;                       // Input channel 0 -> col 0, channel 1 -> col 1, etc.
        int row_offset = (channel / 32) * channel_size;      // Each block of 32 channels maps to 64 rows.
        int smem_row = row_offset + within_channel;          // Place the element in the correct row

        // Copy from global memory to shared memory.
        s_input[smem_row][smem_col] = d_input[idx];
    }
    // Synchronize after data copy is completed
    __syncthreads();

    // Get the input channel to be used by this thread
    // Each thread (threadIdx.x) processes one input channel.
    int in_channel_idx = tid % in_channels;
    // Get the column index of output channel to be computed by this thread
    int out_col = tid / BLOCK_DIM_X;

    // Loop over output rows.
    for (int out_row = 0; out_row < out_height; out_row++) {
        float sum = 0.0f;
        
        // Perform a 3x3 convolution (with padding = 1).
        for (int fr = 0; fr < filter_size; fr++) {
            for (int fc = 0; fc < filter_size; fc++) {
                int in_r = out_row + fr - (filter_size / 2);
                int in_c = out_col + fc - (filter_size / 2);
                // padding part: value 0
                float in_val = 0.0f;
                // not padding part
                if (in_r >= 0 && in_r < in_height && in_c >= 0 && in_c < in_width) {
                    in_val = s_input[(in_channel_idx / 32) * channel_size + in_r * in_width + in_c][in_channel_idx % 32];
                }

                float weight = d_filter[(out_channel_idx * in_channels + in_channel_idx) * (filter_size * filter_size) + fr * filter_size + fc];
                sum += in_val * weight;
            }
        }
        // Atomically add the computed sum into the output pixel.
        atomicAdd(&d_output[out_channel_idx * (out_height * out_width) + out_row * out_width + out_col], sum);
        __syncthreads();
    }

    __syncthreads();
}

// C++ interface: Expects input tensor of shape [1, 128, 8, 8] and filter of shape [128, 128, 3, 3].
// Returns a tensor of shape [128, 8, 8] (batch dimension is dropped).
torch::Tensor torso_conv_forward(torch::Tensor input, torch::Tensor filter) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(filter.is_cuda(), "filter must be a CUDA tensor");

    if (!input.is_contiguous()) {
        input = input.contiguous();
    }
    if (!filter.is_contiguous()) {
        filter = filter.contiguous();
    }
    
    // We assume batch = 1.
    int B = input.size(0);
    int in_channels = input.size(1);   // 128.
    int in_height = input.size(2);       // 8.
    int in_width = input.size(3);        // 8.
    int filter_size = filter.size(2);    // 3.
    int out_height = in_height;
    int out_width = in_width;
    int out_channels = filter.size(0);   // 128.
    
    // Create output tensor of shape [out_channels, out_height, out_width].
    auto output = torch::zeros({B, out_channels, out_height, out_width}, input.options());
    
    // Launch one block per output channel.
    //dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y); // organize 128 x 8 threads in a TB
    //dim3 grid(out_channels); // organize TBs (1 kernel call/SM per TB)
    
    torsoConvKernel<<<out_channels, BLOCK_DIM_X * BLOCK_DIM_Y>>>(
         input.data_ptr<float>(),
         filter.data_ptr<float>(),
         output.data_ptr<float>(),
         in_channels, in_height, in_width,
         filter_size, out_height, out_width
    );
    cudaDeviceSynchronize();
    return output;
}

// Only register the function if BUILD_PY_EXTENSION is defined
#ifdef BUILD_PY_EXTENSION
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torso_conv_forward", &torso_conv_forward, "Torso Convolution Forward (CUDA)");
}
#endif
