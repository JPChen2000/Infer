#include "fc_compute.h"

namespace feather {
namespace kernel {

__global__ void linear_cuda_kernel(const float* input, const float* weights, const float* bias, float* output, int input_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        float result = bias[idx];
        for (int j = 0; j < input_size; ++j) {
            result += input[j] * weights[idx * input_size + j];
        }
        output[idx] = result;
    }
}

} // namespace kernel
} // namespace feather
