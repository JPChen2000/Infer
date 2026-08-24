#include "src/kernel/qwen_depthwise_conv.h"

#include <memory>
#include <vector>

#include "src/kernel/cuda/kernel_io.cuh"
#include "util/timer.h"

namespace feather {
namespace kernel {
namespace {

bool HasShape(const Tensor* tensor, const std::vector<int64_t>& shape) {
    return tensor != nullptr && tensor->dims().data() == shape;
}

bool Validate(const operators::QwenDepthwiseConvStateParam* param, int64_t* channels) {
    if (param == nullptr || channels == nullptr || param->state == nullptr || param->mixed == nullptr ||
        param->weight == nullptr || param->conv_out == nullptr || param->discarded_prefix == nullptr ||
        param->next_state == nullptr || !param->state->IsInitialized() || !param->mixed->IsInitialized() ||
        !param->weight->IsInitialized() || !param->conv_out->IsInitialized() ||
        !param->discarded_prefix->IsInitialized() || !param->next_state->IsInitialized()) {
        return false;
    }
    if (param->state->data_type() != DataType::BF16 || param->mixed->data_type() != DataType::BF16 ||
        param->weight->data_type() != DataType::BF16 || param->conv_out->data_type() != DataType::BF16 ||
        param->discarded_prefix->data_type() != DataType::BF16 || param->next_state->data_type() != DataType::BF16 ||
        param->state->dims().size() != 3) {
        return false;
    }
    const int64_t c = param->state->dims()[1];
    if (c <= 0 || !HasShape(param->state.get(), {1, c, 3}) || !HasShape(param->mixed.get(), {1, c, 1}) ||
        !HasShape(param->weight.get(), {c, 1, 1, 4}) || !HasShape(param->conv_out.get(), {1, c, 1, 1}) ||
        !HasShape(param->discarded_prefix.get(), {1, c, 1}) || !HasShape(param->next_state.get(), {1, c, 3})) {
        return false;
    }
    *channels = c;
    return true;
}

__global__ void QwenDepthwiseConvStateKernelCuda(const BFloat16* state, const BFloat16* mixed,
                                                 const BFloat16* weight, BFloat16* conv_out,
                                                 BFloat16* discarded_prefix, BFloat16* next_state,
                                                 int64_t channels) {
    const int64_t channel = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channel >= channels) {
        return;
    }
    const int64_t state_offset = channel * 3;
    const int64_t weight_offset = channel * 4;
    const float state0 = cuda_detail::ReadDevice(state, state_offset);
    const float state1 = cuda_detail::ReadDevice(state, state_offset + 1);
    const float state2 = cuda_detail::ReadDevice(state, state_offset + 2);
    const float mixed_value = cuda_detail::ReadDevice(mixed, channel);
    float value = state0 * cuda_detail::ReadDevice(weight, weight_offset);
    value += state1 * cuda_detail::ReadDevice(weight, weight_offset + 1);
    value += state2 * cuda_detail::ReadDevice(weight, weight_offset + 2);
    value += mixed_value * cuda_detail::ReadDevice(weight, weight_offset + 3);
    cuda_detail::WriteDevice(conv_out, channel, value);
    cuda_detail::WriteDevice(discarded_prefix, channel, state0);
    cuda_detail::WriteDevice(next_state, state_offset, state1);
    cuda_detail::WriteDevice(next_state, state_offset + 1, state2);
    cuda_detail::WriteDevice(next_state, state_offset + 2, mixed_value);
}

}  // namespace

template <>
int32_t QwenDepthwiseConvStateKernel<DeviceType::CUDA, DataType::BF16>::compute() {
    AutoTimer timer("CUDA::QwenDepthwiseConvState::BF16");
    auto* param = static_cast<operators::QwenDepthwiseConvStateParam*>(param_);
    int64_t channels = 0;
    if (!Validate(param, &channels)) {
        return -1;
    }
    cuda_detail::DeviceBuffer<BFloat16> state;
    cuda_detail::DeviceBuffer<BFloat16> mixed;
    cuda_detail::DeviceBuffer<BFloat16> weight;
    cuda_detail::DeviceBuffer<BFloat16> conv_out;
    cuda_detail::DeviceBuffer<BFloat16> discarded_prefix;
    cuda_detail::DeviceBuffer<BFloat16> next_state;
    if (cuda_detail::CopyTensorToDevice(param->state.get(), &state) != 0 ||
        cuda_detail::CopyTensorToDevice(param->mixed.get(), &mixed) != 0 ||
        cuda_detail::CopyTensorToDevice(param->weight.get(), &weight) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->conv_out.get(), &conv_out) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->discarded_prefix.get(), &discarded_prefix) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->next_state.get(), &next_state) != 0) {
        return -1;
    }
    QwenDepthwiseConvStateKernelCuda
        <<<static_cast<int>(cuda_detail::DivUp(channels, cuda_detail::kCudaThreads)), cuda_detail::kCudaThreads, 0,
           cuda_detail::InferenceStream()>>>(state.get(), mixed.get(), weight.get(), conv_out.get(),
                                               discarded_prefix.get(), next_state.get(), channels);
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) {
        return -1;
    }
    if (cuda_detail::CopyDeviceToTensor(&conv_out, param->conv_out.get()) != 0 ||
        cuda_detail::CopyDeviceToTensor(&discarded_prefix, param->discarded_prefix.get()) != 0 ||
        cuda_detail::CopyDeviceToTensor(&next_state, param->next_state.get()) != 0) {
        return -1;
    }
    return 0;
}

bool g_cuda_qwen_depthwise_conv_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::CUDA, DataType::BF16, "QwenDepthwiseConvState", []() {
        return std::make_unique<QwenDepthwiseConvStateKernel<DeviceType::CUDA, DataType::BF16>>();
    });
    return true;
}();

void EnsureCudaQwenDepthwiseConvStateKernelsRegistered() { (void)g_cuda_qwen_depthwise_conv_kernels_registered; }

}  // namespace kernel
}  // namespace feather
