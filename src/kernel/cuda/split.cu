#include "src/kernel/split.h"

#include <memory>
#include <vector>

#include "core/kernel.h"
#include "src/kernel/cuda/kernel_io.cuh"
#include "src/operator/params.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_split_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Split",
                              []() { return std::make_unique<SplitKernel<DeviceType::CUDA, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Split",
                              []() { return std::make_unique<SplitKernel<DeviceType::CUDA, DataType::FP16>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::BF16, "Split",
                              []() { return std::make_unique<SplitKernel<DeviceType::CUDA, DataType::BF16>>(); });
    return true;
}();

template <typename T>
__global__ void SplitCopyKernelCuda(const T* input, T* out, int64_t total, int64_t output_axis, int64_t inner,
                                    int64_t input_axis, int64_t axis_offset) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) {
        return;
    }
    const int64_t inner_idx = linear % inner;
    const int64_t axis_idx = (linear / inner) % output_axis;
    const int64_t outer_idx = linear / (output_axis * inner);
    const int64_t input_idx = (outer_idx * input_axis + axis_offset + axis_idx) * inner + inner_idx;
    out[linear] = input[input_idx];
}

template <DataType dtype>
int RunSplit(feather::operators::SplitParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->input == nullptr || param->outputs.empty()) {
        return -1;
    }
    const auto& in_dims = param->input->dims().data();
    int axis = param->axis < 0 ? param->axis + static_cast<int>(in_dims.size()) : param->axis;
    if (axis < 0 || axis >= static_cast<int>(in_dims.size())) {
        return -1;
    }
    const int64_t outer = cuda_detail::ComputeProduct(in_dims, 0, static_cast<size_t>(axis));
    const int64_t inner = cuda_detail::ComputeProduct(in_dims, static_cast<size_t>(axis) + 1, in_dims.size());
    const int64_t input_axis = in_dims[static_cast<size_t>(axis)];
    cuda_detail::DeviceBuffer<T> input;
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0) {
        return -1;
    }
    std::vector<cuda_detail::DeviceBuffer<T>> outputs(param->outputs.size());
    for (size_t i = 0; i < param->outputs.size(); ++i) {
        if (param->outputs[i] == nullptr ||
            cuda_detail::AllocateTensorOnDevice(param->outputs[i].get(), &outputs[i]) != 0) {
            return -1;
        }
        param->outputs[i]->set_data_type(dtype);
    }
    int64_t axis_offset = 0;
    for (size_t i = 0; i < param->outputs.size(); ++i) {
        const int64_t output_axis = param->outputs[i]->dims()[axis];
        const int64_t copy_count = output_axis * inner;
        const int64_t total = outer * copy_count;
        if (total > 0) {
            SplitCopyKernelCuda<T><<<static_cast<int>(cuda_detail::DivUp(total, cuda_detail::kCudaThreads)),
                                     cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
                input.get(), outputs[i].get(), total, output_axis, inner, input_axis, axis_offset);
            if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) {
                return -1;
            }
        }
        axis_offset += output_axis;
    }
    for (size_t i = 0; i < param->outputs.size(); ++i) {
        if (cuda_detail::CopyDeviceToTensor(&outputs[i], param->outputs[i].get()) != 0) {
            return -1;
        }
    }
    return 0;
}

}  // namespace

template <>
int32_t SplitKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return RunSplit<DataType::FP32>(static_cast<feather::operators::SplitParam*>(param_), "CUDA::Split::FP32");
}

template <>
int32_t SplitKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return RunSplit<DataType::FP16>(static_cast<feather::operators::SplitParam*>(param_), "CUDA::Split::FP16");
}

template <>
int32_t SplitKernel<DeviceType::CUDA, DataType::BF16>::compute() {
    return RunSplit<DataType::BF16>(static_cast<feather::operators::SplitParam*>(param_), "CUDA::Split::BF16");
}

void EnsureCudaSplitKernelsRegistered() { (void)g_cuda_split_kernels_registered; }

}  // namespace kernel
}  // namespace feather
