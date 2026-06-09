#include "src/kernel/concat.h"

#include <memory>
#include <vector>

#include "core/kernel.h"
#include "src/kernel/cuda/kernel_io.cuh"
#include "src/operator/params.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_concat_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Concat",
                              []() { return std::make_unique<ConcatKernel<DeviceType::CUDA, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Concat",
                              []() { return std::make_unique<ConcatKernel<DeviceType::CUDA, DataType::FP16>>(); });
    return true;
}();

template <typename T>
__global__ void ConcatCopyKernelCuda(const T* input, T* out, int64_t total, int64_t input_axis, int64_t inner,
                                     int64_t out_axis, int64_t axis_offset) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) {
        return;
    }
    const int64_t inner_idx = linear % inner;
    const int64_t axis_idx = (linear / inner) % input_axis;
    const int64_t outer_idx = linear / (input_axis * inner);
    const int64_t output_idx = (outer_idx * out_axis + axis_offset + axis_idx) * inner + inner_idx;
    out[output_idx] = input[linear];
}

template <DataType dtype>
int RunConcat(feather::operators::ConcatParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->out == nullptr || param->inputs.size() < 2) {
        return -1;
    }
    const auto& out_dims = param->out->dims().data();
    int axis = param->axis < 0 ? param->axis + static_cast<int>(out_dims.size()) : param->axis;
    if (axis < 0 || axis >= static_cast<int>(out_dims.size())) {
        return -1;
    }
    std::vector<cuda_detail::DeviceBuffer<T>> inputs(param->inputs.size());
    for (size_t i = 0; i < param->inputs.size(); ++i) {
        if (param->inputs[i] == nullptr || cuda_detail::CopyTensorToDevice(param->inputs[i].get(), &inputs[i]) != 0) {
            return -1;
        }
    }
    const int64_t outer = cuda_detail::ComputeProduct(out_dims, 0, static_cast<size_t>(axis));
    const int64_t inner = cuda_detail::ComputeProduct(out_dims, static_cast<size_t>(axis) + 1, out_dims.size());
    const int64_t out_axis = out_dims[static_cast<size_t>(axis)];
    cuda_detail::DeviceBuffer<T> out;
    if (cuda_detail::AllocateTensorOnDevice(param->out.get(), &out) != 0) {
        return -1;
    }
    param->out->set_data_type(dtype);
    const size_t element_bytes = sizeof(T);
    const size_t dst_pitch_bytes = static_cast<size_t>(out_axis * inner) * element_bytes;
    int64_t axis_offset = 0;
    for (size_t i = 0; i < param->inputs.size(); ++i) {
        const auto& input_tensor = param->inputs[i];
        const int64_t input_axis = input_tensor->dims()[axis];
        const size_t width_bytes = static_cast<size_t>(input_axis * inner) * element_bytes;
        if (outer > 0 && width_bytes > 0) {
            char* dst_ptr = reinterpret_cast<char*>(out.get()) + static_cast<size_t>(axis_offset * inner) * element_bytes;
            if (cuda_detail::CudaCheck(cudaMemcpy2DAsync(dst_ptr, dst_pitch_bytes, inputs[i].get(), width_bytes,
                                                         width_bytes, static_cast<size_t>(outer),
                                                         cudaMemcpyDeviceToDevice, cuda_detail::InferenceStream())) != 0) {
                return -1;
            }
        }
        axis_offset += input_axis;
    }
    SetLastCudaConcatBackend(CudaConcatBackend::kMemcpy2D);
    return cuda_detail::CopyDeviceToTensor(&out, param->out.get());
}

}  // namespace

template <>
int32_t ConcatKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return RunConcat<DataType::FP32>(static_cast<feather::operators::ConcatParam*>(param_), "CUDA::Concat::FP32");
}

template <>
int32_t ConcatKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return RunConcat<DataType::FP16>(static_cast<feather::operators::ConcatParam*>(param_), "CUDA::Concat::FP16");
}

void EnsureCudaConcatKernelsRegistered() { (void)g_cuda_concat_kernels_registered; }

}  // namespace kernel
}  // namespace feather
