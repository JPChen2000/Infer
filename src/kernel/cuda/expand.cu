#include "src/kernel/expand.h"

#include <memory>
#include <vector>

#include "src/kernel/cuda/kernel_io.cuh"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_expand_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Expand", []() {
        return std::make_unique<ExpandKernel<DeviceType::CUDA, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Expand", []() {
        return std::make_unique<ExpandKernel<DeviceType::CUDA, DataType::FP16>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::BF16, "Expand", []() {
        return std::make_unique<ExpandKernel<DeviceType::CUDA, DataType::BF16>>();
    });
    return true;
}();

__device__ inline int64_t BroadcastOffsetForOutputIndex(int64_t output_index, cuda_detail::CudaShape output_shape,
                                                         cuda_detail::CudaShape input_shape) {
    int64_t remaining = output_index;
    int64_t input_offset = 0;
    const int rank_gap = output_shape.rank - input_shape.rank;
    for (int axis = 0; axis < output_shape.rank; ++axis) {
        const int64_t coordinate = remaining / output_shape.strides[axis];
        remaining %= output_shape.strides[axis];
        const int input_axis = axis - rank_gap;
        if (input_axis >= 0 && input_shape.dims[input_axis] != 1) {
            input_offset += coordinate * input_shape.strides[input_axis];
        }
    }
    return input_offset;
}

template <typename T>
__global__ void ExpandKernelCuda(const T* input, T* output, int64_t output_numel, cuda_detail::CudaShape output_shape,
                                 cuda_detail::CudaShape input_shape) {
    const int64_t output_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output_index >= output_numel) {
        return;
    }
    const int64_t input_offset = BroadcastOffsetForOutputIndex(output_index, output_shape, input_shape);
    cuda_detail::WriteDevice(output, output_index, cuda_detail::ReadDevice(input, input_offset));
}

template <DataType dtype>
int32_t RunExpand(feather::operators::ExpandParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->input == nullptr || param->shape == nullptr || param->out == nullptr ||
        param->input->data_type() != dtype ||
        (param->shape->data_type() != DataType::INT32 && param->shape->data_type() != DataType::INT64)) {
        return -1;
    }
    const auto& input_dims = param->input->dims().data();
    const auto& output_dims = param->out->dims().data();
    std::vector<int64_t> expected_dims;
    if (!cuda_detail::InferBroadcastShape(input_dims, output_dims, &expected_dims) || expected_dims != output_dims) {
        return -1;
    }
    cuda_detail::CudaShape input_shape;
    cuda_detail::CudaShape output_shape;
    if (!cuda_detail::MakeCudaShape(input_dims, &input_shape) ||
        !cuda_detail::MakeCudaShape(output_dims, &output_shape)) {
        return -1;
    }
    cuda_detail::DeviceBuffer<T> input;
    cuda_detail::DeviceBuffer<T> output;
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &output) != 0) {
        return -1;
    }
    const int64_t output_numel = param->out->numel();
    ExpandKernelCuda<T>
        <<<static_cast<int>(cuda_detail::DivUp(output_numel, cuda_detail::kCudaThreads)), cuda_detail::kCudaThreads, 0,
           cuda_detail::InferenceStream()>>>(input.get(), output.get(), output_numel, output_shape, input_shape);
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) {
        return -1;
    }
    return cuda_detail::CopyDeviceToTensor(&output, param->out.get());
}

}  // namespace

template <>
int32_t ExpandKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return RunExpand<DataType::FP32>(static_cast<feather::operators::ExpandParam*>(param_), "CUDA::Expand::FP32");
}

template <>
int32_t ExpandKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return RunExpand<DataType::FP16>(static_cast<feather::operators::ExpandParam*>(param_), "CUDA::Expand::FP16");
}

template <>
int32_t ExpandKernel<DeviceType::CUDA, DataType::BF16>::compute() {
    return RunExpand<DataType::BF16>(static_cast<feather::operators::ExpandParam*>(param_), "CUDA::Expand::BF16");
}

void EnsureCudaExpandKernelsRegistered() { (void)g_cuda_expand_kernels_registered; }

}  // namespace kernel
}  // namespace feather
