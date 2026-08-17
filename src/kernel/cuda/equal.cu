#include "src/kernel/equal.h"

#include <memory>
#include <vector>

#include "src/kernel/cuda/kernel_io.cuh"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_equal_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Equal", []() {
        return std::make_unique<EqualKernel<DeviceType::CUDA, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Equal", []() {
        return std::make_unique<EqualKernel<DeviceType::CUDA, DataType::FP16>>();
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
__global__ void EqualKernelCuda(const T* lhs, const T* rhs, uint8_t* output, int64_t output_numel,
                                cuda_detail::CudaShape output_shape, cuda_detail::CudaShape lhs_shape,
                                cuda_detail::CudaShape rhs_shape) {
    const int64_t output_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output_index >= output_numel) {
        return;
    }
    const int64_t lhs_offset = BroadcastOffsetForOutputIndex(output_index, output_shape, lhs_shape);
    const int64_t rhs_offset = BroadcastOffsetForOutputIndex(output_index, output_shape, rhs_shape);
    output[output_index] = cuda_detail::ReadDevice(lhs, lhs_offset) == cuda_detail::ReadDevice(rhs, rhs_offset) ? 1 : 0;
}

template <DataType dtype>
int32_t RunEqual(feather::operators::EqualParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->lhs == nullptr || param->rhs == nullptr || param->out == nullptr ||
        param->lhs->data_type() != dtype || param->rhs->data_type() != dtype ||
        param->out->data_type() != DataType::BOOL) {
        return -1;
    }
    const auto& lhs_dims = param->lhs->dims().data();
    const auto& rhs_dims = param->rhs->dims().data();
    const auto& output_dims = param->out->dims().data();
    std::vector<int64_t> expected_dims;
    if (!cuda_detail::InferBroadcastShape(lhs_dims, rhs_dims, &expected_dims) || expected_dims != output_dims) {
        return -1;
    }
    cuda_detail::CudaShape lhs_shape;
    cuda_detail::CudaShape rhs_shape;
    cuda_detail::CudaShape output_shape;
    if (!cuda_detail::MakeCudaShape(lhs_dims, &lhs_shape) || !cuda_detail::MakeCudaShape(rhs_dims, &rhs_shape) ||
        !cuda_detail::MakeCudaShape(output_dims, &output_shape)) {
        return -1;
    }
    cuda_detail::DeviceBuffer<T> lhs;
    cuda_detail::DeviceBuffer<T> rhs;
    cuda_detail::DeviceBuffer<uint8_t> output;
    if (cuda_detail::CopyTensorToDevice(param->lhs.get(), &lhs) != 0 ||
        cuda_detail::CopyTensorToDevice(param->rhs.get(), &rhs) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &output) != 0) {
        return -1;
    }
    const int64_t output_numel = param->out->numel();
    EqualKernelCuda<T>
        <<<static_cast<int>(cuda_detail::DivUp(output_numel, cuda_detail::kCudaThreads)), cuda_detail::kCudaThreads, 0,
           cuda_detail::InferenceStream()>>>(lhs.get(), rhs.get(), output.get(), output_numel, output_shape, lhs_shape,
                                              rhs_shape);
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0 ||
        cuda_detail::CopyDeviceToTensor(&output, param->out.get()) != 0) {
        return -1;
    }
    param->out->set_data_type(DataType::BOOL);
    return 0;
}

}  // namespace

template <>
int32_t EqualKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return RunEqual<DataType::FP32>(static_cast<feather::operators::EqualParam*>(param_), "CUDA::Equal::FP32");
}

template <>
int32_t EqualKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return RunEqual<DataType::FP16>(static_cast<feather::operators::EqualParam*>(param_), "CUDA::Equal::FP16");
}

void EnsureCudaEqualKernelsRegistered() { (void)g_cuda_equal_kernels_registered; }

}  // namespace kernel
}  // namespace feather
