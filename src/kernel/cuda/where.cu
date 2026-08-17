#include "src/kernel/where.h"

#include <memory>
#include <vector>

#include "src/kernel/cuda/kernel_io.cuh"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_where_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Where", []() {
        return std::make_unique<WhereKernel<DeviceType::CUDA, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Where", []() {
        return std::make_unique<WhereKernel<DeviceType::CUDA, DataType::FP16>>();
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
__global__ void WhereKernelCuda(const uint8_t* condition, const T* x, const T* y, T* output, int64_t output_numel,
                                cuda_detail::CudaShape output_shape, cuda_detail::CudaShape condition_shape,
                                cuda_detail::CudaShape x_shape, cuda_detail::CudaShape y_shape) {
    const int64_t output_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output_index >= output_numel) {
        return;
    }
    const int64_t condition_offset = BroadcastOffsetForOutputIndex(output_index, output_shape, condition_shape);
    const int64_t x_offset = BroadcastOffsetForOutputIndex(output_index, output_shape, x_shape);
    const int64_t y_offset = BroadcastOffsetForOutputIndex(output_index, output_shape, y_shape);
    const float value = condition[condition_offset] != 0 ? cuda_detail::ReadDevice(x, x_offset)
                                                           : cuda_detail::ReadDevice(y, y_offset);
    cuda_detail::WriteDevice(output, output_index, value);
}

template <DataType dtype>
int32_t RunWhere(feather::operators::WhereParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->condition == nullptr || param->x == nullptr || param->y == nullptr ||
        param->out == nullptr || param->condition->data_type() != DataType::BOOL || param->x->data_type() != dtype ||
        param->y->data_type() != dtype) {
        return -1;
    }
    const auto& condition_dims = param->condition->dims().data();
    const auto& x_dims = param->x->dims().data();
    const auto& y_dims = param->y->dims().data();
    const auto& output_dims = param->out->dims().data();
    std::vector<int64_t> condition_x_dims;
    std::vector<int64_t> expected_dims;
    if (!cuda_detail::InferBroadcastShape(condition_dims, x_dims, &condition_x_dims) ||
        !cuda_detail::InferBroadcastShape(condition_x_dims, y_dims, &expected_dims) || expected_dims != output_dims) {
        return -1;
    }
    cuda_detail::CudaShape condition_shape;
    cuda_detail::CudaShape x_shape;
    cuda_detail::CudaShape y_shape;
    cuda_detail::CudaShape output_shape;
    if (!cuda_detail::MakeCudaShape(condition_dims, &condition_shape) ||
        !cuda_detail::MakeCudaShape(x_dims, &x_shape) || !cuda_detail::MakeCudaShape(y_dims, &y_shape) ||
        !cuda_detail::MakeCudaShape(output_dims, &output_shape)) {
        return -1;
    }
    cuda_detail::DeviceBuffer<uint8_t> condition;
    cuda_detail::DeviceBuffer<T> x;
    cuda_detail::DeviceBuffer<T> y;
    cuda_detail::DeviceBuffer<T> output;
    if (cuda_detail::CopyTensorToDevice(param->condition.get(), &condition) != 0 ||
        cuda_detail::CopyTensorToDevice(param->x.get(), &x) != 0 ||
        cuda_detail::CopyTensorToDevice(param->y.get(), &y) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &output) != 0) {
        return -1;
    }
    const int64_t output_numel = param->out->numel();
    WhereKernelCuda<T>
        <<<static_cast<int>(cuda_detail::DivUp(output_numel, cuda_detail::kCudaThreads)), cuda_detail::kCudaThreads, 0,
           cuda_detail::InferenceStream()>>>(condition.get(), x.get(), y.get(), output.get(), output_numel,
                                              output_shape, condition_shape, x_shape, y_shape);
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) {
        return -1;
    }
    return cuda_detail::CopyDeviceToTensor(&output, param->out.get());
}

}  // namespace

template <>
int32_t WhereKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return RunWhere<DataType::FP32>(static_cast<feather::operators::WhereParam*>(param_), "CUDA::Where::FP32");
}

template <>
int32_t WhereKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return RunWhere<DataType::FP16>(static_cast<feather::operators::WhereParam*>(param_), "CUDA::Where::FP16");
}

void EnsureCudaWhereKernelsRegistered() { (void)g_cuda_where_kernels_registered; }

}  // namespace kernel
}  // namespace feather
