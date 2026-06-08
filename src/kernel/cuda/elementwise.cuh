#ifndef FEATHER_KERNEL_CUDA_ELEMENTWISE_CUH
#define FEATHER_KERNEL_CUDA_ELEMENTWISE_CUH

#include <cuda_runtime.h>

#include <vector>

#include "src/kernel/cuda/kernel_io.cuh"
#include "src/operator/params.h"
#include "util/timer.h"

namespace feather {
namespace kernel {
namespace cuda_detail {

template <typename T, int Op>
__global__ void UnaryKernelCuda(const T* input, T* output, int64_t numel, float exponent) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    const float x = ReadDevice(input, idx);
    float y = x;
    if constexpr (Op == 0) {
        y = x > 0.0f ? x : 0.0f;
    } else if constexpr (Op == 1) {
        y = 1.0f / (1.0f + expf(-x));
    } else if constexpr (Op == 2) {
        y = powf(x, exponent);
    } else if constexpr (Op == 3) {
        y = x / (1.0f + expf(-x));
    }
    WriteDevice(output, idx, y);
}

template <typename T, int Op>
__global__ void BinaryBroadcastKernelCuda(const T* lhs, const T* rhs, T* out, int64_t numel, CudaShape out_shape,
                                          CudaShape lhs_shape, CudaShape rhs_shape) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= numel) {
        return;
    }
    int64_t remaining = linear;
    int64_t lhs_offset = 0;
    int64_t rhs_offset = 0;
    const int lhs_gap = out_shape.rank - lhs_shape.rank;
    const int rhs_gap = out_shape.rank - rhs_shape.rank;
    for (int axis = 0; axis < out_shape.rank; ++axis) {
        const int64_t coord = remaining / out_shape.strides[axis];
        remaining %= out_shape.strides[axis];
        const int lhs_axis = axis - lhs_gap;
        if (lhs_axis >= 0) {
            lhs_offset += (lhs_shape.dims[lhs_axis] == 1 ? 0 : coord) * lhs_shape.strides[lhs_axis];
        }
        const int rhs_axis = axis - rhs_gap;
        if (rhs_axis >= 0) {
            rhs_offset += (rhs_shape.dims[rhs_axis] == 1 ? 0 : coord) * rhs_shape.strides[rhs_axis];
        }
    }
    const float a = ReadDevice(lhs, lhs_offset);
    const float b = ReadDevice(rhs, rhs_offset);
    WriteDevice(out, linear, Op == 0 ? a + b : a * b);
}

template <DataType dtype, int Op>
int RunUnary(feather::operators::UnaryParam* param, const char* timer_name, float exponent = 1.0f) {
    AutoTimer timer(timer_name);
    using T = StorageT<dtype>;
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }
    DeviceBuffer<T> input;
    DeviceBuffer<T> output;
    const int64_t numel = param->input->numel();
    if (CopyTensorToDevice(param->input.get(), &input) != 0 || AllocateTensorOnDevice(param->out.get(), &output) != 0) {
        return -1;
    }
    UnaryKernelCuda<T, Op><<<static_cast<int>(DivUp(numel, kCudaThreads)), kCudaThreads, 0, InferenceStream()>>>(
        input.get(), output.get(), numel, exponent);
    if (CudaCheck(cudaGetLastError()) != 0) {
        return -1;
    }
    return CopyDeviceToTensor(&output, param->out.get());
}

template <DataType dtype>
int RunPow(feather::operators::PowParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = StorageT<dtype>;
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }
    DeviceBuffer<T> input;
    DeviceBuffer<T> output;
    const int64_t numel = param->input->numel();
    if (CopyTensorToDevice(param->input.get(), &input) != 0 || AllocateTensorOnDevice(param->out.get(), &output) != 0) {
        return -1;
    }
    UnaryKernelCuda<T, 2><<<static_cast<int>(DivUp(numel, kCudaThreads)), kCudaThreads, 0, InferenceStream()>>>(
        input.get(), output.get(), numel, param->exponent);
    if (CudaCheck(cudaGetLastError()) != 0) {
        return -1;
    }
    return CopyDeviceToTensor(&output, param->out.get());
}

template <DataType dtype, int Op>
int RunBinary(feather::operators::BinaryParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = StorageT<dtype>;
    if (param == nullptr || param->lhs == nullptr || param->rhs == nullptr || param->out == nullptr) {
        return -1;
    }
    const auto& lhs_dims = param->lhs->dims().data();
    const auto& rhs_dims = param->rhs->dims().data();
    std::vector<int64_t> out_dims;
    if (!InferBroadcastShape(lhs_dims, rhs_dims, &out_dims) || out_dims.size() > kMaxCudaRank) {
        return -1;
    }
    CudaShape out_shape;
    CudaShape lhs_shape;
    CudaShape rhs_shape;
    if (!MakeCudaShape(out_dims, &out_shape) || !MakeCudaShape(lhs_dims, &lhs_shape) ||
        !MakeCudaShape(rhs_dims, &rhs_shape)) {
        return -1;
    }
    DeviceBuffer<T> lhs;
    DeviceBuffer<T> rhs;
    DeviceBuffer<T> out;
    const int64_t numel = param->out->numel();
    if (CopyTensorToDevice(param->lhs.get(), &lhs) != 0 || CopyTensorToDevice(param->rhs.get(), &rhs) != 0 ||
        AllocateTensorOnDevice(param->out.get(), &out) != 0) {
        return -1;
    }
    BinaryBroadcastKernelCuda<T, Op><<<static_cast<int>(DivUp(numel, kCudaThreads)), kCudaThreads, 0, InferenceStream()>>>(
        lhs.get(), rhs.get(), out.get(), numel, out_shape, lhs_shape, rhs_shape);
    if (CudaCheck(cudaGetLastError()) != 0) {
        return -1;
    }
    return CopyDeviceToTensor(&out, param->out.get());
}

}  // namespace cuda_detail
}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_CUDA_ELEMENTWISE_CUH
