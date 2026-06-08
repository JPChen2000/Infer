#include "src/kernel/slice.h"

#include <algorithm>
#include <memory>

#include "core/kernel.h"
#include "src/kernel/cuda/kernel_io.cuh"
#include "src/operator/params.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_slice_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Slice",
                              []() { return std::make_unique<SliceKernel<DeviceType::CUDA, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Slice",
                              []() { return std::make_unique<SliceKernel<DeviceType::CUDA, DataType::FP16>>(); });
    return true;
}();

template <typename T>
__global__ void SliceKernelCuda(const T* input, T* out, int64_t numel, cuda_detail::CudaShape out_shape,
                                cuda_detail::CudaShape in_shape, int axis, int64_t start) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= numel) {
        return;
    }
    int64_t remaining = linear;
    int64_t in_offset = 0;
    for (int i = 0; i < out_shape.rank; ++i) {
        int64_t coord = remaining / out_shape.strides[i];
        remaining %= out_shape.strides[i];
        if (i == axis) {
            coord += start;
        }
        in_offset += coord * in_shape.strides[i];
    }
    out[linear] = input[in_offset];
}

template <DataType dtype>
int RunSlice(feather::operators::SliceParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }
    const auto& in_dims = param->input->dims().data();
    const auto& out_dims = param->out->dims().data();
    const int rank = static_cast<int>(in_dims.size());
    const int axis = param->axis < 0 ? param->axis + rank : param->axis;
    if (axis < 0 || axis >= rank || in_dims.size() > cuda_detail::kMaxCudaRank) {
        return -1;
    }
    int64_t start = param->start < 0 ? param->start + in_dims[static_cast<size_t>(axis)] : param->start;
    start = std::max<int64_t>(0, start);
    cuda_detail::CudaShape in_shape;
    cuda_detail::CudaShape out_shape;
    if (!cuda_detail::MakeCudaShape(in_dims, &in_shape) || !cuda_detail::MakeCudaShape(out_dims, &out_shape)) {
        return -1;
    }
    cuda_detail::DeviceBuffer<T> input;
    cuda_detail::DeviceBuffer<T> out;
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &out) != 0) {
        return -1;
    }
    SliceKernelCuda<T><<<static_cast<int>(cuda_detail::DivUp(param->out->numel(), cuda_detail::kCudaThreads)),
                        cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
        input.get(), out.get(), param->out->numel(), out_shape, in_shape, axis, start);
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) {
        return -1;
    }
    return cuda_detail::CopyDeviceToTensor(&out, param->out.get());
}

}  // namespace

template <>
int32_t SliceKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return RunSlice<DataType::FP32>(static_cast<feather::operators::SliceParam*>(param_), "CUDA::Slice::FP32");
}

template <>
int32_t SliceKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return RunSlice<DataType::FP16>(static_cast<feather::operators::SliceParam*>(param_), "CUDA::Slice::FP16");
}

void EnsureCudaSliceKernelsRegistered() { (void)g_cuda_slice_kernels_registered; }

}  // namespace kernel
}  // namespace feather
