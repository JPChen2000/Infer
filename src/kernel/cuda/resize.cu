#include "src/kernel/resize.h"

#include <memory>

#include "core/kernel.h"
#include "src/kernel/cuda/kernel_io.cuh"
#include "src/operator/params.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

struct CudaResizeScales {
    int rank{0};
    float values[cuda_detail::kMaxCudaRank]{};
};

bool g_cuda_resize_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Resize",
                              []() { return std::make_unique<ResizeKernel<DeviceType::CUDA, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Resize",
                              []() { return std::make_unique<ResizeKernel<DeviceType::CUDA, DataType::FP16>>(); });
    return true;
}();

template <typename T>
__global__ void ResizeKernelCuda(const T* input, T* out, int64_t numel, cuda_detail::CudaShape in_shape,
                                 cuda_detail::CudaShape out_shape, CudaResizeScales scales) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= numel) {
        return;
    }
    int64_t remaining = linear;
    int64_t in_offset = 0;
    for (int axis = 0; axis < out_shape.rank; ++axis) {
        const int64_t out_coord = remaining / out_shape.strides[axis];
        remaining %= out_shape.strides[axis];
        int64_t in_coord = static_cast<int64_t>(static_cast<double>(out_coord) / scales.values[axis]);
        in_coord = in_coord < 0 ? 0 : in_coord;
        in_coord = in_coord >= in_shape.dims[axis] ? in_shape.dims[axis] - 1 : in_coord;
        in_offset += in_coord * in_shape.strides[axis];
    }
    out[linear] = input[in_offset];
}

template <DataType dtype>
int RunResize(feather::operators::ResizeParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }
    const auto& in_dims = param->input->dims().data();
    const auto& out_dims = param->out->dims().data();
    if (in_dims.size() != out_dims.size() || in_dims.size() != param->scales.size() ||
        in_dims.size() > cuda_detail::kMaxCudaRank) {
        return -1;
    }
    cuda_detail::CudaShape in_shape;
    cuda_detail::CudaShape out_shape;
    CudaResizeScales scales;
    if (!cuda_detail::MakeCudaShape(in_dims, &in_shape) || !cuda_detail::MakeCudaShape(out_dims, &out_shape)) {
        return -1;
    }
    scales.rank = static_cast<int>(param->scales.size());
    for (size_t i = 0; i < param->scales.size(); ++i) {
        scales.values[i] = param->scales[i];
    }
    cuda_detail::DeviceBuffer<T> input;
    cuda_detail::DeviceBuffer<T> out;
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &out) != 0) {
        return -1;
    }
    ResizeKernelCuda<T><<<static_cast<int>(cuda_detail::DivUp(param->out->numel(), cuda_detail::kCudaThreads)),
                         cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
        input.get(), out.get(), param->out->numel(), in_shape, out_shape, scales);
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) {
        return -1;
    }
    return cuda_detail::CopyDeviceToTensor(&out, param->out.get());
}

}  // namespace

template <>
int32_t ResizeKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return RunResize<DataType::FP32>(static_cast<feather::operators::ResizeParam*>(param_), "CUDA::Resize::FP32");
}

template <>
int32_t ResizeKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return RunResize<DataType::FP16>(static_cast<feather::operators::ResizeParam*>(param_), "CUDA::Resize::FP16");
}

void EnsureCudaResizeKernelsRegistered() { (void)g_cuda_resize_kernels_registered; }

}  // namespace kernel
}  // namespace feather
