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

template <typename T>
__global__ void Resize4DNchwKernelCuda(const T* input, T* out, int64_t total, int64_t channels, int64_t in_h,
                                       int64_t in_w, int64_t out_h, int64_t out_w, float scale_h, float scale_w) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int64_t ow = idx % out_w;
    const int64_t oh = (idx / out_w) % out_h;
    const int64_t c = (idx / (out_w * out_h)) % channels;
    const int64_t n = idx / (channels * out_h * out_w);
    int64_t ih = static_cast<int64_t>(static_cast<double>(oh) / scale_h);
    int64_t iw = static_cast<int64_t>(static_cast<double>(ow) / scale_w);
    ih = ih < 0 ? 0 : (ih >= in_h ? in_h - 1 : ih);
    iw = iw < 0 ? 0 : (iw >= in_w ? in_w - 1 : iw);
    const int64_t input_offset = ((n * channels + c) * in_h + ih) * in_w + iw;
    out[idx] = input[input_offset];
}

template <typename T>
__global__ void Resize4DNhwcKernelCuda(const T* input, T* out, int64_t total, int64_t channels, int64_t in_h,
                                       int64_t in_w, int64_t out_h, int64_t out_w, float scale_h, float scale_w) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int64_t c = idx % channels;
    const int64_t ow = (idx / channels) % out_w;
    const int64_t oh = (idx / (channels * out_w)) % out_h;
    const int64_t n = idx / (channels * out_w * out_h);
    int64_t ih = static_cast<int64_t>(static_cast<double>(oh) / scale_h);
    int64_t iw = static_cast<int64_t>(static_cast<double>(ow) / scale_w);
    ih = ih < 0 ? 0 : (ih >= in_h ? in_h - 1 : ih);
    iw = iw < 0 ? 0 : (iw >= in_w ? in_w - 1 : iw);
    const int64_t input_offset = ((n * in_h + ih) * in_w + iw) * channels + c;
    out[idx] = input[input_offset];
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
    const auto total = param->out->numel();
    bool used_direct_4d = false;
    if (in_dims.size() == 4) {
        ImageShape4D input_shape;
        ImageShape4D output_shape;
        if (DecodeImageShape4D(param->input->dims().data(), param->input->layout(), &input_shape) &&
            DecodeImageShape4D(param->out->dims().data(), param->out->layout(), &output_shape)) {
            const float scale_h = param->scales[IsChannelLastLayout(param->input->layout()) ? 1 : 2];
            const float scale_w = param->scales[IsChannelLastLayout(param->input->layout()) ? 2 : 3];
            if (IsChannelLastLayout(param->input->layout())) {
                Resize4DNhwcKernelCuda<T><<<static_cast<int>(cuda_detail::DivUp(total, cuda_detail::kCudaThreads)),
                                           cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
                    input.get(), out.get(), total, input_shape.c, input_shape.h, input_shape.w, output_shape.h,
                    output_shape.w, scale_h, scale_w);
            } else {
                Resize4DNchwKernelCuda<T><<<static_cast<int>(cuda_detail::DivUp(total, cuda_detail::kCudaThreads)),
                                           cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
                    input.get(), out.get(), total, input_shape.c, input_shape.h, input_shape.w, output_shape.h,
                    output_shape.w, scale_h, scale_w);
            }
            used_direct_4d = true;
            SetLastCudaResizeBackend(CudaResizeBackend::kDirect4D);
        }
    }
    if (!used_direct_4d) {
        ResizeKernelCuda<T><<<static_cast<int>(cuda_detail::DivUp(total, cuda_detail::kCudaThreads)),
                             cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
            input.get(), out.get(), total, in_shape, out_shape, scales);
        SetLastCudaResizeBackend(CudaResizeBackend::kGeneric);
    }
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
