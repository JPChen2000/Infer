#include "src/kernel/resize_concat.h"

#include <memory>

#include "core/kernel.h"
#include "src/kernel/cuda/kernel_io.cuh"
#include "src/operator/params.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_resize_concat_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(
        DeviceType::CUDA, DataType::FP32, "ResizeConcat",
        []() { return std::make_unique<ResizeConcatKernel<DeviceType::CUDA, DataType::FP32>>(); });
    dispatcher.registerKernel(
        DeviceType::CUDA, DataType::FP16, "ResizeConcat",
        []() { return std::make_unique<ResizeConcatKernel<DeviceType::CUDA, DataType::FP16>>(); });
    return true;
}();

template <typename T>
__global__ void ResizeConcatNchwKernelCuda(const T* resize_input, const T* concat_input, T* out, int64_t total,
                                           int64_t resize_channels, int64_t concat_channels, int64_t in_h, int64_t in_w,
                                           int64_t out_h, int64_t out_w, float scale_h, float scale_w,
                                           int resize_input_index) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int64_t ow = idx % out_w;
    const int64_t oh = (idx / out_w) % out_h;
    const int64_t oc = (idx / (out_w * out_h)) % (resize_channels + concat_channels);
    const int64_t n = idx / ((resize_channels + concat_channels) * out_h * out_w);

    const bool use_resize = resize_input_index == 0 ? (oc < resize_channels) : (oc >= concat_channels);
    if (use_resize) {
        const int64_t rc = resize_input_index == 0 ? oc : (oc - concat_channels);
        int64_t ih = static_cast<int64_t>(static_cast<double>(oh) / scale_h);
        int64_t iw = static_cast<int64_t>(static_cast<double>(ow) / scale_w);
        ih = ih < 0 ? 0 : (ih >= in_h ? in_h - 1 : ih);
        iw = iw < 0 ? 0 : (iw >= in_w ? in_w - 1 : iw);
        const int64_t input_offset = ((n * resize_channels + rc) * in_h + ih) * in_w + iw;
        cuda_detail::WriteDevice(out, idx, cuda_detail::ReadDevice(resize_input, input_offset));
        return;
    }

    const int64_t cc = resize_input_index == 0 ? (oc - resize_channels) : oc;
    const int64_t input_offset = ((n * concat_channels + cc) * out_h + oh) * out_w + ow;
    cuda_detail::WriteDevice(out, idx, cuda_detail::ReadDevice(concat_input, input_offset));
}

template <typename T>
__global__ void ResizeConcatNhwcKernelCuda(const T* resize_input, const T* concat_input, T* out, int64_t total,
                                           int64_t resize_channels, int64_t concat_channels, int64_t in_h, int64_t in_w,
                                           int64_t out_h, int64_t out_w, float scale_h, float scale_w,
                                           int resize_input_index) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t out_channels = resize_channels + concat_channels;
    if (idx >= total) {
        return;
    }
    const int64_t oc = idx % out_channels;
    const int64_t ow = (idx / out_channels) % out_w;
    const int64_t oh = (idx / (out_channels * out_w)) % out_h;
    const int64_t n = idx / (out_channels * out_w * out_h);

    const bool use_resize = resize_input_index == 0 ? (oc < resize_channels) : (oc >= concat_channels);
    if (use_resize) {
        const int64_t rc = resize_input_index == 0 ? oc : (oc - concat_channels);
        int64_t ih = static_cast<int64_t>(static_cast<double>(oh) / scale_h);
        int64_t iw = static_cast<int64_t>(static_cast<double>(ow) / scale_w);
        ih = ih < 0 ? 0 : (ih >= in_h ? in_h - 1 : ih);
        iw = iw < 0 ? 0 : (iw >= in_w ? in_w - 1 : iw);
        const int64_t input_offset = ((n * in_h + ih) * in_w + iw) * resize_channels + rc;
        cuda_detail::WriteDevice(out, idx, cuda_detail::ReadDevice(resize_input, input_offset));
        return;
    }

    const int64_t cc = resize_input_index == 0 ? (oc - resize_channels) : oc;
    const int64_t input_offset = ((n * out_h + oh) * out_w + ow) * concat_channels + cc;
    cuda_detail::WriteDevice(out, idx, cuda_detail::ReadDevice(concat_input, input_offset));
}

template <DataType dtype>
int RunResizeConcat(feather::operators::ResizeConcatParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->resize_input == nullptr || param->concat_input == nullptr || param->out == nullptr) {
        return -1;
    }

    ImageShape4D resize_input_shape;
    ImageShape4D concat_input_shape;
    ImageShape4D output_shape;
    if (!DecodeImageShape4D(param->resize_input->dims().data(), param->resize_input->layout(), &resize_input_shape) ||
        !DecodeImageShape4D(param->concat_input->dims().data(), param->concat_input->layout(), &concat_input_shape) ||
        !DecodeImageShape4D(param->out->dims().data(), param->out->layout(), &output_shape)) {
        return -1;
    }

    cuda_detail::DeviceBuffer<T> resize_input;
    cuda_detail::DeviceBuffer<T> concat_input;
    cuda_detail::DeviceBuffer<T> out;
    if (cuda_detail::CopyTensorToDevice(param->resize_input.get(), &resize_input) != 0 ||
        cuda_detail::CopyTensorToDevice(param->concat_input.get(), &concat_input) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &out) != 0) {
        return -1;
    }

    const auto total = param->out->numel();
    const bool channel_last = IsChannelLastLayout(param->out->layout());
    const float scale_h = param->scales[channel_last ? 1 : 2];
    const float scale_w = param->scales[channel_last ? 2 : 3];
    if (channel_last) {
        ResizeConcatNhwcKernelCuda<T><<<static_cast<int>(cuda_detail::DivUp(total, cuda_detail::kCudaThreads)),
                                       cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
            resize_input.get(), concat_input.get(), out.get(), total, resize_input_shape.c, concat_input_shape.c,
            resize_input_shape.h, resize_input_shape.w, output_shape.h, output_shape.w, scale_h, scale_w,
            param->resize_input_index);
    } else {
        ResizeConcatNchwKernelCuda<T><<<static_cast<int>(cuda_detail::DivUp(total, cuda_detail::kCudaThreads)),
                                       cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
            resize_input.get(), concat_input.get(), out.get(), total, resize_input_shape.c, concat_input_shape.c,
            resize_input_shape.h, resize_input_shape.w, output_shape.h, output_shape.w, scale_h, scale_w,
            param->resize_input_index);
    }
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) {
        return -1;
    }
    return cuda_detail::CopyDeviceToTensor(&out, param->out.get());
}

}  // namespace

template <>
int32_t ResizeConcatKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return RunResizeConcat<DataType::FP32>(static_cast<feather::operators::ResizeConcatParam*>(param_),
                                           "CUDA::ResizeConcat::FP32");
}

template <>
int32_t ResizeConcatKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return RunResizeConcat<DataType::FP16>(static_cast<feather::operators::ResizeConcatParam*>(param_),
                                           "CUDA::ResizeConcat::FP16");
}

void EnsureCudaResizeConcatKernelsRegistered() { (void)g_cuda_resize_concat_kernels_registered; }
void EnsureResizeConcatKernelsRegistered() { EnsureCudaResizeConcatKernelsRegistered(); }

}  // namespace kernel
}  // namespace feather
