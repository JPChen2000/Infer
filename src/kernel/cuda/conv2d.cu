#include "src/kernel/conv2d.h"

#include <algorithm>
#include <memory>

#include "core/kernel.h"
#include "src/kernel/cuda/kernel_io.cuh"
#include "src/operator/params.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_conv2d_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Conv2D",
                              []() { return std::make_unique<Conv2DKernel<DeviceType::CUDA, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Conv2D",
                              []() { return std::make_unique<Conv2DKernel<DeviceType::CUDA, DataType::FP16>>(); });
    return true;
}();

template <typename T>
__global__ void Conv2D4DKernelCuda(const T* input, const T* weight, const T* bias, T* out, int64_t batch,
                                   int64_t in_c, int64_t in_h, int64_t in_w, int64_t out_c, int64_t kernel_c,
                                   int64_t kernel_h, int64_t kernel_w, int64_t out_h, int64_t out_w, int stride_h,
                                   int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, int group) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * out_c * out_h * out_w;
    if (idx >= total) {
        return;
    }
    const int64_t ow = idx % out_w;
    const int64_t oh = (idx / out_w) % out_h;
    const int64_t oc = (idx / (out_w * out_h)) % out_c;
    const int64_t n = idx / (out_c * out_h * out_w);
    const int64_t out_c_per_group = out_c / group;
    const int64_t in_c_per_group = in_c / group;
    const int64_t g = oc / out_c_per_group;
    float sum = bias != nullptr ? cuda_detail::ReadDevice(bias, oc) : 0.0f;
    for (int64_t ic = 0; ic < in_c_per_group; ++ic) {
        const int64_t global_ic = g * in_c_per_group + ic;
        for (int64_t kh = 0; kh < kernel_h; ++kh) {
            const int64_t ih = oh * stride_h + kh * dilation_h - pad_h;
            if (ih < 0 || ih >= in_h) {
                continue;
            }
            for (int64_t kw = 0; kw < kernel_w; ++kw) {
                const int64_t iw = ow * stride_w + kw * dilation_w - pad_w;
                if (iw < 0 || iw >= in_w) {
                    continue;
                }
                const int64_t input_offset = ((n * in_c + global_ic) * in_h + ih) * in_w + iw;
                const int64_t weight_offset = ((oc * kernel_c + ic) * kernel_h + kh) * kernel_w + kw;
                sum += cuda_detail::ReadDevice(input, input_offset) * cuda_detail::ReadDevice(weight, weight_offset);
            }
        }
    }
    cuda_detail::WriteDevice(out, idx, sum);
}

template <typename T>
__global__ void PointwiseConv2D4DKernelCuda(const T* input, const T* weight, const T* bias, T* out, int64_t batch,
                                            int64_t in_c, int64_t in_h, int64_t in_w, int64_t out_c,
                                            int64_t kernel_c, int64_t out_h, int64_t out_w, int stride_h,
                                            int stride_w, int pad_h, int pad_w, int group) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * out_c * out_h * out_w;
    if (idx >= total) {
        return;
    }
    const int64_t ow = idx % out_w;
    const int64_t oh = (idx / out_w) % out_h;
    const int64_t oc = (idx / (out_w * out_h)) % out_c;
    const int64_t n = idx / (out_c * out_h * out_w);
    const int64_t ih = oh * stride_h - pad_h;
    const int64_t iw = ow * stride_w - pad_w;
    float sum = bias != nullptr ? cuda_detail::ReadDevice(bias, oc) : 0.0f;
    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
        const int64_t out_c_per_group = out_c / group;
        const int64_t in_c_per_group = in_c / group;
        const int64_t g = oc / out_c_per_group;
        for (int64_t ic = 0; ic < in_c_per_group; ++ic) {
            const int64_t global_ic = g * in_c_per_group + ic;
            const int64_t input_offset = ((n * in_c + global_ic) * in_h + ih) * in_w + iw;
            const int64_t weight_offset = oc * kernel_c + ic;
            sum += cuda_detail::ReadDevice(input, input_offset) * cuda_detail::ReadDevice(weight, weight_offset);
        }
    }
    cuda_detail::WriteDevice(out, idx, sum);
}

template <typename T>
__global__ void DepthwiseConv2D4DKernelCuda(const T* input, const T* weight, const T* bias, T* out, int64_t batch,
                                            int64_t in_c, int64_t in_h, int64_t in_w, int64_t out_c,
                                            int64_t kernel_h, int64_t kernel_w, int64_t out_h, int64_t out_w,
                                            int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h,
                                            int dilation_w) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * out_c * out_h * out_w;
    if (idx >= total) {
        return;
    }
    const int64_t ow = idx % out_w;
    const int64_t oh = (idx / out_w) % out_h;
    const int64_t oc = (idx / (out_w * out_h)) % out_c;
    const int64_t n = idx / (out_c * out_h * out_w);
    const int64_t depth_multiplier = out_c / in_c;
    const int64_t ic = oc / depth_multiplier;
    float sum = bias != nullptr ? cuda_detail::ReadDevice(bias, oc) : 0.0f;
    for (int64_t kh = 0; kh < kernel_h; ++kh) {
        const int64_t ih = oh * stride_h + kh * dilation_h - pad_h;
        if (ih < 0 || ih >= in_h) {
            continue;
        }
        for (int64_t kw = 0; kw < kernel_w; ++kw) {
            const int64_t iw = ow * stride_w + kw * dilation_w - pad_w;
            if (iw < 0 || iw >= in_w) {
                continue;
            }
            const int64_t input_offset = ((n * in_c + ic) * in_h + ih) * in_w + iw;
            const int64_t weight_offset = ((oc * 1) * kernel_h + kh) * kernel_w + kw;
            sum += cuda_detail::ReadDevice(input, input_offset) * cuda_detail::ReadDevice(weight, weight_offset);
        }
    }
    cuda_detail::WriteDevice(out, idx, sum);
}

template <typename T>
__global__ void Conv2D2DKernelCuda(const T* input, const T* weight, const T* bias, T* out, int64_t in_h,
                                   int64_t in_w, int64_t kernel_h, int64_t kernel_w, int64_t out_h, int64_t out_w,
                                   int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h,
                                   int dilation_w) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = out_h * out_w;
    if (idx >= total) {
        return;
    }
    const int64_t oh = idx / out_w;
    const int64_t ow = idx % out_w;
    float sum = 0.0f;
    for (int64_t kh = 0; kh < kernel_h; ++kh) {
        const int64_t ih = oh * stride_h + kh * dilation_h - pad_h;
        if (ih < 0 || ih >= in_h) {
            continue;
        }
        for (int64_t kw = 0; kw < kernel_w; ++kw) {
            const int64_t iw = ow * stride_w + kw * dilation_w - pad_w;
            if (iw < 0 || iw >= in_w) {
                continue;
            }
            sum += cuda_detail::ReadDevice(input, ih * in_w + iw) * cuda_detail::ReadDevice(weight, kh * kernel_w + kw);
        }
    }
    if (bias != nullptr) {
        sum += cuda_detail::ReadDevice(bias, idx);
    }
    cuda_detail::WriteDevice(out, idx, sum);
}

template <DataType dtype>
int RunConv2D(feather::operators::Conv2dParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->input == nullptr || param->w == nullptr || param->out == nullptr) {
        return -1;
    }
    cuda_detail::DeviceBuffer<T> input;
    cuda_detail::DeviceBuffer<T> weight;
    cuda_detail::DeviceBuffer<T> bias;
    cuda_detail::DeviceBuffer<T> out;
    T* bias_ptr = nullptr;
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0 ||
        cuda_detail::CopyTensorToDevice(param->w.get(), &weight) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &out) != 0) {
        return -1;
    }
    if (param->bias != nullptr && param->bias->IsInitialized()) {
        if (cuda_detail::CopyTensorToDevice(param->bias.get(), &bias) != 0) {
            return -1;
        }
        bias_ptr = bias.get();
    }
    if (param->input->dims().size() == 2 && param->w->dims().size() == 2) {
        const int64_t out_numel = param->out->numel();
        Conv2D2DKernelCuda<T><<<static_cast<int>(cuda_detail::DivUp(out_numel, cuda_detail::kCudaThreads)),
                               cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
            input.get(), weight.get(), bias_ptr, out.get(), param->input->dims()[0], param->input->dims()[1],
            param->w->dims()[0], param->w->dims()[1], param->out->dims()[0], param->out->dims()[1],
            param->stride_h, param->stride_w, param->pad_h, param->pad_w, param->dilation_h, param->dilation_w);
    } else if (param->input->dims().size() == 4 && param->w->dims().size() == 4) {
        const int64_t out_numel = param->out->numel();
        const auto group = std::max(1, param->group);
        const bool is_pointwise = param->w->dims()[2] == 1 && param->w->dims()[3] == 1 && param->dilation_h == 1 &&
                                  param->dilation_w == 1;
        const bool is_depthwise = group == param->input->dims()[1] && param->w->dims()[1] == 1 &&
                                  param->w->dims()[0] % param->input->dims()[1] == 0;
        if (is_pointwise) {
            PointwiseConv2D4DKernelCuda<T>
                <<<static_cast<int>(cuda_detail::DivUp(out_numel, cuda_detail::kCudaThreads)),
                   cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
                    input.get(), weight.get(), bias_ptr, out.get(), param->input->dims()[0], param->input->dims()[1],
                    param->input->dims()[2], param->input->dims()[3], param->w->dims()[0], param->w->dims()[1],
                    param->out->dims()[2], param->out->dims()[3], param->stride_h, param->stride_w, param->pad_h,
                    param->pad_w, group);
        } else if (is_depthwise) {
            DepthwiseConv2D4DKernelCuda<T>
                <<<static_cast<int>(cuda_detail::DivUp(out_numel, cuda_detail::kCudaThreads)),
                   cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
                    input.get(), weight.get(), bias_ptr, out.get(), param->input->dims()[0], param->input->dims()[1],
                    param->input->dims()[2], param->input->dims()[3], param->w->dims()[0], param->w->dims()[2],
                    param->w->dims()[3], param->out->dims()[2], param->out->dims()[3], param->stride_h,
                    param->stride_w, param->pad_h, param->pad_w, param->dilation_h, param->dilation_w);
        } else {
            Conv2D4DKernelCuda<T><<<static_cast<int>(cuda_detail::DivUp(out_numel, cuda_detail::kCudaThreads)),
                                   cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
                input.get(), weight.get(), bias_ptr, out.get(), param->input->dims()[0], param->input->dims()[1],
                param->input->dims()[2], param->input->dims()[3], param->w->dims()[0], param->w->dims()[1],
                param->w->dims()[2], param->w->dims()[3], param->out->dims()[2], param->out->dims()[3],
                param->stride_h, param->stride_w, param->pad_h, param->pad_w, param->dilation_h, param->dilation_w,
                group);
        }
    } else {
        return -1;
    }
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) {
        return -1;
    }
    return cuda_detail::CopyDeviceToTensor(&out, param->out.get());
}

}  // namespace

template <>
int32_t Conv2DKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return RunConv2D<DataType::FP32>(static_cast<feather::operators::Conv2dParam*>(param_), "CUDA::Conv2D::FP32");
}

template <>
int32_t Conv2DKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return RunConv2D<DataType::FP16>(static_cast<feather::operators::Conv2dParam*>(param_), "CUDA::Conv2D::FP16");
}

void EnsureCudaConv2DKernelsRegistered() { (void)g_cuda_conv2d_kernels_registered; }

}  // namespace kernel
}  // namespace feather
