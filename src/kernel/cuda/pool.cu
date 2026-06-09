#include "src/kernel/pool.h"

#include <limits>
#include <memory>

#include "core/kernel.h"
#include "src/kernel/cuda/kernel_io.cuh"
#include "src/kernel/cuda/runtime.h"
#include "src/operator/params.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_pool_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "AvgPool",
                              []() { return std::make_unique<AvgPoolKernel<DeviceType::CUDA, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "AvgPool",
                              []() { return std::make_unique<AvgPoolKernel<DeviceType::CUDA, DataType::FP16>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "MaxPool",
                              []() { return std::make_unique<MaxPoolKernel<DeviceType::CUDA, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "MaxPool",
                              []() { return std::make_unique<MaxPoolKernel<DeviceType::CUDA, DataType::FP16>>(); });
    return true;
}();

#ifdef FEATHER_WITH_CUDNN
bool CudnnCheck(cudnnStatus_t status) {
    return status == CUDNN_STATUS_SUCCESS;
}

bool SetCudnnTensor4dDescriptor(cudnnTensorDescriptor_t desc, DataType dtype, DataLayout layout,
                                const ImageShape4D& shape) {
    const auto tensor_format =
        NormalizeDataLayout(layout) == DataLayout::NHWC ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;
    const auto tensor_dtype = dtype == DataType::FP16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;
    return CudnnCheck(cudnnSetTensor4dDescriptor(desc, tensor_format, tensor_dtype, static_cast<int>(shape.n),
                                                 static_cast<int>(shape.c), static_cast<int>(shape.h),
                                                 static_cast<int>(shape.w)));
}

template <DataType dtype, bool IsMax>
bool RunPoolWithCudnn(feather::operators::PoolParam* param) {
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return false;
    }
    if (param->input->dims().size() != 4 || param->out->dims().size() != 4) {
        return false;
    }

    auto handle = cuda_detail::CudnnHandle();
    if (handle == nullptr) {
        return false;
    }

    ImageShape4D input_shape;
    ImageShape4D output_shape;
    if (!DecodeImageShape4D(param->input->dims().data(), param->input->layout(), &input_shape) ||
        !DecodeImageShape4D(param->out->dims().data(), param->out->layout(), &output_shape)) {
        return false;
    }

    cuda_detail::DeviceBuffer<T> input;
    cuda_detail::DeviceBuffer<T> output;
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &output) != 0) {
        return false;
    }

    cudnnTensorDescriptor_t input_desc = nullptr;
    cudnnTensorDescriptor_t output_desc = nullptr;
    cudnnPoolingDescriptor_t pooling_desc = nullptr;
    if (!CudnnCheck(cudnnCreateTensorDescriptor(&input_desc)) ||
        !CudnnCheck(cudnnCreateTensorDescriptor(&output_desc)) ||
        !CudnnCheck(cudnnCreatePoolingDescriptor(&pooling_desc))) {
        if (input_desc != nullptr) {
            cudnnDestroyTensorDescriptor(input_desc);
        }
        if (output_desc != nullptr) {
            cudnnDestroyTensorDescriptor(output_desc);
        }
        if (pooling_desc != nullptr) {
            cudnnDestroyPoolingDescriptor(pooling_desc);
        }
        return false;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const bool ok =
        SetCudnnTensor4dDescriptor(input_desc, dtype, param->input->layout(), input_shape) &&
        SetCudnnTensor4dDescriptor(output_desc, dtype, param->out->layout(), output_shape) &&
        CudnnCheck(cudnnSetPooling2dDescriptor(pooling_desc,
                                               IsMax ? CUDNN_POOLING_MAX : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
                                               CUDNN_NOT_PROPAGATE_NAN, param->kernel_h, param->kernel_w, param->pad_h,
                                               param->pad_w, param->stride_h, param->stride_w)) &&
        CudnnCheck(cudnnPoolingForward(handle, pooling_desc, &alpha, input_desc, input.get(), &beta, output_desc,
                                       output.get()));

    cudnnDestroyPoolingDescriptor(pooling_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyTensorDescriptor(input_desc);

    if (!ok || cuda_detail::CudaCheck(cudaGetLastError()) != 0) {
        return false;
    }
    SetLastCudaPoolBackend(CudaPoolBackend::kCudnn);
    return cuda_detail::CopyDeviceToTensor(&output, param->out.get()) == 0;
}
#endif

template <typename T, bool IsMax>
__global__ void PoolKernelCuda(const T* input, T* out, int64_t batch, int64_t channels, int64_t in_h, int64_t in_w,
                               int64_t out_h, int64_t out_w, int kernel_h, int kernel_w, int stride_h,
                               int stride_w, int pad_h, int pad_w, bool is_channel_last) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * channels * out_h * out_w;
    if (idx >= total) {
        return;
    }
    const int64_t ow = idx % out_w;
    const int64_t oh = (idx / out_w) % out_h;
    const int64_t c = (idx / (out_w * out_h)) % channels;
    const int64_t n = idx / (channels * out_h * out_w);
    float value = IsMax ? -3.4028234663852886e+38F : 0.0f;
    int count = 0;
    for (int kh = 0; kh < kernel_h; ++kh) {
        const int64_t ih = oh * stride_h + kh - pad_h;
        if (ih < 0 || ih >= in_h) {
            continue;
        }
        for (int kw = 0; kw < kernel_w; ++kw) {
            const int64_t iw = ow * stride_w + kw - pad_w;
            if (iw < 0 || iw >= in_w) {
                continue;
            }
            const int64_t input_offset = is_channel_last ? ((n * in_h + ih) * in_w + iw) * channels + c
                                                         : ((n * channels + c) * in_h + ih) * in_w + iw;
            const float input_value = cuda_detail::ReadDevice(input, input_offset);
            if constexpr (IsMax) {
                value = fmaxf(value, input_value);
            } else {
                value += input_value;
                ++count;
            }
        }
    }
    if constexpr (!IsMax) {
        value = count == 0 ? 0.0f : value / static_cast<float>(count);
    }
    const int64_t output_offset = is_channel_last ? ((n * out_h + oh) * out_w + ow) * channels + c : idx;
    cuda_detail::WriteDevice(out, output_offset, value);
}

template <DataType dtype, bool IsMax>
int RunPool(feather::operators::PoolParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }
#ifdef FEATHER_WITH_CUDNN
    if (RunPoolWithCudnn<dtype, IsMax>(param)) {
        return 0;
    }
#endif
    SetLastCudaPoolBackend(CudaPoolBackend::kFallback);
    const bool is_4d = param->input->dims().size() == 4;
    ImageShape4D input_shape;
    ImageShape4D output_shape;
    if (is_4d &&
        (!DecodeImageShape4D(param->input->dims().data(), param->input->layout(), &input_shape) ||
         !DecodeImageShape4D(param->out->dims().data(), param->out->layout(), &output_shape))) {
        return -1;
    }
    const int64_t batch = is_4d ? input_shape.n : 1;
    const int64_t channels = is_4d ? input_shape.c : 1;
    const int64_t in_h = is_4d ? input_shape.h : param->input->dims()[0];
    const int64_t in_w = is_4d ? input_shape.w : param->input->dims()[1];
    const int64_t out_h = is_4d ? output_shape.h : param->out->dims()[0];
    const int64_t out_w = is_4d ? output_shape.w : param->out->dims()[1];
    const bool is_channel_last = is_4d && IsChannelLastLayout(param->input->layout());
    cuda_detail::DeviceBuffer<T> input;
    cuda_detail::DeviceBuffer<T> out;
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &out) != 0) {
        return -1;
    }
    PoolKernelCuda<T, IsMax><<<static_cast<int>(cuda_detail::DivUp(param->out->numel(), cuda_detail::kCudaThreads)),
                              cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
        input.get(), out.get(), batch, channels, in_h, in_w, out_h, out_w, param->kernel_h, param->kernel_w,
        param->stride_h, param->stride_w, param->pad_h, param->pad_w, is_channel_last);
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) {
        return -1;
    }
    return cuda_detail::CopyDeviceToTensor(&out, param->out.get());
}

}  // namespace

template <>
int32_t AvgPoolKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return RunPool<DataType::FP32, false>(static_cast<feather::operators::PoolParam*>(param_),
                                         "CUDA::AvgPool::FP32");
}

template <>
int32_t AvgPoolKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return RunPool<DataType::FP16, false>(static_cast<feather::operators::PoolParam*>(param_),
                                         "CUDA::AvgPool::FP16");
}

template <>
int32_t MaxPoolKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return RunPool<DataType::FP32, true>(static_cast<feather::operators::PoolParam*>(param_),
                                        "CUDA::MaxPool::FP32");
}

template <>
int32_t MaxPoolKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return RunPool<DataType::FP16, true>(static_cast<feather::operators::PoolParam*>(param_),
                                        "CUDA::MaxPool::FP16");
}

void EnsureCudaPoolKernelsRegistered() { (void)g_cuda_pool_kernels_registered; }

}  // namespace kernel
}  // namespace feather
