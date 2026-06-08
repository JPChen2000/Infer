#include "src/kernel/pool.h"

#include <memory>

#include "core/kernel.h"
#include "src/kernel/cuda/kernel_io.cuh"
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

template <typename T, bool IsMax>
__global__ void PoolKernelCuda(const T* input, T* out, int64_t batch, int64_t channels, int64_t in_h, int64_t in_w,
                               int64_t out_h, int64_t out_w, int kernel_h, int kernel_w, int stride_h,
                               int stride_w, int pad_h, int pad_w, bool is_nchw) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * channels * out_h * out_w;
    if (idx >= total) {
        return;
    }
    const int64_t ow = idx % out_w;
    const int64_t oh = (idx / out_w) % out_h;
    const int64_t c = is_nchw ? (idx / (out_w * out_h)) % channels : 0;
    const int64_t n = is_nchw ? idx / (channels * out_h * out_w) : 0;
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
            const int64_t input_offset = is_nchw ? ((n * channels + c) * in_h + ih) * in_w + iw : ih * in_w + iw;
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
    cuda_detail::WriteDevice(out, idx, value);
}

template <DataType dtype, bool IsMax>
int RunPool(feather::operators::PoolParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }
    const bool is_nchw = param->input->dims().size() == 4;
    const int64_t batch = is_nchw ? param->input->dims()[0] : 1;
    const int64_t channels = is_nchw ? param->input->dims()[1] : 1;
    const int64_t in_h = param->input->dims()[is_nchw ? 2 : 0];
    const int64_t in_w = param->input->dims()[is_nchw ? 3 : 1];
    const int64_t out_h = param->out->dims()[is_nchw ? 2 : 0];
    const int64_t out_w = param->out->dims()[is_nchw ? 3 : 1];
    cuda_detail::DeviceBuffer<T> input;
    cuda_detail::DeviceBuffer<T> out;
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &out) != 0) {
        return -1;
    }
    PoolKernelCuda<T, IsMax><<<static_cast<int>(cuda_detail::DivUp(param->out->numel(), cuda_detail::kCudaThreads)),
                              cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
        input.get(), out.get(), batch, channels, in_h, in_w, out_h, out_w, param->kernel_h, param->kernel_w,
        param->stride_h, param->stride_w, param->pad_h, param->pad_w, is_nchw);
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
