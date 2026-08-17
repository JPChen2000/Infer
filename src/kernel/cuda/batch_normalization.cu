#include "src/kernel/batch_normalization.h"

#include <cmath>
#include <memory>

#include "core/kernel.h"
#include "src/kernel/cuda/kernel_io.cuh"
#include "src/operator/params.h"
#include "util/timer.h"
#include "util/types.h"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_batch_normalization_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(
        DeviceType::CUDA, DataType::FP32, "BatchNormalization",
        []() { return std::make_unique<BatchNormalizationKernel<DeviceType::CUDA, DataType::FP32>>(); });
    dispatcher.registerKernel(
        DeviceType::CUDA, DataType::FP16, "BatchNormalization",
        []() { return std::make_unique<BatchNormalizationKernel<DeviceType::CUDA, DataType::FP16>>(); });
    return true;
}();

struct CudaBatchNormalizationShape {
    int64_t n{0};
    int64_t c{0};
    int64_t h{0};
    int64_t w{0};
    int channel_last{0};
};

template <typename T>
__global__ void BatchNormalizationKernelCuda(const T* input, const T* scale, const T* bias, const T* mean,
                                             const T* var, T* out, int64_t numel,
                                             CudaBatchNormalizationShape shape, float epsilon) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= numel) {
        return;
    }

    int64_t remaining = index;
    int64_t channel = 0;
    if (shape.channel_last != 0) {
        channel = remaining % shape.c;
    } else {
        remaining /= shape.w;
        remaining /= shape.h;
        channel = remaining % shape.c;
    }
    const float x = cuda_detail::ReadDevice(input, index);
    const float scale_value = cuda_detail::ReadDevice(scale, channel);
    const float bias_value = cuda_detail::ReadDevice(bias, channel);
    const float mean_value = cuda_detail::ReadDevice(mean, channel);
    const float var_value = cuda_detail::ReadDevice(var, channel);
    cuda_detail::WriteDevice(out, index, (x - mean_value) * rsqrtf(var_value + epsilon) * scale_value + bias_value);
}

template <DataType dtype>
int RunBatchNormalization(feather::operators::BatchNormParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->input == nullptr || param->scale == nullptr || param->bias == nullptr ||
        param->mean == nullptr || param->var == nullptr || param->out == nullptr || param->input->dims().size() != 4 ||
        param->input->data_type() != dtype || param->scale->data_type() != dtype || param->bias->data_type() != dtype ||
        param->mean->data_type() != dtype || param->var->data_type() != dtype) {
        return -1;
    }

    ImageShape4D host_shape{};
    const DataLayout layout = NormalizeDataLayout(param->input->layout());
    if (!DecodeImageShape4D(param->input->dims().data(), layout, &host_shape) ||
        param->scale->numel() != host_shape.c || param->bias->numel() != host_shape.c ||
        param->mean->numel() != host_shape.c || param->var->numel() != host_shape.c) {
        return -1;
    }

    cuda_detail::DeviceBuffer<T> input;
    cuda_detail::DeviceBuffer<T> scale;
    cuda_detail::DeviceBuffer<T> bias;
    cuda_detail::DeviceBuffer<T> mean;
    cuda_detail::DeviceBuffer<T> var;
    cuda_detail::DeviceBuffer<T> out;
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0 ||
        cuda_detail::CopyTensorToDevice(param->scale.get(), &scale) != 0 ||
        cuda_detail::CopyTensorToDevice(param->bias.get(), &bias) != 0 ||
        cuda_detail::CopyTensorToDevice(param->mean.get(), &mean) != 0 ||
        cuda_detail::CopyTensorToDevice(param->var.get(), &var) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &out) != 0) {
        return -1;
    }

    param->out->set_layout(param->input->layout());
    const CudaBatchNormalizationShape shape{host_shape.n, host_shape.c, host_shape.h, host_shape.w,
                                             IsChannelLastLayout(layout) ? 1 : 0};
    const int64_t numel = param->out->numel();
    BatchNormalizationKernelCuda<T><<<static_cast<int>(cuda_detail::DivUp(numel, cuda_detail::kCudaThreads)),
                                      cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
        input.get(), scale.get(), bias.get(), mean.get(), var.get(), out.get(), numel, shape, param->epsilon);
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) {
        return -1;
    }
    return cuda_detail::CopyDeviceToTensor(&out, param->out.get());
}

}  // namespace

template <>
int32_t BatchNormalizationKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return RunBatchNormalization<DataType::FP32>(static_cast<feather::operators::BatchNormParam*>(param_),
                                                 "CUDA::BatchNormalization::FP32");
}

template <>
int32_t BatchNormalizationKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return RunBatchNormalization<DataType::FP16>(static_cast<feather::operators::BatchNormParam*>(param_),
                                                 "CUDA::BatchNormalization::FP16");
}

void EnsureCudaBatchNormalizationKernelsRegistered() { (void)g_cuda_batch_normalization_kernels_registered; }

}  // namespace kernel
}  // namespace feather
