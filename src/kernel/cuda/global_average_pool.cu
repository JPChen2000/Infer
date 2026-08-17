#include "src/kernel/global_average_pool.h"

#include <memory>

#include "core/kernel.h"
#include "src/kernel/cuda/kernel_io.cuh"
#include "src/operator/params.h"
#include "util/timer.h"
#include "util/types.h"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_global_average_pool_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(
        DeviceType::CUDA, DataType::FP32, "GlobalAveragePool",
        []() { return std::make_unique<GlobalAveragePoolKernel<DeviceType::CUDA, DataType::FP32>>(); });
    dispatcher.registerKernel(
        DeviceType::CUDA, DataType::FP16, "GlobalAveragePool",
        []() { return std::make_unique<GlobalAveragePoolKernel<DeviceType::CUDA, DataType::FP16>>(); });
    return true;
}();

struct CudaGlobalAveragePoolShape {
    int64_t n{0};
    int64_t c{0};
    int64_t h{0};
    int64_t w{0};
    int channel_last{0};
};

template <typename T>
__global__ void GlobalAveragePoolKernelCuda(const T* input, T* out, int64_t output_numel,
                                            CudaGlobalAveragePoolShape shape) {
    const int64_t output_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output_index >= output_numel) {
        return;
    }
    const int64_t batch = output_index / shape.c;
    const int64_t channel = output_index % shape.c;
    float sum = 0.0f;
    for (int64_t height = 0; height < shape.h; ++height) {
        for (int64_t width = 0; width < shape.w; ++width) {
            const int64_t input_index = shape.channel_last != 0
                                            ? ((batch * shape.h + height) * shape.w + width) * shape.c + channel
                                            : ((batch * shape.c + channel) * shape.h + height) * shape.w + width;
            sum += cuda_detail::ReadDevice(input, input_index);
        }
    }
    cuda_detail::WriteDevice(out, output_index, sum / static_cast<float>(shape.h * shape.w));
}

template <DataType dtype>
int RunGlobalAveragePool(feather::operators::GlobalAveragePoolParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->input == nullptr || param->out == nullptr || param->input->dims().size() != 4 ||
        param->input->data_type() != dtype) {
        return -1;
    }

    ImageShape4D host_shape{};
    const DataLayout layout = NormalizeDataLayout(param->input->layout());
    if (!DecodeImageShape4D(param->input->dims().data(), layout, &host_shape)) {
        return -1;
    }

    cuda_detail::DeviceBuffer<T> input;
    cuda_detail::DeviceBuffer<T> out;
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &out) != 0) {
        return -1;
    }

    param->out->set_layout(param->input->layout());
    const CudaGlobalAveragePoolShape shape{host_shape.n, host_shape.c, host_shape.h, host_shape.w,
                                           IsChannelLastLayout(layout) ? 1 : 0};
    const int64_t output_numel = param->out->numel();
    if (output_numel != host_shape.n * host_shape.c) {
        return -1;
    }
    GlobalAveragePoolKernelCuda<T><<<static_cast<int>(cuda_detail::DivUp(output_numel, cuda_detail::kCudaThreads)),
                                     cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
        input.get(), out.get(), output_numel, shape);
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) {
        return -1;
    }
    return cuda_detail::CopyDeviceToTensor(&out, param->out.get());
}

}  // namespace

template <>
int32_t GlobalAveragePoolKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return RunGlobalAveragePool<DataType::FP32>(static_cast<feather::operators::GlobalAveragePoolParam*>(param_),
                                                "CUDA::GlobalAveragePool::FP32");
}

template <>
int32_t GlobalAveragePoolKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return RunGlobalAveragePool<DataType::FP16>(static_cast<feather::operators::GlobalAveragePoolParam*>(param_),
                                                "CUDA::GlobalAveragePool::FP16");
}

void EnsureCudaGlobalAveragePoolKernelsRegistered() { (void)g_cuda_global_average_pool_kernels_registered; }

}  // namespace kernel
}  // namespace feather
