#include "src/kernel/squeeze.h"

#include <memory>

#include "src/kernel/cuda/kernel_io.cuh"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_squeeze_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Squeeze", []() {
        return std::make_unique<SqueezeKernel<DeviceType::CUDA, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Squeeze", []() {
        return std::make_unique<SqueezeKernel<DeviceType::CUDA, DataType::FP16>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::BF16, "Squeeze", []() {
        return std::make_unique<SqueezeKernel<DeviceType::CUDA, DataType::BF16>>();
    });
    return true;
}();

template <DataType dtype>
int32_t RunSqueeze(feather::operators::AxesParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->input == nullptr || param->out == nullptr || param->input->data_type() != dtype ||
        param->input->numel() != param->out->numel()) {
        return -1;
    }
    return cuda_detail::RunDeviceCopy<T>(param->input.get(), param->out.get());
}

}  // namespace

template <>
int32_t SqueezeKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return RunSqueeze<DataType::FP32>(static_cast<feather::operators::AxesParam*>(param_),
                                      "CUDA::Squeeze::FP32");
}

template <>
int32_t SqueezeKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return RunSqueeze<DataType::FP16>(static_cast<feather::operators::AxesParam*>(param_),
                                      "CUDA::Squeeze::FP16");
}

template <>
int32_t SqueezeKernel<DeviceType::CUDA, DataType::BF16>::compute() {
    return RunSqueeze<DataType::BF16>(static_cast<feather::operators::AxesParam*>(param_),
                                      "CUDA::Squeeze::BF16");
}

void EnsureCudaSqueezeKernelsRegistered() { (void)g_cuda_squeeze_kernels_registered; }

}  // namespace kernel
}  // namespace feather
