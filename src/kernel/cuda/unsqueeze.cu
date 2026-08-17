#include "src/kernel/unsqueeze.h"

#include <memory>

#include "src/kernel/cuda/kernel_io.cuh"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_unsqueeze_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Unsqueeze", []() {
        return std::make_unique<UnsqueezeKernel<DeviceType::CUDA, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Unsqueeze", []() {
        return std::make_unique<UnsqueezeKernel<DeviceType::CUDA, DataType::FP16>>();
    });
    return true;
}();

template <DataType dtype>
int32_t RunUnsqueeze(feather::operators::AxesParam* param, const char* timer_name) {
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
int32_t UnsqueezeKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return RunUnsqueeze<DataType::FP32>(static_cast<feather::operators::AxesParam*>(param_),
                                        "CUDA::Unsqueeze::FP32");
}

template <>
int32_t UnsqueezeKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return RunUnsqueeze<DataType::FP16>(static_cast<feather::operators::AxesParam*>(param_),
                                        "CUDA::Unsqueeze::FP16");
}

void EnsureCudaUnsqueezeKernelsRegistered() { (void)g_cuda_unsqueeze_kernels_registered; }

}  // namespace kernel
}  // namespace feather
