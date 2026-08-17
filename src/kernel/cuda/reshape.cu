#include "src/kernel/reshape.h"

#include <memory>

#include "core/kernel.h"
#include "src/kernel/cuda/kernel_io.cuh"
#include "src/operator/params.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_reshape_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Reshape",
                              []() { return std::make_unique<ReshapeKernel<DeviceType::CUDA, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Reshape",
                              []() { return std::make_unique<ReshapeKernel<DeviceType::CUDA, DataType::FP16>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::BF16, "Reshape",
                              []() { return std::make_unique<ReshapeKernel<DeviceType::CUDA, DataType::BF16>>(); });
    return true;
}();

template <DataType dtype>
int RunReshape(feather::operators::ReshapeParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }
    return cuda_detail::RunDeviceCopy<T>(param->input.get(), param->out.get());
}

}  // namespace

template <>
int32_t ReshapeKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return RunReshape<DataType::FP32>(static_cast<feather::operators::ReshapeParam*>(param_),
                                     "CUDA::Reshape::FP32");
}

template <>
int32_t ReshapeKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return RunReshape<DataType::FP16>(static_cast<feather::operators::ReshapeParam*>(param_),
                                     "CUDA::Reshape::FP16");
}

template <>
int32_t ReshapeKernel<DeviceType::CUDA, DataType::BF16>::compute() {
    return RunReshape<DataType::BF16>(static_cast<feather::operators::ReshapeParam*>(param_),
                                      "CUDA::Reshape::BF16");
}

void EnsureCudaReshapeKernelsRegistered() { (void)g_cuda_reshape_kernels_registered; }

}  // namespace kernel
}  // namespace feather
