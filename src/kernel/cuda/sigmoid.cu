#include "src/kernel/sigmoid.h"

#include <memory>

#include "core/kernel.h"
#include "src/kernel/cuda/elementwise.cuh"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_sigmoid_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Sigmoid",
                              []() { return std::make_unique<SigmoidKernel<DeviceType::CUDA, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Sigmoid",
                              []() { return std::make_unique<SigmoidKernel<DeviceType::CUDA, DataType::FP16>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::BF16, "Sigmoid",
                              []() { return std::make_unique<SigmoidKernel<DeviceType::CUDA, DataType::BF16>>(); });
    return true;
}();

}  // namespace

template <>
int32_t SigmoidKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return cuda_detail::RunUnary<DataType::FP32, 1>(static_cast<feather::operators::UnaryParam*>(param_),
                                                   "CUDA::Sigmoid::FP32");
}

template <>
int32_t SigmoidKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return cuda_detail::RunUnary<DataType::FP16, 1>(static_cast<feather::operators::UnaryParam*>(param_),
                                                   "CUDA::Sigmoid::FP16");
}

template <>
int32_t SigmoidKernel<DeviceType::CUDA, DataType::BF16>::compute() {
    return cuda_detail::RunUnary<DataType::BF16, 1>(static_cast<feather::operators::UnaryParam*>(param_),
                                                    "CUDA::Sigmoid::BF16");
}

void EnsureCudaSigmoidKernelsRegistered() { (void)g_cuda_sigmoid_kernels_registered; }

}  // namespace kernel
}  // namespace feather
