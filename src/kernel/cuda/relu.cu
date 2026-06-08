#include "src/kernel/relu.h"

#include <memory>

#include "core/kernel.h"
#include "src/kernel/cuda/elementwise.cuh"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_relu_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "ReLU",
                              []() { return std::make_unique<ReluKernel<DeviceType::CUDA, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "ReLU",
                              []() { return std::make_unique<ReluKernel<DeviceType::CUDA, DataType::FP16>>(); });
    return true;
}();

}  // namespace

template <>
int32_t ReluKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return cuda_detail::RunUnary<DataType::FP32, 0>(static_cast<feather::operators::UnaryParam*>(param_),
                                                   "CUDA::ReLU::FP32");
}

template <>
int32_t ReluKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return cuda_detail::RunUnary<DataType::FP16, 0>(static_cast<feather::operators::UnaryParam*>(param_),
                                                   "CUDA::ReLU::FP16");
}

void EnsureCudaReluKernelsRegistered() { (void)g_cuda_relu_kernels_registered; }

}  // namespace kernel
}  // namespace feather
