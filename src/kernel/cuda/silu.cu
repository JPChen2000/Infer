#include "src/kernel/silu.h"

#include <memory>

#include "core/kernel.h"
#include "src/kernel/cuda/elementwise.cuh"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_silu_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "SiLU",
                              []() { return std::make_unique<SiluKernel<DeviceType::CUDA, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "SiLU",
                              []() { return std::make_unique<SiluKernel<DeviceType::CUDA, DataType::FP16>>(); });
    return true;
}();

}  // namespace

template <>
int32_t SiluKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return cuda_detail::RunUnary<DataType::FP32, 3>(static_cast<feather::operators::UnaryParam*>(param_),
                                                   "CUDA::SiLU::FP32");
}

template <>
int32_t SiluKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return cuda_detail::RunUnary<DataType::FP16, 3>(static_cast<feather::operators::UnaryParam*>(param_),
                                                   "CUDA::SiLU::FP16");
}

void EnsureCudaSiluKernelsRegistered() { (void)g_cuda_silu_kernels_registered; }

}  // namespace kernel
}  // namespace feather
