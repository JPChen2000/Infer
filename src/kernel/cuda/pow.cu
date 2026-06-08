#include "src/kernel/pow.h"

#include <memory>

#include "core/kernel.h"
#include "src/kernel/cuda/elementwise.cuh"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_pow_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Pow",
                              []() { return std::make_unique<PowKernel<DeviceType::CUDA, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Pow",
                              []() { return std::make_unique<PowKernel<DeviceType::CUDA, DataType::FP16>>(); });
    return true;
}();

}  // namespace

template <>
int32_t PowKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return cuda_detail::RunPow<DataType::FP32>(static_cast<feather::operators::PowParam*>(param_),
                                              "CUDA::Pow::FP32");
}

template <>
int32_t PowKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return cuda_detail::RunPow<DataType::FP16>(static_cast<feather::operators::PowParam*>(param_),
                                              "CUDA::Pow::FP16");
}

void EnsureCudaPowKernelsRegistered() { (void)g_cuda_pow_kernels_registered; }

}  // namespace kernel
}  // namespace feather
