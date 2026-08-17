#include "src/kernel/softplus.h"

#include <memory>

#include "src/kernel/cuda/elementwise.cuh"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_softplus_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Softplus", []() {
        return std::make_unique<SoftplusKernel<DeviceType::CUDA, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Softplus", []() {
        return std::make_unique<SoftplusKernel<DeviceType::CUDA, DataType::FP16>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::BF16, "Softplus", []() {
        return std::make_unique<SoftplusKernel<DeviceType::CUDA, DataType::BF16>>();
    });
    return true;
}();

}  // namespace

template <>
int32_t SoftplusKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return cuda_detail::RunUnaryElementwise<DataType::FP32, 11>(
        static_cast<feather::operators::UnaryParam*>(param_), "CUDA::Softplus::FP32");
}

template <>
int32_t SoftplusKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return cuda_detail::RunUnaryElementwise<DataType::FP16, 11>(
        static_cast<feather::operators::UnaryParam*>(param_), "CUDA::Softplus::FP16");
}

template <>
int32_t SoftplusKernel<DeviceType::CUDA, DataType::BF16>::compute() {
    return cuda_detail::RunUnaryElementwise<DataType::BF16, 11>(
        static_cast<feather::operators::UnaryParam*>(param_), "CUDA::Softplus::BF16");
}

void EnsureCudaSoftplusKernelsRegistered() { (void)g_cuda_softplus_kernels_registered; }

}  // namespace kernel
}  // namespace feather
