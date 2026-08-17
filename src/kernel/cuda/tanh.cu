#include "src/kernel/tanh.h"

#include <memory>

#include "src/kernel/cuda/elementwise.cuh"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_tanh_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Tanh", []() {
        return std::make_unique<TanhKernel<DeviceType::CUDA, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Tanh", []() {
        return std::make_unique<TanhKernel<DeviceType::CUDA, DataType::FP16>>();
    });
    return true;
}();

}  // namespace

template <>
int32_t TanhKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return cuda_detail::RunUnaryElementwise<DataType::FP32, 5>(
        static_cast<feather::operators::UnaryParam*>(param_), "CUDA::Tanh::FP32");
}

template <>
int32_t TanhKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return cuda_detail::RunUnaryElementwise<DataType::FP16, 5>(
        static_cast<feather::operators::UnaryParam*>(param_), "CUDA::Tanh::FP16");
}

void EnsureCudaTanhKernelsRegistered() { (void)g_cuda_tanh_kernels_registered; }

}  // namespace kernel
}  // namespace feather
