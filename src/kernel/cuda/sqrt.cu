#include "src/kernel/sqrt.h"

#include <memory>

#include "src/kernel/cuda/elementwise.cuh"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_sqrt_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Sqrt", []() {
        return std::make_unique<SqrtKernel<DeviceType::CUDA, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Sqrt", []() {
        return std::make_unique<SqrtKernel<DeviceType::CUDA, DataType::FP16>>();
    });
    return true;
}();

}  // namespace

template <>
int32_t SqrtKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return cuda_detail::RunUnaryElementwise<DataType::FP32, 4>(
        static_cast<feather::operators::UnaryParam*>(param_), "CUDA::Sqrt::FP32");
}

template <>
int32_t SqrtKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return cuda_detail::RunUnaryElementwise<DataType::FP16, 4>(
        static_cast<feather::operators::UnaryParam*>(param_), "CUDA::Sqrt::FP16");
}

void EnsureCudaSqrtKernelsRegistered() { (void)g_cuda_sqrt_kernels_registered; }

}  // namespace kernel
}  // namespace feather
