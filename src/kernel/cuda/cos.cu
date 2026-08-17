#include "src/kernel/cos.h"

#include <memory>

#include "src/kernel/cuda/elementwise.cuh"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_cos_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Cos", []() {
        return std::make_unique<CosKernel<DeviceType::CUDA, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Cos", []() {
        return std::make_unique<CosKernel<DeviceType::CUDA, DataType::FP16>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::BF16, "Cos", []() {
        return std::make_unique<CosKernel<DeviceType::CUDA, DataType::BF16>>();
    });
    return true;
}();

}  // namespace

template <>
int32_t CosKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return cuda_detail::RunUnaryElementwise<DataType::FP32, 9>(
        static_cast<feather::operators::UnaryParam*>(param_), "CUDA::Cos::FP32");
}

template <>
int32_t CosKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return cuda_detail::RunUnaryElementwise<DataType::FP16, 9>(
        static_cast<feather::operators::UnaryParam*>(param_), "CUDA::Cos::FP16");
}

template <>
int32_t CosKernel<DeviceType::CUDA, DataType::BF16>::compute() {
    return cuda_detail::RunUnaryElementwise<DataType::BF16, 9>(
        static_cast<feather::operators::UnaryParam*>(param_), "CUDA::Cos::BF16");
}

void EnsureCudaCosKernelsRegistered() { (void)g_cuda_cos_kernels_registered; }

}  // namespace kernel
}  // namespace feather
