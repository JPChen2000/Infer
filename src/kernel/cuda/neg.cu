#include "src/kernel/neg.h"

#include <memory>

#include "src/kernel/cuda/elementwise.cuh"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_neg_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Neg", []() {
        return std::make_unique<NegKernel<DeviceType::CUDA, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Neg", []() {
        return std::make_unique<NegKernel<DeviceType::CUDA, DataType::FP16>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::BF16, "Neg", []() {
        return std::make_unique<NegKernel<DeviceType::CUDA, DataType::BF16>>();
    });
    return true;
}();

}  // namespace

template <>
int32_t NegKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return cuda_detail::RunUnaryElementwise<DataType::FP32, 10>(
        static_cast<feather::operators::UnaryParam*>(param_), "CUDA::Neg::FP32");
}

template <>
int32_t NegKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return cuda_detail::RunUnaryElementwise<DataType::FP16, 10>(
        static_cast<feather::operators::UnaryParam*>(param_), "CUDA::Neg::FP16");
}

template <>
int32_t NegKernel<DeviceType::CUDA, DataType::BF16>::compute() {
    return cuda_detail::RunUnaryElementwise<DataType::BF16, 10>(
        static_cast<feather::operators::UnaryParam*>(param_), "CUDA::Neg::BF16");
}

void EnsureCudaNegKernelsRegistered() { (void)g_cuda_neg_kernels_registered; }

}  // namespace kernel
}  // namespace feather
