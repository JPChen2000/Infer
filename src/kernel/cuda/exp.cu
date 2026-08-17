#include "src/kernel/exp.h"

#include <memory>

#include "src/kernel/cuda/elementwise.cuh"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_exp_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Exp", []() {
        return std::make_unique<ExpKernel<DeviceType::CUDA, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Exp", []() {
        return std::make_unique<ExpKernel<DeviceType::CUDA, DataType::FP16>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::BF16, "Exp", []() {
        return std::make_unique<ExpKernel<DeviceType::CUDA, DataType::BF16>>();
    });
    return true;
}();

}  // namespace

template <>
int32_t ExpKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return cuda_detail::RunUnaryElementwise<DataType::FP32, 7>(
        static_cast<feather::operators::UnaryParam*>(param_), "CUDA::Exp::FP32");
}

template <>
int32_t ExpKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return cuda_detail::RunUnaryElementwise<DataType::FP16, 7>(
        static_cast<feather::operators::UnaryParam*>(param_), "CUDA::Exp::FP16");
}

template <>
int32_t ExpKernel<DeviceType::CUDA, DataType::BF16>::compute() {
    return cuda_detail::RunUnaryElementwise<DataType::BF16, 7>(
        static_cast<feather::operators::UnaryParam*>(param_), "CUDA::Exp::BF16");
}

void EnsureCudaExpKernelsRegistered() { (void)g_cuda_exp_kernels_registered; }

}  // namespace kernel
}  // namespace feather
