#include "src/kernel/sin.h"

#include <memory>

#include "src/kernel/cuda/elementwise.cuh"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_sin_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Sin", []() {
        return std::make_unique<SinKernel<DeviceType::CUDA, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Sin", []() {
        return std::make_unique<SinKernel<DeviceType::CUDA, DataType::FP16>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::BF16, "Sin", []() {
        return std::make_unique<SinKernel<DeviceType::CUDA, DataType::BF16>>();
    });
    return true;
}();

}  // namespace

template <>
int32_t SinKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return cuda_detail::RunUnaryElementwise<DataType::FP32, 8>(
        static_cast<feather::operators::UnaryParam*>(param_), "CUDA::Sin::FP32");
}

template <>
int32_t SinKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return cuda_detail::RunUnaryElementwise<DataType::FP16, 8>(
        static_cast<feather::operators::UnaryParam*>(param_), "CUDA::Sin::FP16");
}

template <>
int32_t SinKernel<DeviceType::CUDA, DataType::BF16>::compute() {
    return cuda_detail::RunUnaryElementwise<DataType::BF16, 8>(
        static_cast<feather::operators::UnaryParam*>(param_), "CUDA::Sin::BF16");
}

void EnsureCudaSinKernelsRegistered() { (void)g_cuda_sin_kernels_registered; }

}  // namespace kernel
}  // namespace feather
