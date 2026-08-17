#include "src/kernel/erf.h"

#include <memory>

#include "src/kernel/cuda/elementwise.cuh"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_erf_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Erf", []() {
        return std::make_unique<ErfKernel<DeviceType::CUDA, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Erf", []() {
        return std::make_unique<ErfKernel<DeviceType::CUDA, DataType::FP16>>();
    });
    return true;
}();

}  // namespace

template <>
int32_t ErfKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return cuda_detail::RunUnaryElementwise<DataType::FP32, 6>(
        static_cast<feather::operators::UnaryParam*>(param_), "CUDA::Erf::FP32");
}

template <>
int32_t ErfKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return cuda_detail::RunUnaryElementwise<DataType::FP16, 6>(
        static_cast<feather::operators::UnaryParam*>(param_), "CUDA::Erf::FP16");
}

void EnsureCudaErfKernelsRegistered() { (void)g_cuda_erf_kernels_registered; }

}  // namespace kernel
}  // namespace feather
