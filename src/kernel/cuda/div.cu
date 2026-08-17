#include "src/kernel/div.h"

#include <memory>

#include "src/kernel/cuda/elementwise.cuh"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_div_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Div", []() {
        return std::make_unique<DivKernel<DeviceType::CUDA, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Div", []() {
        return std::make_unique<DivKernel<DeviceType::CUDA, DataType::FP16>>();
    });
    return true;
}();

}  // namespace

template <>
int32_t DivKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return cuda_detail::RunBinary<DataType::FP32, 3>(static_cast<feather::operators::BinaryParam*>(param_),
                                                     "CUDA::Div::FP32");
}

template <>
int32_t DivKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return cuda_detail::RunBinary<DataType::FP16, 3>(static_cast<feather::operators::BinaryParam*>(param_),
                                                     "CUDA::Div::FP16");
}

void EnsureCudaDivKernelsRegistered() { (void)g_cuda_div_kernels_registered; }

}  // namespace kernel
}  // namespace feather
