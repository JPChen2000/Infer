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
    dispatcher.registerKernel(DeviceType::CUDA, DataType::BF16, "Div", []() {
        return std::make_unique<DivKernel<DeviceType::CUDA, DataType::BF16>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP8E4M3, "Div", []() {
        return std::make_unique<DivKernel<DeviceType::CUDA, DataType::FP8E4M3>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP8E5M2, "Div", []() {
        return std::make_unique<DivKernel<DeviceType::CUDA, DataType::FP8E5M2>>();
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

template <>
int32_t DivKernel<DeviceType::CUDA, DataType::BF16>::compute() {
    return cuda_detail::RunBinary<DataType::BF16, 3>(static_cast<feather::operators::BinaryParam*>(param_),
                                                     "CUDA::Div::BF16");
}

template <DataType dtype>
int32_t ComputeCudaFp8Div(feather::operators::BinaryParam* param) {
    return cuda_detail::RunBinary<dtype, 3>(param, "CUDA::Div::FP8");
}

template <>
int32_t DivKernel<DeviceType::CUDA, DataType::FP8E4M3>::compute() {
    return ComputeCudaFp8Div<DataType::FP8E4M3>(static_cast<feather::operators::BinaryParam*>(param_));
}

template <>
int32_t DivKernel<DeviceType::CUDA, DataType::FP8E5M2>::compute() {
    return ComputeCudaFp8Div<DataType::FP8E5M2>(static_cast<feather::operators::BinaryParam*>(param_));
}

void EnsureCudaDivKernelsRegistered() { (void)g_cuda_div_kernels_registered; }

}  // namespace kernel
}  // namespace feather
