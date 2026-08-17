#include "src/kernel/sub.h"

#include <memory>

#include "src/kernel/cuda/elementwise.cuh"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_sub_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Sub", []() {
        return std::make_unique<SubKernel<DeviceType::CUDA, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Sub", []() {
        return std::make_unique<SubKernel<DeviceType::CUDA, DataType::FP16>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::BF16, "Sub", []() {
        return std::make_unique<SubKernel<DeviceType::CUDA, DataType::BF16>>();
    });
    return true;
}();

}  // namespace

template <>
int32_t SubKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return cuda_detail::RunBinary<DataType::FP32, 2>(static_cast<feather::operators::BinaryParam*>(param_),
                                                     "CUDA::Sub::FP32");
}

template <>
int32_t SubKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return cuda_detail::RunBinary<DataType::FP16, 2>(static_cast<feather::operators::BinaryParam*>(param_),
                                                     "CUDA::Sub::FP16");
}

template <>
int32_t SubKernel<DeviceType::CUDA, DataType::BF16>::compute() {
    return cuda_detail::RunBinary<DataType::BF16, 2>(static_cast<feather::operators::BinaryParam*>(param_),
                                                     "CUDA::Sub::BF16");
}

void EnsureCudaSubKernelsRegistered() { (void)g_cuda_sub_kernels_registered; }

}  // namespace kernel
}  // namespace feather
