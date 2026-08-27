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
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP8E4M3, "Sub", []() {
        return std::make_unique<SubKernel<DeviceType::CUDA, DataType::FP8E4M3>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP8E5M2, "Sub", []() {
        return std::make_unique<SubKernel<DeviceType::CUDA, DataType::FP8E5M2>>();
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

template <DataType dtype>
int32_t ComputeCudaFp8Sub(feather::operators::BinaryParam* param) {
    return cuda_detail::RunBinary<dtype, 2>(param, "CUDA::Sub::FP8");
}

template <>
int32_t SubKernel<DeviceType::CUDA, DataType::FP8E4M3>::compute() {
    return ComputeCudaFp8Sub<DataType::FP8E4M3>(static_cast<feather::operators::BinaryParam*>(param_));
}

template <>
int32_t SubKernel<DeviceType::CUDA, DataType::FP8E5M2>::compute() {
    return ComputeCudaFp8Sub<DataType::FP8E5M2>(static_cast<feather::operators::BinaryParam*>(param_));
}

void EnsureCudaSubKernelsRegistered() { (void)g_cuda_sub_kernels_registered; }

}  // namespace kernel
}  // namespace feather
