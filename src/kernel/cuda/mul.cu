#include "src/kernel/mul.h"

#include <memory>

#include "core/kernel.h"
#include "src/kernel/cuda/elementwise.cuh"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_mul_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Mul",
                              []() { return std::make_unique<MulKernel<DeviceType::CUDA, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Mul",
                              []() { return std::make_unique<MulKernel<DeviceType::CUDA, DataType::FP16>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::BF16, "Mul",
                              []() { return std::make_unique<MulKernel<DeviceType::CUDA, DataType::BF16>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP8E4M3, "Mul",
                              []() { return std::make_unique<MulKernel<DeviceType::CUDA, DataType::FP8E4M3>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP8E5M2, "Mul",
                              []() { return std::make_unique<MulKernel<DeviceType::CUDA, DataType::FP8E5M2>>(); });
    return true;
}();

}  // namespace

template <>
int32_t MulKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return cuda_detail::RunBinary<DataType::FP32, 1>(ParamAs<feather::operators::BinaryParam>(),
                                                    "CUDA::Mul::FP32");
}

template <>
int32_t MulKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return cuda_detail::RunBinary<DataType::FP16, 1>(ParamAs<feather::operators::BinaryParam>(),
                                                    "CUDA::Mul::FP16");
}

template <>
int32_t MulKernel<DeviceType::CUDA, DataType::BF16>::compute() {
    return cuda_detail::RunBinary<DataType::BF16, 1>(ParamAs<feather::operators::BinaryParam>(),
                                                     "CUDA::Mul::BF16");
}

template <DataType dtype>
int32_t ComputeCudaFp8Mul(feather::operators::BinaryParam* param) {
    return cuda_detail::RunBinary<dtype, 1>(param, "CUDA::Mul::FP8");
}

template <>
int32_t MulKernel<DeviceType::CUDA, DataType::FP8E4M3>::compute() {
    return ComputeCudaFp8Mul<DataType::FP8E4M3>(ParamAs<feather::operators::BinaryParam>());
}

template <>
int32_t MulKernel<DeviceType::CUDA, DataType::FP8E5M2>::compute() {
    return ComputeCudaFp8Mul<DataType::FP8E5M2>(ParamAs<feather::operators::BinaryParam>());
}

void EnsureCudaMulKernelsRegistered() { (void)g_cuda_mul_kernels_registered; }

}  // namespace kernel
}  // namespace feather
