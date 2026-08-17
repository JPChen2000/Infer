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
    return true;
}();

}  // namespace

template <>
int32_t MulKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return cuda_detail::RunBinary<DataType::FP32, 1>(static_cast<feather::operators::BinaryParam*>(param_),
                                                    "CUDA::Mul::FP32");
}

template <>
int32_t MulKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return cuda_detail::RunBinary<DataType::FP16, 1>(static_cast<feather::operators::BinaryParam*>(param_),
                                                    "CUDA::Mul::FP16");
}

template <>
int32_t MulKernel<DeviceType::CUDA, DataType::BF16>::compute() {
    return cuda_detail::RunBinary<DataType::BF16, 1>(static_cast<feather::operators::BinaryParam*>(param_),
                                                     "CUDA::Mul::BF16");
}

void EnsureCudaMulKernelsRegistered() { (void)g_cuda_mul_kernels_registered; }

}  // namespace kernel
}  // namespace feather
