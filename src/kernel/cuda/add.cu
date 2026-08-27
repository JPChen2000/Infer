#include "src/kernel/add.h"

#include <memory>

#include "core/kernel.h"
#include "src/kernel/cuda/elementwise.cuh"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_add_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Add",
                              []() { return std::make_unique<AddKernel<DeviceType::CUDA, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Add",
                              []() { return std::make_unique<AddKernel<DeviceType::CUDA, DataType::FP16>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::BF16, "Add",
                              []() { return std::make_unique<AddKernel<DeviceType::CUDA, DataType::BF16>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP8E4M3, "Add",
                              []() { return std::make_unique<AddKernel<DeviceType::CUDA, DataType::FP8E4M3>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP8E5M2, "Add",
                              []() { return std::make_unique<AddKernel<DeviceType::CUDA, DataType::FP8E5M2>>(); });
    return true;
}();

}  // namespace

template <>
int32_t AddKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return cuda_detail::RunBinary<DataType::FP32, 0>(static_cast<feather::operators::BinaryParam*>(param_),
                                                    "CUDA::Add::FP32");
}

template <>
int32_t AddKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return cuda_detail::RunBinary<DataType::FP16, 0>(static_cast<feather::operators::BinaryParam*>(param_),
                                                    "CUDA::Add::FP16");
}

template <>
int32_t AddKernel<DeviceType::CUDA, DataType::BF16>::compute() {
    return cuda_detail::RunBinary<DataType::BF16, 0>(static_cast<feather::operators::BinaryParam*>(param_),
                                                     "CUDA::Add::BF16");
}

template <DataType dtype>
int32_t ComputeCudaFp8Add(feather::operators::BinaryParam* param) {
    return cuda_detail::RunBinary<dtype, 0>(param, "CUDA::Add::FP8");
}

template <>
int32_t AddKernel<DeviceType::CUDA, DataType::FP8E4M3>::compute() {
    return ComputeCudaFp8Add<DataType::FP8E4M3>(static_cast<feather::operators::BinaryParam*>(param_));
}

template <>
int32_t AddKernel<DeviceType::CUDA, DataType::FP8E5M2>::compute() {
    return ComputeCudaFp8Add<DataType::FP8E5M2>(static_cast<feather::operators::BinaryParam*>(param_));
}

void EnsureCudaAddKernelsRegistered() { (void)g_cuda_add_kernels_registered; }

}  // namespace kernel
}  // namespace feather
