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

void EnsureCudaAddKernelsRegistered() { (void)g_cuda_add_kernels_registered; }

}  // namespace kernel
}  // namespace feather
