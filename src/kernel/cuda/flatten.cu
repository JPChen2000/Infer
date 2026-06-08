#include "src/kernel/flatten.h"

#include <memory>

#include "core/kernel.h"
#include "src/kernel/cuda/kernel_io.cuh"
#include "src/operator/params.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_flatten_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Flatten",
                              []() { return std::make_unique<FlattenKernel<DeviceType::CUDA, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Flatten",
                              []() { return std::make_unique<FlattenKernel<DeviceType::CUDA, DataType::FP16>>(); });
    return true;
}();

template <DataType dtype>
int RunFlatten(feather::operators::FlattenParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }
    return cuda_detail::RunDeviceCopy<T>(param->input.get(), param->out.get());
}

}  // namespace

template <>
int32_t FlattenKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return RunFlatten<DataType::FP32>(static_cast<feather::operators::FlattenParam*>(param_),
                                     "CUDA::Flatten::FP32");
}

template <>
int32_t FlattenKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return RunFlatten<DataType::FP16>(static_cast<feather::operators::FlattenParam*>(param_),
                                     "CUDA::Flatten::FP16");
}

void EnsureCudaFlattenKernelsRegistered() { (void)g_cuda_flatten_kernels_registered; }

}  // namespace kernel
}  // namespace feather
