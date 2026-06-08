#include "src/kernel/identity.h"

#include <memory>

#include "core/kernel.h"
#include "src/kernel/cuda/kernel_io.cuh"
#include "src/operator/params.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_identity_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Identity",
                              []() { return std::make_unique<IdentityKernel<DeviceType::CUDA, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Identity",
                              []() { return std::make_unique<IdentityKernel<DeviceType::CUDA, DataType::FP16>>(); });
    return true;
}();

template <DataType dtype>
int RunIdentity(feather::operators::UnaryParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }
    return cuda_detail::RunDeviceCopy<T>(param->input.get(), param->out.get());
}

}  // namespace

template <>
int32_t IdentityKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return RunIdentity<DataType::FP32>(static_cast<feather::operators::UnaryParam*>(param_),
                                      "CUDA::Identity::FP32");
}

template <>
int32_t IdentityKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return RunIdentity<DataType::FP16>(static_cast<feather::operators::UnaryParam*>(param_),
                                      "CUDA::Identity::FP16");
}

void EnsureCudaIdentityKernelsRegistered() { (void)g_cuda_identity_kernels_registered; }

}  // namespace kernel
}  // namespace feather
