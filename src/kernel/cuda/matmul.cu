#include "src/kernel/matmul.h"

#include <memory>

#include "core/kernel.h"
#include "src/kernel/cuda/linear_kernels.cuh"
#include "src/operator/params.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_matmul_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "MatMul",
                              []() { return std::make_unique<MatMulKernel<DeviceType::CUDA, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "MatMul",
                              []() { return std::make_unique<MatMulKernel<DeviceType::CUDA, DataType::FP16>>(); });
    return true;
}();

template <DataType dtype>
int RunMatMul(feather::operators::MatMulParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->a == nullptr || param->b == nullptr || param->out == nullptr) {
        return -1;
    }
    const int64_t m = param->a->dims()[0];
    const int64_t k = param->a->dims()[1];
    const int64_t n = param->b->dims()[1];
    cuda_detail::DeviceBuffer<T> a;
    cuda_detail::DeviceBuffer<T> b;
    cuda_detail::DeviceBuffer<T> out;
    if (cuda_detail::CopyTensorToDevice(param->a.get(), &a) != 0 ||
        cuda_detail::CopyTensorToDevice(param->b.get(), &b) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &out) != 0) {
        return -1;
    }
    if (cuda_detail::LaunchCublasMatMul<dtype>(a.get(), b.get(), out.get(), m, k, n) != 0 ||
        cuda_detail::CudaCheck(cudaGetLastError()) != 0) {
        return -1;
    }
    return cuda_detail::CopyDeviceToTensor(&out, param->out.get());
}

}  // namespace

template <>
int32_t MatMulKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return RunMatMul<DataType::FP32>(static_cast<feather::operators::MatMulParam*>(param_), "CUDA::MatMul::FP32");
}

template <>
int32_t MatMulKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return RunMatMul<DataType::FP16>(static_cast<feather::operators::MatMulParam*>(param_), "CUDA::MatMul::FP16");
}

void EnsureCudaMatMulKernelsRegistered() { (void)g_cuda_matmul_kernels_registered; }

}  // namespace kernel
}  // namespace feather
