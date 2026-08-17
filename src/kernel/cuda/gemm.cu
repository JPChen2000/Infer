#include "src/kernel/gemm.h"

#include <memory>

#include "core/kernel.h"
#include "src/kernel/cuda/linear_kernels.cuh"
#include "src/operator/params.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_gemm_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Gemm",
                              []() { return std::make_unique<GemmKernel<DeviceType::CUDA, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Gemm",
                              []() { return std::make_unique<GemmKernel<DeviceType::CUDA, DataType::FP16>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::BF16, "Gemm",
                              []() { return std::make_unique<GemmKernel<DeviceType::CUDA, DataType::BF16>>(); });
    return true;
}();

template <DataType dtype>
int RunGemm(feather::operators::GemmParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->a == nullptr || param->b == nullptr || param->out == nullptr) {
        return -1;
    }
    if (param->a->dims().size() != 2 || param->b->dims().size() != 2 || param->trans_a) {
        return -1;
    }
    const int64_t m = param->a->dims()[0];
    const int64_t k = param->a->dims()[1];
    const int64_t b_k = param->trans_b ? param->b->dims()[1] : param->b->dims()[0];
    const int64_t n = param->trans_b ? param->b->dims()[0] : param->b->dims()[1];
    if (k != b_k) {
        return -1;
    }
    cuda_detail::DeviceBuffer<T> a;
    cuda_detail::DeviceBuffer<T> b;
    cuda_detail::DeviceBuffer<T> bias;
    cuda_detail::DeviceBuffer<T> out;
    T* bias_ptr = nullptr;
    int bias_mode = 0;
    if (cuda_detail::CopyTensorToDevice(param->a.get(), &a) != 0 ||
        cuda_detail::CopyTensorToDevice(param->b.get(), &b) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &out) != 0) {
        return -1;
    }
    if (param->bias != nullptr && param->bias->IsInitialized()) {
        if (cuda_detail::CopyTensorToDevice(param->bias.get(), &bias) != 0) {
            return -1;
        }
        bias_ptr = bias.get();
        bias_mode = param->bias->dims().size() == 1 ? 1 : 2;
    }
    if (cuda_detail::LaunchCublasMatMul<dtype>(a.get(), b.get(), out.get(), m, k, n, param->alpha, param->trans_b) != 0) {
        return -1;
    }
    if (bias_ptr != nullptr) {
        const int64_t total = m * n;
        cuda_detail::AddScaledBiasKernelCuda<T>
            <<<static_cast<int>(cuda_detail::DivUp(total, cuda_detail::kCudaThreads)), cuda_detail::kCudaThreads, 0,
               cuda_detail::InferenceStream()>>>(out.get(), bias_ptr, m, n, bias_mode, param->beta);
    }
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) {
        return -1;
    }
    return cuda_detail::CopyDeviceToTensor(&out, param->out.get());
}

}  // namespace

template <>
int32_t GemmKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return RunGemm<DataType::FP32>(static_cast<feather::operators::GemmParam*>(param_), "CUDA::Gemm::FP32");
}

template <>
int32_t GemmKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return RunGemm<DataType::FP16>(static_cast<feather::operators::GemmParam*>(param_), "CUDA::Gemm::FP16");
}

template <>
int32_t GemmKernel<DeviceType::CUDA, DataType::BF16>::compute() {
    return RunGemm<DataType::BF16>(static_cast<feather::operators::GemmParam*>(param_), "CUDA::Gemm::BF16");
}

void EnsureCudaGemmKernelsRegistered() { (void)g_cuda_gemm_kernels_registered; }

}  // namespace kernel
}  // namespace feather
