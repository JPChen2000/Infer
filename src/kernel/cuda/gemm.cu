#include "src/kernel/gemm.h"

#include <cmath>
#include <memory>

#include "core/kernel.h"
#include "src/kernel/cuda/linear_kernels.cuh"
#include "src/operator/params.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool IsVectorBias(const Tensor* bias, int64_t n) {
    if (bias == nullptr || !bias->IsInitialized() || bias->dims().empty() || n <= 0 ||
        bias->dims()[bias->dims().size() - 1] != n || bias->numel() != n) {
        return false;
    }
    for (size_t index = 0; index + 1 < bias->dims().size(); ++index) {
        if (bias->dims()[index] != 1) {
            return false;
        }
    }
    return true;
}

bool g_cuda_gemm_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Gemm",
                              []() { return std::make_unique<GemmKernel<DeviceType::CUDA, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Gemm",
                              []() { return std::make_unique<GemmKernel<DeviceType::CUDA, DataType::FP16>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::BF16, "Gemm",
                              []() { return std::make_unique<GemmKernel<DeviceType::CUDA, DataType::BF16>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP8E4M3, "Gemm",
                              []() { return std::make_unique<GemmKernel<DeviceType::CUDA, DataType::FP8E4M3>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP8E5M2, "Gemm",
                              []() { return std::make_unique<GemmKernel<DeviceType::CUDA, DataType::FP8E5M2>>(); });
    return true;
}();

template <DataType dtype>
int RunGemm(feather::operators::GemmParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->a == nullptr || param->b == nullptr || param->out == nullptr) {
        return -1;
    }
    if (param->a->dims().size() < 2 || param->b->dims().size() != 2 || param->trans_a ||
        param->a->data_type() != dtype || param->b->data_type() != dtype || param->out->data_type() != dtype) {
        return -1;
    }
    const int64_t k = param->a->dims()[param->a->dims().size() - 1];
    if (k <= 0 || param->a->numel() <= 0 || param->a->numel() % k != 0) {
        return -1;
    }
    const int64_t m = param->a->numel() / k;
    const int64_t b_k = param->trans_b ? param->b->dims()[1] : param->b->dims()[0];
    const int64_t n = param->trans_b ? param->b->dims()[0] : param->b->dims()[1];
    if (k != b_k || n <= 0 || param->out->numel() != m * n) {
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
        std::fprintf(stderr, "CUDA Gemm device buffer setup failed\n");
        return -1;
    }
    if (param->bias != nullptr && param->bias->IsInitialized()) {
        if (param->bias->data_type() != dtype) {
            return -1;
        }
        if (cuda_detail::CopyTensorToDevice(param->bias.get(), &bias) != 0) {
            return -1;
        }
        bias_ptr = bias.get();
        if (IsVectorBias(param->bias.get(), n)) {
            bias_mode = 1;
        } else {
            if (param->a->dims().size() != 2 || param->bias->dims().size() != 2 ||
                param->bias->dims()[0] != m || param->bias->dims()[1] != n) {
                return -1;
            }
            bias_mode = 2;
        }
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

template <DataType dtype>
int RunFp8Gemm(feather::operators::GemmParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->a == nullptr || param->b == nullptr || param->out == nullptr ||
        !cuda_detail::IsTensorReady<dtype>(param->a.get()) || !cuda_detail::IsTensorReady<dtype>(param->b.get()) ||
        param->trans_a || param->a->dims().size() < 2 || param->b->dims().size() != 2 ||
        !std::isfinite(param->alpha) || !std::isfinite(param->beta)) {
        return -1;
    }
    const int64_t k = param->a->dims()[param->a->dims().size() - 1];
    if (k <= 0 || param->a->numel() <= 0 || param->a->numel() % k != 0) {
        return -1;
    }
    const int64_t m = param->a->numel() / k;
    const int64_t b_k = param->trans_b ? param->b->dims()[1] : param->b->dims()[0];
    const int64_t n = param->trans_b ? param->b->dims()[0] : param->b->dims()[1];
    if (m <= 0 || n <= 0 || k != b_k || m > std::numeric_limits<int64_t>::max() / n ||
        param->out->numel() != m * n) {
        return -1;
    }
    auto expected_output_dims = param->a->dims().data();
    expected_output_dims.back() = n;
    if (!cuda_detail::IsOutputReady<dtype>(param->out.get(), &expected_output_dims)) return -1;
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
    if (param->bias != nullptr) {
        if (!cuda_detail::IsTensorReady<dtype>(param->bias.get()) ||
            cuda_detail::CopyTensorToDevice(param->bias.get(), &bias) != 0) {
            return -1;
        }
        bias_ptr = bias.get();
        if (IsVectorBias(param->bias.get(), n)) {
            bias_mode = 1;
        } else if (param->a->dims().size() == 2 && param->bias->dims().size() == 2 &&
                   param->bias->dims()[0] == m && param->bias->dims()[1] == n) {
            bias_mode = 2;
        } else {
            return -1;
        }
    }
    cuda_detail::LaunchFp8MatMulKernelCuda<T>(
        a.get(), b.get(), bias_ptr, out.get(), m, k, n, bias_mode, param->a->quantization_scale(),
        param->b->quantization_scale(), param->bias != nullptr ? param->bias->quantization_scale() : 1.0f,
        param->out->quantization_scale(), param->alpha, param->beta, param->trans_b);
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

template <>
int32_t GemmKernel<DeviceType::CUDA, DataType::FP8E4M3>::compute() {
    return RunFp8Gemm<DataType::FP8E4M3>(static_cast<feather::operators::GemmParam*>(param_),
                                         "CUDA::Gemm::FP8E4M3");
}

template <>
int32_t GemmKernel<DeviceType::CUDA, DataType::FP8E5M2>::compute() {
    return RunFp8Gemm<DataType::FP8E5M2>(static_cast<feather::operators::GemmParam*>(param_),
                                         "CUDA::Gemm::FP8E5M2");
}

void EnsureCudaGemmKernelsRegistered() { (void)g_cuda_gemm_kernels_registered; }

}  // namespace kernel
}  // namespace feather
