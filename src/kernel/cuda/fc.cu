#include "src/kernel/fc.h"

#include <limits>
#include <memory>
#include <vector>

#include "core/kernel.h"
#include "src/kernel/cuda/linear_kernels.cuh"
#include "src/operator/params.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_fc_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "FC",
                              []() { return std::make_unique<FcKernel<DeviceType::CUDA, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "FC",
                              []() { return std::make_unique<FcKernel<DeviceType::CUDA, DataType::FP16>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::BF16, "FC",
                              []() { return std::make_unique<FcKernel<DeviceType::CUDA, DataType::BF16>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP8E4M3, "FC",
                              []() { return std::make_unique<FcKernel<DeviceType::CUDA, DataType::FP8E4M3>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP8E5M2, "FC",
                              []() { return std::make_unique<FcKernel<DeviceType::CUDA, DataType::FP8E5M2>>(); });
    return true;
}();

template <DataType dtype>
int RunFc(feather::operators::FcParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->input == nullptr || param->w == nullptr || param->out == nullptr) {
        return -1;
    }
    const int64_t m = param->input->dims()[0];
    const int64_t in_features = param->input->dims()[1];
    const int64_t out_features = param->w->dims()[1];
    cuda_detail::DeviceBuffer<T> input;
    cuda_detail::DeviceBuffer<T> weight;
    cuda_detail::DeviceBuffer<T> bias;
    cuda_detail::DeviceBuffer<T> out;
    T* bias_ptr = nullptr;
    int bias_mode = 0;
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0 ||
        cuda_detail::CopyTensorToDevice(param->w.get(), &weight) != 0 ||
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
    if (cuda_detail::LaunchCublasMatMul<dtype>(input.get(), weight.get(), out.get(), m, in_features, out_features, 1.0f,
                                               false) != 0) {
        return -1;
    }
    if (bias_ptr != nullptr) {
        const int64_t total = m * out_features;
        cuda_detail::AddBiasKernelCuda<T>
            <<<static_cast<int>(cuda_detail::DivUp(total, cuda_detail::kCudaThreads)), cuda_detail::kCudaThreads, 0,
               cuda_detail::InferenceStream()>>>(out.get(), bias_ptr, m, out_features, bias_mode);
    }
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) {
        return -1;
    }
    return cuda_detail::CopyDeviceToTensor(&out, param->out.get());
}

template <DataType dtype>
int RunFp8Fc(feather::operators::FcParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->input == nullptr || param->w == nullptr || param->out == nullptr ||
        !cuda_detail::IsTensorReady<dtype>(param->input.get()) || !cuda_detail::IsTensorReady<dtype>(param->w.get()) ||
        param->input->dims().size() != 2 || param->w->dims().size() != 2) {
        return -1;
    }
    const int64_t m = param->input->dims()[0];
    const int64_t k = param->input->dims()[1];
    const int64_t n = param->w->dims()[1];
    if (m <= 0 || k <= 0 || n <= 0 || param->w->dims()[0] != k || m > std::numeric_limits<int64_t>::max() / n ||
        param->out->numel() != m * n) {
        return -1;
    }
    const std::vector<int64_t> expected_output_dims{m, n};
    if (!cuda_detail::IsOutputReady<dtype>(param->out.get(), &expected_output_dims)) return -1;
    cuda_detail::DeviceBuffer<T> input;
    cuda_detail::DeviceBuffer<T> weight;
    cuda_detail::DeviceBuffer<T> bias;
    cuda_detail::DeviceBuffer<T> out;
    T* bias_ptr = nullptr;
    int bias_mode = 0;
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0 ||
        cuda_detail::CopyTensorToDevice(param->w.get(), &weight) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &out) != 0) {
        return -1;
    }
    if (param->bias != nullptr && param->bias->IsInitialized()) {
        if (!cuda_detail::IsTensorReady<dtype>(param->bias.get()) || cuda_detail::CopyTensorToDevice(param->bias.get(), &bias) != 0) {
            return -1;
        }
        if (param->bias->dims().size() == 1 && param->bias->dims()[0] == n) {
            bias_mode = 1;
        } else if (param->bias->dims().size() == 2 && param->bias->dims()[0] == m &&
                   param->bias->dims()[1] == n) {
            bias_mode = 2;
        } else {
            return -1;
        }
        bias_ptr = bias.get();
    }
    cuda_detail::LaunchFp8MatMulKernelCuda<T>(
        input.get(), weight.get(), bias_ptr, out.get(), m, k, n, bias_mode, param->input->quantization_scale(),
        param->w->quantization_scale(), param->bias != nullptr ? param->bias->quantization_scale() : 1.0f,
        param->out->quantization_scale());
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) {
        return -1;
    }
    return cuda_detail::CopyDeviceToTensor(&out, param->out.get());
}

}  // namespace

template <>
int32_t FcKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return RunFc<DataType::FP32>(static_cast<feather::operators::FcParam*>(param_), "CUDA::FC::FP32");
}

template <>
int32_t FcKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return RunFc<DataType::FP16>(static_cast<feather::operators::FcParam*>(param_), "CUDA::FC::FP16");
}

template <>
int32_t FcKernel<DeviceType::CUDA, DataType::BF16>::compute() {
    return RunFc<DataType::BF16>(static_cast<feather::operators::FcParam*>(param_), "CUDA::FC::BF16");
}

template <>
int32_t FcKernel<DeviceType::CUDA, DataType::FP8E4M3>::compute() {
    return RunFp8Fc<DataType::FP8E4M3>(static_cast<feather::operators::FcParam*>(param_), "CUDA::FC::FP8E4M3");
}

template <>
int32_t FcKernel<DeviceType::CUDA, DataType::FP8E5M2>::compute() {
    return RunFp8Fc<DataType::FP8E5M2>(static_cast<feather::operators::FcParam*>(param_), "CUDA::FC::FP8E5M2");
}

void EnsureCudaFcKernelsRegistered() { (void)g_cuda_fc_kernels_registered; }

}  // namespace kernel
}  // namespace feather
