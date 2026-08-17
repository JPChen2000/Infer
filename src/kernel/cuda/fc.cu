#include "src/kernel/fc.h"

#include <memory>

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

void EnsureCudaFcKernelsRegistered() { (void)g_cuda_fc_kernels_registered; }

}  // namespace kernel
}  // namespace feather
