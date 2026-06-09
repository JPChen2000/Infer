#include "src/kernel/silu.h"

#include <memory>

#include "core/kernel.h"
#include "src/kernel/cuda/elementwise.cuh"
#include "src/kernel/cuda/runtime.h"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_silu_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "SiLU",
                              []() { return std::make_unique<SiluKernel<DeviceType::CUDA, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "SiLU",
                              []() { return std::make_unique<SiluKernel<DeviceType::CUDA, DataType::FP16>>(); });
    return true;
}();

template <typename T>
__global__ void SiluDirectKernelCuda(const T* input, T* output, int64_t numel) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        const float x = cuda_detail::ReadDevice(input, idx);
        const float y = x / (1.0f + __expf(-x));
        cuda_detail::WriteDevice(output, idx, y);
    }
}

template <DataType dtype>
bool RunSiluDirect(feather::operators::UnaryParam* param) {
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return false;
    }
    cuda_detail::DeviceBuffer<T> input;
    cuda_detail::DeviceBuffer<T> output;
    const int64_t numel = param->input->numel();
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &output) != 0) {
        return false;
    }
    SiluDirectKernelCuda<T><<<static_cast<int>(cuda_detail::DivUp(numel, cuda_detail::kCudaThreads)),
                             cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
        input.get(), output.get(), numel);
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) {
        return false;
    }
    SetLastCudaSiluBackend(CudaSiluBackend::kDirect);
    return cuda_detail::CopyDeviceToTensor(&output, param->out.get()) == 0;
}

#ifdef FEATHER_WITH_CUDNN
bool CudnnCheck(cudnnStatus_t status) {
    return status == CUDNN_STATUS_SUCCESS;
}

cudnnDataType_t CudnnTensorDataType(DataType dtype) {
    switch (dtype) {
        case DataType::FP16:
            return CUDNN_DATA_HALF;
        case DataType::FP32:
        default:
            return CUDNN_DATA_FLOAT;
    }
}

template <DataType dtype>
bool RunSiluWithCudnn(feather::operators::UnaryParam* param) {
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return false;
    }
    auto handle = cuda_detail::CudnnHandle();
    if (handle == nullptr) {
        return false;
    }

    cuda_detail::DeviceBuffer<T> input;
    cuda_detail::DeviceBuffer<T> output;
    const int64_t numel = param->input->numel();
    if (numel < 0 || numel > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &output) != 0) {
        return false;
    }

    cudnnTensorDescriptor_t tensor_desc = nullptr;
    cudnnActivationDescriptor_t activation_desc = nullptr;
    if (!CudnnCheck(cudnnCreateTensorDescriptor(&tensor_desc)) ||
        !CudnnCheck(cudnnCreateActivationDescriptor(&activation_desc))) {
        if (tensor_desc != nullptr) {
            cudnnDestroyTensorDescriptor(tensor_desc);
        }
        if (activation_desc != nullptr) {
            cudnnDestroyActivationDescriptor(activation_desc);
        }
        return false;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const bool ok =
        CudnnCheck(cudnnSetTensor4dDescriptor(tensor_desc, CUDNN_TENSOR_NCHW, CudnnTensorDataType(dtype), 1, 1, 1,
                                              static_cast<int>(numel))) &&
        CudnnCheck(cudnnSetActivationDescriptor(activation_desc, CUDNN_ACTIVATION_SWISH, CUDNN_NOT_PROPAGATE_NAN, 1.0)) &&
        CudnnCheck(cudnnActivationForward(handle, activation_desc, &alpha, tensor_desc, input.get(), &beta,
                                          tensor_desc, output.get()));

    cudnnDestroyActivationDescriptor(activation_desc);
    cudnnDestroyTensorDescriptor(tensor_desc);

    if (!ok || cuda_detail::CudaCheck(cudaGetLastError()) != 0) {
        return false;
    }
    SetLastCudaSiluBackend(CudaSiluBackend::kCudnn);
    return cuda_detail::CopyDeviceToTensor(&output, param->out.get()) == 0;
}
#endif

}  // namespace

template <>
int32_t SiluKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    auto* param = static_cast<feather::operators::UnaryParam*>(param_);
    if (RunSiluDirect<DataType::FP32>(param)) {
        return 0;
    }
#ifdef FEATHER_WITH_CUDNN
    if (RunSiluWithCudnn<DataType::FP32>(param)) {
        return 0;
    }
#endif
    SetLastCudaSiluBackend(CudaSiluBackend::kFallback);
    return cuda_detail::RunUnary<DataType::FP32, 3>(param, "CUDA::SiLU::FP32");
}

template <>
int32_t SiluKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    auto* param = static_cast<feather::operators::UnaryParam*>(param_);
    if (RunSiluDirect<DataType::FP16>(param)) {
        return 0;
    }
#ifdef FEATHER_WITH_CUDNN
    if (RunSiluWithCudnn<DataType::FP16>(param)) {
        return 0;
    }
#endif
    SetLastCudaSiluBackend(CudaSiluBackend::kFallback);
    return cuda_detail::RunUnary<DataType::FP16, 3>(param, "CUDA::SiLU::FP16");
}

void EnsureCudaSiluKernelsRegistered() { (void)g_cuda_silu_kernels_registered; }

}  // namespace kernel
}  // namespace feather
