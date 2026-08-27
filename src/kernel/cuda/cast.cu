#include "src/kernel/cast.h"

#include <memory>

#include "src/kernel/cuda/kernel_io.cuh"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_cast_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Cast", []() {
        return std::make_unique<CastKernel<DeviceType::CUDA, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Cast", []() {
        return std::make_unique<CastKernel<DeviceType::CUDA, DataType::FP16>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::BF16, "Cast", []() {
        return std::make_unique<CastKernel<DeviceType::CUDA, DataType::BF16>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP8E4M3, "Cast", []() {
        return std::make_unique<CastKernel<DeviceType::CUDA, DataType::FP8E4M3>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP8E5M2, "Cast", []() {
        return std::make_unique<CastKernel<DeviceType::CUDA, DataType::FP8E5M2>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::INT32, "Cast", []() {
        return std::make_unique<CastKernel<DeviceType::CUDA, DataType::INT32>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::INT64, "Cast", []() {
        return std::make_unique<CastKernel<DeviceType::CUDA, DataType::INT64>>();
    });
    return true;
}();

template <typename InputT, typename OutputT>
__global__ void CastFloatingKernelCuda(const InputT* input, OutputT* output, int64_t numel, float input_scale,
                                       float output_scale) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= numel) {
        return;
    }
    cuda_detail::WriteDevice(output, index, cuda_detail::ReadDevice(input, index, input_scale), output_scale);
}

template <DataType input_dtype, DataType output_dtype>
int32_t RunFloatingCast(feather::operators::CastParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using InputT = cuda_detail::StorageT<input_dtype>;
    using OutputT = cuda_detail::StorageT<output_dtype>;
    if (param == nullptr || param->input == nullptr || param->out == nullptr ||
        param->input->data_type() != input_dtype || param->to != output_dtype) {
        return -1;
    }
    const std::vector<int64_t> expected_output_dims = param->input->dims().data();
    if (!cuda_detail::IsTensorReady<input_dtype>(param->input.get()) ||
        !cuda_detail::IsOutputReady<output_dtype>(param->out.get(), &expected_output_dims)) {
        return -1;
    }
    cuda_detail::DeviceBuffer<InputT> input;
    cuda_detail::DeviceBuffer<OutputT> output;
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &output) != 0) {
        return -1;
    }
    const int64_t numel = param->input->numel();
    CastFloatingKernelCuda<InputT, OutputT>
        <<<static_cast<int>(cuda_detail::DivUp(numel, cuda_detail::kCudaThreads)), cuda_detail::kCudaThreads, 0,
           cuda_detail::InferenceStream()>>>(input.get(), output.get(), numel, param->input->quantization_scale(),
                                             param->out->quantization_scale());
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) {
        return -1;
    }
    return cuda_detail::CopyDeviceToTensor(&output, param->out.get());
}

}  // namespace

template <>
int32_t CastKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    auto* param = static_cast<feather::operators::CastParam*>(param_);
    if (param == nullptr) {
        return -1;
    }
    if (param->to == DataType::FP32) {
        return RunFloatingCast<DataType::FP32, DataType::FP32>(param, "CUDA::Cast::FP32ToFP32");
    }
    if (param->to == DataType::FP16) {
        return RunFloatingCast<DataType::FP32, DataType::FP16>(param, "CUDA::Cast::FP32ToFP16");
    }
    if (param->to == DataType::BF16) {
        return RunFloatingCast<DataType::FP32, DataType::BF16>(param, "CUDA::Cast::FP32ToBF16");
    }
    if (param->to == DataType::FP8E4M3) {
        return RunFloatingCast<DataType::FP32, DataType::FP8E4M3>(param, "CUDA::Cast::FP32ToFP8E4M3");
    }
    if (param->to == DataType::FP8E5M2) {
        return RunFloatingCast<DataType::FP32, DataType::FP8E5M2>(param, "CUDA::Cast::FP32ToFP8E5M2");
    }
    return -1;
}

template <>
int32_t CastKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    auto* param = static_cast<feather::operators::CastParam*>(param_);
    if (param == nullptr) {
        return -1;
    }
    if (param->to == DataType::FP32) {
        return RunFloatingCast<DataType::FP16, DataType::FP32>(param, "CUDA::Cast::FP16ToFP32");
    }
    if (param->to == DataType::FP16) {
        return RunFloatingCast<DataType::FP16, DataType::FP16>(param, "CUDA::Cast::FP16ToFP16");
    }
    if (param->to == DataType::BF16) {
        return RunFloatingCast<DataType::FP16, DataType::BF16>(param, "CUDA::Cast::FP16ToBF16");
    }
    if (param->to == DataType::FP8E4M3) {
        return RunFloatingCast<DataType::FP16, DataType::FP8E4M3>(param, "CUDA::Cast::FP16ToFP8E4M3");
    }
    if (param->to == DataType::FP8E5M2) {
        return RunFloatingCast<DataType::FP16, DataType::FP8E5M2>(param, "CUDA::Cast::FP16ToFP8E5M2");
    }
    return -1;
}

template <>
int32_t CastKernel<DeviceType::CUDA, DataType::BF16>::compute() {
    auto* param = static_cast<feather::operators::CastParam*>(param_);
    if (param == nullptr) {
        return -1;
    }
    if (param->to == DataType::FP32) {
        return RunFloatingCast<DataType::BF16, DataType::FP32>(param, "CUDA::Cast::BF16ToFP32");
    }
    if (param->to == DataType::FP16) {
        return RunFloatingCast<DataType::BF16, DataType::FP16>(param, "CUDA::Cast::BF16ToFP16");
    }
    if (param->to == DataType::BF16) {
        return RunFloatingCast<DataType::BF16, DataType::BF16>(param, "CUDA::Cast::BF16ToBF16");
    }
    if (param->to == DataType::FP8E4M3) {
        return RunFloatingCast<DataType::BF16, DataType::FP8E4M3>(param, "CUDA::Cast::BF16ToFP8E4M3");
    }
    if (param->to == DataType::FP8E5M2) {
        return RunFloatingCast<DataType::BF16, DataType::FP8E5M2>(param, "CUDA::Cast::BF16ToFP8E5M2");
    }
    return -1;
}

template <>
int32_t CastKernel<DeviceType::CUDA, DataType::INT32>::compute() {
    auto* param = static_cast<feather::operators::CastParam*>(param_);
    if (param == nullptr) {
        return -1;
    }
    if (param->to == DataType::FP32) {
        return RunFloatingCast<DataType::INT32, DataType::FP32>(param, "CUDA::Cast::INT32ToFP32");
    }
    if (param->to == DataType::FP16) {
        return RunFloatingCast<DataType::INT32, DataType::FP16>(param, "CUDA::Cast::INT32ToFP16");
    }
    if (param->to == DataType::BF16) {
        return RunFloatingCast<DataType::INT32, DataType::BF16>(param, "CUDA::Cast::INT32ToBF16");
    }
    if (param->to == DataType::FP8E4M3) {
        return RunFloatingCast<DataType::INT32, DataType::FP8E4M3>(param, "CUDA::Cast::INT32ToFP8E4M3");
    }
    if (param->to == DataType::FP8E5M2) {
        return RunFloatingCast<DataType::INT32, DataType::FP8E5M2>(param, "CUDA::Cast::INT32ToFP8E5M2");
    }
    return -1;
}

template <>
int32_t CastKernel<DeviceType::CUDA, DataType::INT64>::compute() {
    auto* param = static_cast<feather::operators::CastParam*>(param_);
    if (param == nullptr) {
        return -1;
    }
    if (param->to == DataType::FP32) {
        return RunFloatingCast<DataType::INT64, DataType::FP32>(param, "CUDA::Cast::INT64ToFP32");
    }
    if (param->to == DataType::FP16) {
        return RunFloatingCast<DataType::INT64, DataType::FP16>(param, "CUDA::Cast::INT64ToFP16");
    }
    if (param->to == DataType::BF16) {
        return RunFloatingCast<DataType::INT64, DataType::BF16>(param, "CUDA::Cast::INT64ToBF16");
    }
    if (param->to == DataType::FP8E4M3) {
        return RunFloatingCast<DataType::INT64, DataType::FP8E4M3>(param, "CUDA::Cast::INT64ToFP8E4M3");
    }
    if (param->to == DataType::FP8E5M2) {
        return RunFloatingCast<DataType::INT64, DataType::FP8E5M2>(param, "CUDA::Cast::INT64ToFP8E5M2");
    }
    return -1;
}

template <DataType dtype>
int32_t RunCudaFp8InputCast(feather::operators::CastParam* param, const char* timer_name) {
    if (param == nullptr) {
        return -1;
    }
    if (param->to == DataType::FP32) {
        return RunFloatingCast<dtype, DataType::FP32>(param, timer_name);
    }
    if (param->to == DataType::FP16) {
        return RunFloatingCast<dtype, DataType::FP16>(param, timer_name);
    }
    if (param->to == DataType::BF16) {
        return RunFloatingCast<dtype, DataType::BF16>(param, timer_name);
    }
    if (param->to == DataType::FP8E4M3) {
        return RunFloatingCast<dtype, DataType::FP8E4M3>(param, timer_name);
    }
    if (param->to == DataType::FP8E5M2) {
        return RunFloatingCast<dtype, DataType::FP8E5M2>(param, timer_name);
    }
    return -1;
}

template <>
int32_t CastKernel<DeviceType::CUDA, DataType::FP8E4M3>::compute() {
    return RunCudaFp8InputCast<DataType::FP8E4M3>(static_cast<feather::operators::CastParam*>(param_),
                                                   "CUDA::Cast::FP8E4M3");
}

template <>
int32_t CastKernel<DeviceType::CUDA, DataType::FP8E5M2>::compute() {
    return RunCudaFp8InputCast<DataType::FP8E5M2>(static_cast<feather::operators::CastParam*>(param_),
                                                  "CUDA::Cast::FP8E5M2");
}

void EnsureCudaCastKernelsRegistered() { (void)g_cuda_cast_kernels_registered; }

}  // namespace kernel
}  // namespace feather
