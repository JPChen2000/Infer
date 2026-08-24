#include "src/kernel/qwen_rms_norm.h"

#include <cmath>
#include <memory>

#include "src/kernel/common/tensor_op_utils.h"
#include "src/kernel/cuda/kernel_io.cuh"
#include "util/bf16.h"
#include "util/timer.h"

namespace feather {
namespace kernel {
namespace {

bool ReadEpsilon(const Tensor* tensor, float* epsilon) {
    if (tensor == nullptr || epsilon == nullptr || !tensor->IsInitialized() || tensor->numel() != 1) {
        return false;
    }
    switch (tensor->data_type()) {
        case DataType::FP32:
            *epsilon = tensor->data<float>()[0];
            return true;
        case DataType::BF16:
            *epsilon = BFloat16ToFloat(tensor->data<BFloat16>()[0].bits);
            return true;
        default:
            return false;
    }
}

bool Validate(const operators::QwenRmsNormParam* param, int64_t* rows, int64_t* hidden, float* epsilon) {
    if (param == nullptr || rows == nullptr || hidden == nullptr || epsilon == nullptr || param->input == nullptr ||
        param->weight == nullptr || param->epsilon == nullptr || param->out == nullptr ||
        !param->input->IsInitialized() || !param->weight->IsInitialized() || !param->epsilon->IsInitialized() ||
        !param->out->IsInitialized()) {
        return false;
    }
    const auto input_type = param->input->data_type();
    const auto weight_type = param->weight->data_type();
    const auto output_type = param->out->data_type();
    if ((input_type != DataType::FP32 && input_type != DataType::BF16) ||
        (weight_type != DataType::FP32 && weight_type != DataType::BF16) ||
        (output_type != DataType::FP32 && output_type != DataType::BF16) || param->input->dims().empty() ||
        param->input->dims() != param->out->dims() || param->epsilon->numel() != 1) {
        return false;
    }
    *hidden = param->input->dims()[param->input->dims().size() - 1];
    if (*hidden <= 0 || param->input->numel() <= 0 || param->input->numel() % *hidden != 0 ||
        param->weight->numel() != *hidden || param->out->numel() != param->input->numel() ||
        !ReadEpsilon(param->epsilon.get(), epsilon) || !std::isfinite(*epsilon) || *epsilon < 0.0f) {
        return false;
    }
    *rows = param->input->numel() / *hidden;
    return DataTypeBytes(input_type) != 0 && DataTypeBytes(weight_type) != 0 && DataTypeBytes(output_type) != 0 &&
           param->input->memory_size() >= static_cast<size_t>(param->input->numel()) * DataTypeBytes(input_type) &&
           param->weight->memory_size() >= static_cast<size_t>(param->weight->numel()) * DataTypeBytes(weight_type) &&
           param->out->memory_size() >= static_cast<size_t>(param->out->numel()) * DataTypeBytes(output_type);
}

template <typename InputT, typename WeightT, typename OutputT>
__global__ void QwenRmsNormKernelCuda(const InputT* input, const WeightT* weight, OutputT* output, int64_t rows,
                                      int64_t hidden, float epsilon, float weight_offset) {
    __shared__ float partial[cuda_detail::kCudaThreads];
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows) {
        return;
    }
    const int lane = static_cast<int>(threadIdx.x);
    const int64_t row_offset = row * hidden;
    float sum_square = 0.0f;
    for (int64_t column = lane; column < hidden; column += blockDim.x) {
        const float value = cuda_detail::ReadDevice(input, row_offset + column);
        sum_square += value * value;
    }
    partial[lane] = sum_square;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        __syncthreads();
    }
    const float inverse_rms = rsqrtf(partial[0] / static_cast<float>(hidden) + epsilon);
    for (int64_t column = lane; column < hidden; column += blockDim.x) {
        const float value = cuda_detail::ReadDevice(input, row_offset + column);
        const float scale = cuda_detail::ReadDevice(weight, column) + weight_offset;
        cuda_detail::WriteDevice(output, row_offset + column, value * inverse_rms * scale);
    }
}

template <DataType input_dtype, DataType weight_dtype, DataType output_dtype>
int Run(operators::QwenRmsNormParam* param, int64_t rows, int64_t hidden, float epsilon) {
    using InputT = cuda_detail::StorageT<input_dtype>;
    using WeightT = cuda_detail::StorageT<weight_dtype>;
    using OutputT = cuda_detail::StorageT<output_dtype>;
    cuda_detail::DeviceBuffer<InputT> input;
    cuda_detail::DeviceBuffer<WeightT> weight;
    cuda_detail::DeviceBuffer<OutputT> output;
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0 ||
        cuda_detail::CopyTensorToDevice(param->weight.get(), &weight) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &output) != 0) {
        return -1;
    }
    QwenRmsNormKernelCuda<InputT, WeightT, OutputT>
        <<<static_cast<int>(rows), cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
            input.get(), weight.get(), output.get(), rows, hidden, epsilon, param->weight_offset);
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) {
        return -1;
    }
    return cuda_detail::CopyDeviceToTensor(&output, param->out.get());
}

template <DataType input_dtype, DataType output_dtype>
int RunWeight(operators::QwenRmsNormParam* param, int64_t rows, int64_t hidden, float epsilon) {
    if (param->weight->data_type() == DataType::FP32) {
        return Run<input_dtype, DataType::FP32, output_dtype>(param, rows, hidden, epsilon);
    }
    return Run<input_dtype, DataType::BF16, output_dtype>(param, rows, hidden, epsilon);
}

int RunQwenRmsNorm(operators::QwenRmsNormParam* param) {
    int64_t rows = 0;
    int64_t hidden = 0;
    float epsilon = 0.0f;
    if (!Validate(param, &rows, &hidden, &epsilon)) {
        return -1;
    }
    const auto output_type = param->out->data_type();
    if (param->input->data_type() == DataType::FP32) {
        return output_type == DataType::FP32 ? RunWeight<DataType::FP32, DataType::FP32>(param, rows, hidden, epsilon)
                                             : RunWeight<DataType::FP32, DataType::BF16>(param, rows, hidden, epsilon);
    }
    return output_type == DataType::FP32 ? RunWeight<DataType::BF16, DataType::FP32>(param, rows, hidden, epsilon)
                                         : RunWeight<DataType::BF16, DataType::BF16>(param, rows, hidden, epsilon);
}

bool g_cuda_qwen_rms_norm_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "QwenRmsNorm", []() {
        return std::make_unique<QwenRmsNormKernel<DeviceType::CUDA, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::BF16, "QwenRmsNorm", []() {
        return std::make_unique<QwenRmsNormKernel<DeviceType::CUDA, DataType::BF16>>();
    });
    return true;
}();

}  // namespace

template <>
int32_t QwenRmsNormKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    AutoTimer timer("CUDA::QwenRmsNorm::FP32");
    return RunQwenRmsNorm(static_cast<operators::QwenRmsNormParam*>(param_));
}

template <>
int32_t QwenRmsNormKernel<DeviceType::CUDA, DataType::BF16>::compute() {
    AutoTimer timer("CUDA::QwenRmsNorm::BF16");
    return RunQwenRmsNorm(static_cast<operators::QwenRmsNormParam*>(param_));
}

void EnsureCudaQwenRmsNormKernelsRegistered() { (void)g_cuda_qwen_rms_norm_kernels_registered; }

}  // namespace kernel
}  // namespace feather
