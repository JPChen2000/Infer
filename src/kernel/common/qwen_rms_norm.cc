#include "src/kernel/qwen_rms_norm.h"

#include <cmath>
#include <memory>

#include "src/kernel/common/kernel_io.h"
#include "src/kernel/common/tensor_op_utils.h"
#include "util/timer.h"

namespace feather {
namespace kernel {
namespace {

bool ValidateQwenRmsNorm(const operators::QwenRmsNormParam* param, int64_t* rows, int64_t* hidden,
                         float* epsilon) {
    if (param == nullptr || param->input == nullptr || param->weight == nullptr || param->epsilon == nullptr ||
        param->out == nullptr || !param->input->IsInitialized() || !param->weight->IsInitialized() ||
        !param->epsilon->IsInitialized() || !param->out->IsInitialized() || rows == nullptr || hidden == nullptr ||
        epsilon == nullptr) {
        return false;
    }
    if ((param->input->data_type() != DataType::FP32 && param->input->data_type() != DataType::BF16) ||
        (param->weight->data_type() != DataType::FP32 && param->weight->data_type() != DataType::BF16) ||
        (param->out->data_type() != DataType::FP32 && param->out->data_type() != DataType::BF16) ||
        param->input->dims().empty() || param->input->dims() != param->out->dims() ||
        param->epsilon->numel() != 1) {
        return false;
    }
    *hidden = param->input->dims()[param->input->dims().size() - 1];
    if (*hidden <= 0 || param->input->numel() <= 0 || param->input->numel() % *hidden != 0 ||
        param->out->numel() != param->input->numel() || param->weight->numel() != *hidden) {
        return false;
    }
    *rows = param->input->numel() / *hidden;
    if (!ReadScalarFloatTensor(param->epsilon.get(), epsilon) || !std::isfinite(*epsilon) || *epsilon < 0.0f) {
        return false;
    }
    const size_t input_bytes = static_cast<size_t>(param->input->numel()) * DataTypeBytes(param->input->data_type());
    const size_t weight_bytes = static_cast<size_t>(param->weight->numel()) * DataTypeBytes(param->weight->data_type());
    const size_t output_bytes = static_cast<size_t>(param->out->numel()) * DataTypeBytes(param->out->data_type());
    return DataTypeBytes(param->input->data_type()) != 0 && DataTypeBytes(param->weight->data_type()) != 0 &&
           DataTypeBytes(param->out->data_type()) != 0 && param->input->memory_size() >= input_bytes &&
           param->weight->memory_size() >= weight_bytes && param->out->memory_size() >= output_bytes;
}

float ReadValue(const Tensor* tensor, int64_t index) { return common_tensor_detail::ReadFloat(tensor, index); }

void WriteValue(Tensor* tensor, int64_t index, float value) {
    if (tensor->data_type() == DataType::BF16) {
        static_cast<BFloat16*>(tensor->raw_data())[index].bits = FloatToBFloat16(value);
    } else {
        static_cast<float*>(tensor->raw_data())[index] = value;
    }
}

template <DataType input_dtype>
int32_t ComputeQwenRmsNorm(operators::QwenRmsNormParam* param) {
    int64_t rows = 0;
    int64_t hidden = 0;
    float epsilon = 0.0f;
    if (!ValidateQwenRmsNorm(param, &rows, &hidden, &epsilon) || param->input->data_type() != input_dtype) {
        return -1;
    }
    for (int64_t row = 0; row < rows; ++row) {
        float sum_square = 0.0f;
        const int64_t row_offset = row * hidden;
        for (int64_t column = 0; column < hidden; ++column) {
            const float value = ReadValue(param->input.get(), row_offset + column);
            sum_square += value * value;
        }
        const float inverse_rms = 1.0f / std::sqrt(sum_square / static_cast<float>(hidden) + epsilon);
        for (int64_t column = 0; column < hidden; ++column) {
            const float value = ReadValue(param->input.get(), row_offset + column);
            const float scale = ReadValue(param->weight.get(), column) + param->weight_offset;
            WriteValue(param->out.get(), row_offset + column, value * inverse_rms * scale);
        }
    }
    return 0;
}

bool g_common_qwen_rms_norm_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP32, "QwenRmsNorm", []() {
        return std::make_unique<QwenRmsNormKernel<DeviceType::COMMON, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::BF16, "QwenRmsNorm", []() {
        return std::make_unique<QwenRmsNormKernel<DeviceType::COMMON, DataType::BF16>>();
    });
    return true;
}();

}  // namespace

template <>
int32_t QwenRmsNormKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::QwenRmsNorm::FP32");
    return ComputeQwenRmsNorm<DataType::FP32>(static_cast<operators::QwenRmsNormParam*>(param_));
}

template <>
int32_t QwenRmsNormKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    AutoTimer timer("Common::QwenRmsNorm::BF16");
    return ComputeQwenRmsNorm<DataType::BF16>(static_cast<operators::QwenRmsNormParam*>(param_));
}

void EnsureCommonQwenRmsNormKernelsRegistered() { (void)g_common_qwen_rms_norm_kernels_registered; }

void EnsureQwenRmsNormKernelsRegistered() {
    EnsureCommonQwenRmsNormKernelsRegistered();
    EnsureX86QwenRmsNormKernelsRegistered();
#ifdef FEATHER_WITH_CUDA
    EnsureCudaQwenRmsNormKernelsRegistered();
#endif
}

}  // namespace kernel
}  // namespace feather
