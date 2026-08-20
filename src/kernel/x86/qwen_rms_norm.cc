#include "src/kernel/qwen_rms_norm.h"

#include <immintrin.h>

#include <cmath>
#include <memory>

#include "src/kernel/common/kernel_io.h"
#include "util/bf16.h"
#include "util/timer.h"

namespace feather {
namespace kernel {
namespace {

inline __m256 LoadBf16x8AsFloat(const uint16_t* input) {
    const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input));
    const __m256i expanded = _mm256_cvtepu16_epi32(packed);
    return _mm256_castsi256_ps(_mm256_slli_epi32(expanded, 16));
}

inline void StoreBf16x8(__m256 value, uint16_t* output) {
    const __m256i bits = _mm256_castps_si256(value);
    const __m256i exponent = _mm256_and_si256(bits, _mm256_set1_epi32(0x7f800000));
    const __m256i mantissa = _mm256_and_si256(bits, _mm256_set1_epi32(0x007fffff));
    const __m256i exponent_is_inf_or_nan = _mm256_cmpeq_epi32(exponent, _mm256_set1_epi32(0x7f800000));
    const __m256i mantissa_is_zero = _mm256_cmpeq_epi32(mantissa, _mm256_setzero_si256());
    const __m256i nan_mask = _mm256_andnot_si256(mantissa_is_zero, exponent_is_inf_or_nan);
    const __m256i round_bias =
        _mm256_add_epi32(_mm256_set1_epi32(0x7fff),
                         _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1)));
    __m256i high_bits = _mm256_srli_epi32(_mm256_add_epi32(bits, round_bias), 16);
    high_bits = _mm256_or_si256(high_bits, _mm256_and_si256(nan_mask, _mm256_set1_epi32(0x0040)));
    const __m128i packed = _mm_packus_epi32(_mm256_castsi256_si128(high_bits),
                                             _mm256_extracti128_si256(high_bits, 1));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output), packed);
}

bool Validate(const operators::QwenRmsNormParam* param, int64_t* rows, int64_t* hidden, float* epsilon) {
    if (param == nullptr || param->input == nullptr || param->weight == nullptr || param->epsilon == nullptr ||
        param->out == nullptr || !param->input->IsInitialized() || !param->weight->IsInitialized() ||
        !param->epsilon->IsInitialized() || !param->out->IsInitialized() || rows == nullptr || hidden == nullptr ||
        epsilon == nullptr || param->input->dims().empty() || param->input->dims() != param->out->dims() ||
        param->epsilon->numel() != 1 || (param->input->data_type() != DataType::FP32 &&
                                         param->input->data_type() != DataType::BF16) ||
        (param->weight->data_type() != DataType::FP32 && param->weight->data_type() != DataType::BF16) ||
        (param->out->data_type() != DataType::FP32 && param->out->data_type() != DataType::BF16)) {
        return false;
    }
    *hidden = param->input->dims()[param->input->dims().size() - 1];
    if (*hidden <= 0 || param->input->numel() <= 0 || param->input->numel() % *hidden != 0 ||
        param->out->numel() != param->input->numel() || param->weight->numel() != *hidden ||
        !ReadScalarFloatTensor(param->epsilon.get(), epsilon) || !std::isfinite(*epsilon) || *epsilon < 0.0f) {
        return false;
    }
    *rows = param->input->numel() / *hidden;
    const size_t input_bytes = static_cast<size_t>(param->input->numel()) * DataTypeBytes(param->input->data_type());
    const size_t weight_bytes = static_cast<size_t>(param->weight->numel()) * DataTypeBytes(param->weight->data_type());
    const size_t output_bytes = static_cast<size_t>(param->out->numel()) * DataTypeBytes(param->out->data_type());
    return param->input->memory_size() >= input_bytes && param->weight->memory_size() >= weight_bytes &&
           param->out->memory_size() >= output_bytes;
}

template <DataType input_dtype>
inline __m256 LoadInput(const Tensor* input, int64_t index) {
    if constexpr (input_dtype == DataType::BF16) {
        return LoadBf16x8AsFloat(static_cast<const uint16_t*>(input->raw_data()) + index);
    }
    return _mm256_loadu_ps(static_cast<const float*>(input->raw_data()) + index);
}

template <DataType input_dtype>
inline float ReadInput(const Tensor* input, int64_t index) {
    if constexpr (input_dtype == DataType::BF16) {
        return BFloat16ToFloat(static_cast<const uint16_t*>(input->raw_data())[index]);
    }
    return static_cast<const float*>(input->raw_data())[index];
}

inline float ReadWeight(const Tensor* weight, int64_t index) {
    if (weight->data_type() == DataType::BF16) {
        return BFloat16ToFloat(static_cast<const uint16_t*>(weight->raw_data())[index]);
    }
    return static_cast<const float*>(weight->raw_data())[index];
}

template <DataType input_dtype>
int32_t Compute(operators::QwenRmsNormParam* param) {
    int64_t rows = 0;
    int64_t hidden = 0;
    float epsilon = 0.0f;
    if (!Validate(param, &rows, &hidden, &epsilon) || param->input->data_type() != input_dtype) {
        return -1;
    }
    const bool output_bf16 = param->out->data_type() == DataType::BF16;
    const bool weight_bf16 = param->weight->data_type() == DataType::BF16;
    const auto* weight_fp32_data = weight_bf16 ? nullptr : static_cast<const float*>(param->weight->raw_data());
    const auto* weight_bf16_data = weight_bf16 ? static_cast<const uint16_t*>(param->weight->raw_data()) : nullptr;
    const __m256 offset = _mm256_set1_ps(param->weight_offset);

    for (int64_t row = 0; row < rows; ++row) {
        const int64_t row_offset = row * hidden;
        __m256 sum = _mm256_setzero_ps();
        int64_t column = 0;
        for (; column + 8 <= hidden; column += 8) {
            const __m256 value = LoadInput<input_dtype>(param->input.get(), row_offset + column);
            sum = _mm256_fmadd_ps(value, value, sum);
        }
        alignas(32) float partial[8];
        _mm256_store_ps(partial, sum);
        float sum_value = partial[0] + partial[1] + partial[2] + partial[3] + partial[4] + partial[5] + partial[6] +
                          partial[7];
        for (; column < hidden; ++column) {
            const float value = ReadInput<input_dtype>(param->input.get(), row_offset + column);
            sum_value += value * value;
        }
        const float inverse_rms_scalar = 1.0f / std::sqrt(sum_value / static_cast<float>(hidden) + epsilon);
        const __m256 inverse_rms = _mm256_set1_ps(inverse_rms_scalar);
        column = 0;
        for (; column + 8 <= hidden; column += 8) {
            __m256 scale = weight_bf16 ? LoadBf16x8AsFloat(weight_bf16_data + column)
                                       : _mm256_loadu_ps(weight_fp32_data + column);
            scale = _mm256_add_ps(scale, offset);
            const __m256 value = LoadInput<input_dtype>(param->input.get(), row_offset + column);
            const __m256 result = _mm256_mul_ps(_mm256_mul_ps(value, inverse_rms), scale);
            if (output_bf16) {
                StoreBf16x8(result, static_cast<uint16_t*>(param->out->raw_data()) + row_offset + column);
            } else {
                _mm256_storeu_ps(static_cast<float*>(param->out->raw_data()) + row_offset + column, result);
            }
        }
        for (; column < hidden; ++column) {
            const float result = ReadInput<input_dtype>(param->input.get(), row_offset + column) * inverse_rms_scalar *
                                  (ReadWeight(param->weight.get(), column) + param->weight_offset);
            if (output_bf16) {
                static_cast<uint16_t*>(param->out->raw_data())[row_offset + column] = FloatToBFloat16(result);
            } else {
                static_cast<float*>(param->out->raw_data())[row_offset + column] = result;
            }
        }
    }
    return 0;
}

}  // namespace

template <>
int32_t QwenRmsNormKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::QwenRmsNorm::FP32");
    return Compute<DataType::FP32>(static_cast<operators::QwenRmsNormParam*>(param_));
}

template <>
int32_t QwenRmsNormKernel<DeviceType::X86, DataType::BF16>::compute() {
    AutoTimer timer("X86::QwenRmsNorm::BF16");
    return Compute<DataType::BF16>(static_cast<operators::QwenRmsNormParam*>(param_));
}

void EnsureX86QwenRmsNormKernelsRegistered() {
    static bool registered = []() {
        auto& dispatcher = KernelDispatcher::instance();
        dispatcher.registerKernel(DeviceType::X86, DataType::FP32, "QwenRmsNorm", []() {
            return std::make_unique<QwenRmsNormKernel<DeviceType::X86, DataType::FP32>>();
        });
        dispatcher.registerKernel(DeviceType::X86, DataType::BF16, "QwenRmsNorm", []() {
            return std::make_unique<QwenRmsNormKernel<DeviceType::X86, DataType::BF16>>();
        });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
