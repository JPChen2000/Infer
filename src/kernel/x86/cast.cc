#include "src/kernel/cast.h"

#include <immintrin.h>

#include <cstring>
#include <memory>

#include "src/kernel/common/tensor_op_utils.h"
#include "util/bf16.h"
#include "util/fp16.h"
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
        _mm256_add_epi32(_mm256_set1_epi32(0x7fff), _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1)));
    __m256i high_bits = _mm256_srli_epi32(_mm256_add_epi32(bits, round_bias), 16);
    high_bits = _mm256_or_si256(high_bits, _mm256_and_si256(nan_mask, _mm256_set1_epi32(0x0040)));
    const __m128i packed = _mm_packus_epi32(_mm256_castsi256_si128(high_bits),
                                             _mm256_extracti128_si256(high_bits, 1));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output), packed);
}

bool HasValidCastBuffers(const feather::operators::CastParam* param) {
    return param != nullptr && param->input != nullptr && param->out != nullptr && param->input->IsInitialized() &&
           param->out->IsInitialized() && param->input->numel() == param->out->numel() &&
           DataTypeBytes(param->to) != 0 &&
           param->out->memory_size() >= static_cast<size_t>(param->input->numel()) * DataTypeBytes(param->to);
}

int32_t ComputeCastScalar(feather::operators::CastParam* param) {
    if (!HasValidCastBuffers(param)) {
        return -1;
    }

    for (int64_t i = 0; i < param->input->numel(); ++i) {
        const float value = common_tensor_detail::ReadFloat(param->input.get(), i);
        switch (param->to) {
            case DataType::BOOL:
                static_cast<uint8_t*>(param->out->raw_data())[i] =
                    common_tensor_detail::ReadBool(param->input.get(), i) ? 1 : 0;
                break;
            case DataType::UINT8:
                static_cast<uint8_t*>(param->out->raw_data())[i] = static_cast<uint8_t>(value);
                break;
            case DataType::INT8:
                static_cast<int8_t*>(param->out->raw_data())[i] = static_cast<int8_t>(value);
                break;
            case DataType::FP16:
                static_cast<uint16_t*>(param->out->raw_data())[i] = FloatToHalf(value);
                break;
            case DataType::BF16:
                static_cast<uint16_t*>(param->out->raw_data())[i] = FloatToBFloat16(value);
                break;
            case DataType::INT32:
                static_cast<int32_t*>(param->out->raw_data())[i] = static_cast<int32_t>(value);
                break;
            case DataType::INT64:
                static_cast<int64_t*>(param->out->raw_data())[i] = static_cast<int64_t>(value);
                break;
            case DataType::FP32:
                static_cast<float*>(param->out->raw_data())[i] = value;
                break;
            default:
                return -1;
        }
    }
    param->out->set_data_type(param->to);
    return 0;
}

int32_t ComputeBf16ToFp32(feather::operators::CastParam* param) {
    if (!HasValidCastBuffers(param) || param->input->data_type() != DataType::BF16 || param->to != DataType::FP32) {
        return -1;
    }
    const uint16_t* input = static_cast<const uint16_t*>(param->input->raw_data());
    float* output = static_cast<float*>(param->out->raw_data());
    int64_t index = 0;
    const int64_t count = param->input->numel();
    for (; index + 8 <= count; index += 8) {
        _mm256_storeu_ps(output + index, LoadBf16x8AsFloat(input + index));
    }
    for (; index < count; ++index) {
        output[index] = BFloat16ToFloat(input[index]);
    }
    param->out->set_data_type(DataType::FP32);
    return 0;
}

int32_t ComputeFp32ToBf16(feather::operators::CastParam* param) {
    if (!HasValidCastBuffers(param) || param->input->data_type() != DataType::FP32 || param->to != DataType::BF16) {
        return -1;
    }
    const float* input = static_cast<const float*>(param->input->raw_data());
    uint16_t* output = static_cast<uint16_t*>(param->out->raw_data());
    int64_t index = 0;
    const int64_t count = param->input->numel();
    for (; index + 8 <= count; index += 8) {
        StoreBf16x8(_mm256_loadu_ps(input + index), output + index);
    }
    for (; index < count; ++index) {
        output[index] = FloatToBFloat16(input[index]);
    }
    param->out->set_data_type(DataType::BF16);
    return 0;
}

}  // namespace

template <>
int32_t CastKernel<DeviceType::X86, DataType::BF16>::compute() {
    AutoTimer timer("X86::Cast::BF16");
    auto* param = static_cast<feather::operators::CastParam*>(param_);
    if (param != nullptr && param->input != nullptr && param->input->data_type() == DataType::BF16 &&
        param->to == DataType::FP32) {
        return ComputeBf16ToFp32(param);
    }
    return ComputeCastScalar(param);
}

template <>
int32_t CastKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::Cast::FP32");
    auto* param = static_cast<feather::operators::CastParam*>(param_);
    if (param != nullptr && param->input != nullptr && param->input->data_type() == DataType::FP32 &&
        param->to == DataType::BF16) {
        return ComputeFp32ToBf16(param);
    }
    return ComputeCastScalar(param);
}

void EnsureX86CastKernelsRegistered() {
    static bool registered = []() {
        auto& dispatcher = KernelDispatcher::instance();
        dispatcher.registerKernel(DeviceType::X86, DataType::BF16, "Cast", []() {
            return std::make_unique<CastKernel<DeviceType::X86, DataType::BF16>>();
        });
        dispatcher.registerKernel(DeviceType::X86, DataType::FP32, "Cast", []() {
            return std::make_unique<CastKernel<DeviceType::X86, DataType::FP32>>();
        });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
