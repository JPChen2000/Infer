#include "src/kernel/cast.h"

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>

#include "src/kernel/common/tensor_op_utils.h"
#include "src/kernel/x86/fp8_utils.h"
#include "util/bf16.h"
#include "util/fp8.h"
#include "util/fp16.h"
#include "util/timer.h"
#include "util/threading.h"

#if defined(FEATHER_WITH_OPENMP)
#include <omp.h>
#endif

namespace feather {
namespace kernel {

namespace {

bool IsFp8DataType(DataType data_type) {
    return data_type == DataType::FP8E4M3 || data_type == DataType::FP8E5M2;
}

bool HasValidFp8Quantization(const Tensor* tensor) {
    return tensor != nullptr && HasCompatiblePerTensorQuantization(tensor->quantization()) &&
           std::isfinite(tensor->quantization_scale()) && tensor->quantization_scale() > 0.0f;
}

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

template <DataType dtype>
const float* Fp8DecodeTable() {
    static const std::array<float, 256> table = []() {
        std::array<float, 256> values{};
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = dtype == DataType::FP8E4M3 ? Fp8E4M3ToFloat(static_cast<uint8_t>(i))
                                                   : Fp8E5M2ToFloat(static_cast<uint8_t>(i));
        }
        return values;
    }();
    return table.data();
}

inline __m256 LoadFp8x8(const uint8_t* input, const float* table) {
    const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(input));
    const __m256i indices = _mm256_cvtepu8_epi32(bytes);
    return _mm256_i32gather_ps(table, indices, 4);
}

// FP8 normals map directly into IEEE-754 exponent/mantissa fields. This
// avoids an AVX2 gather for every eight elements in the hot FP8 Cast path.
// Subnormals and the format-specific special encodings are patched after the
// common normal reconstruction.
template <DataType dtype>
inline __m256 LoadFp8x8Typed(const uint8_t* input) {
    const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(input));
    const __m256i codes = _mm256_cvtepu8_epi32(bytes);
    const __m256i sign_codes = _mm256_and_si256(codes, _mm256_set1_epi32(0x80));
    const __m256i sign_bits = _mm256_slli_epi32(sign_codes, 24);
    const __m256 sign = _mm256_castsi256_ps(sign_bits);
    const int mantissa_shift = dtype == DataType::FP8E4M3 ? 3 : 2;
    const int mantissa_bits = dtype == DataType::FP8E4M3 ? 3 : 2;
    const __m256i mantissa_mask = _mm256_set1_epi32((1 << mantissa_bits) - 1);
    const __m256i mantissa = _mm256_and_si256(codes, mantissa_mask);
    const __m256i exponent = _mm256_and_si256(
        _mm256_srli_epi32(codes, mantissa_shift),
        _mm256_set1_epi32(dtype == DataType::FP8E4M3 ? 0x0f : 0x1f));
    const __m256i normal_exponent = _mm256_slli_epi32(
        _mm256_add_epi32(exponent, _mm256_set1_epi32(dtype == DataType::FP8E4M3 ? 120 : 112)), 23);
    const __m256i normal_mantissa = _mm256_slli_epi32(mantissa, dtype == DataType::FP8E4M3 ? 20 : 21);
    const __m256 normal = _mm256_castsi256_ps(
        _mm256_or_si256(sign_bits, _mm256_or_si256(normal_exponent, normal_mantissa)));
    const __m256 subnormal = _mm256_xor_ps(
        _mm256_mul_ps(_mm256_cvtepi32_ps(mantissa),
                      _mm256_set1_ps(dtype == DataType::FP8E4M3 ? 0.001953125f : 0.0000152587890625f)),
        sign);
    const __m256i exponent_zero = _mm256_cmpeq_epi32(exponent, _mm256_setzero_si256());
    __m256 result = _mm256_blendv_ps(normal, subnormal, _mm256_castsi256_ps(exponent_zero));

    if constexpr (dtype == DataType::FP8E4M3) {
        const __m256i nan_mask = _mm256_and_si256(
            _mm256_cmpeq_epi32(exponent, _mm256_set1_epi32(0x0f)),
            _mm256_cmpeq_epi32(mantissa, _mm256_set1_epi32(0x07)));
        result = _mm256_blendv_ps(result, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fc00000)),
                                  _mm256_castsi256_ps(nan_mask));
    } else {
        const __m256i infinity_mask = _mm256_and_si256(
            _mm256_cmpeq_epi32(exponent, _mm256_set1_epi32(0x1f)),
            _mm256_cmpeq_epi32(mantissa, _mm256_setzero_si256()));
        const __m256i nan_mask = _mm256_and_si256(
            _mm256_cmpeq_epi32(exponent, _mm256_set1_epi32(0x1f)),
            _mm256_cmpgt_epi32(mantissa, _mm256_setzero_si256()));
        result = _mm256_blendv_ps(result,
                                  _mm256_castsi256_ps(_mm256_or_si256(sign_bits, _mm256_set1_epi32(0x7f800000))),
                                  _mm256_castsi256_ps(infinity_mask));
        result = _mm256_blendv_ps(result, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fc00000)),
                                  _mm256_castsi256_ps(nan_mask));
    }
    return result;
}

template <DataType dtype>
int32_t ComputeFp8ToFp32(const feather::operators::CastParam* param) {
    const auto* input = static_cast<const uint8_t*>(param->input->raw_data());
    auto* output = static_cast<float*>(param->out->raw_data());
    const float* table = Fp8DecodeTable<dtype>();
    const __m256 scale = _mm256_set1_ps(param->input->quantization_scale());
    const int64_t count = param->input->numel();
    int64_t index = 0;
    for (; index + 8 <= count; index += 8) {
        _mm256_storeu_ps(output + index, _mm256_mul_ps(LoadFp8x8Typed<dtype>(input + index), scale));
    }
    for (; index < count; ++index) {
        output[index] = table[input[index]] * param->input->quantization_scale();
    }
    param->out->set_data_type(DataType::FP32);
    return 0;
}

template <DataType dtype>
int32_t ComputeFp8ToBf16(const feather::operators::CastParam* param) {
    const auto* input = static_cast<const uint8_t*>(param->input->raw_data());
    auto* output = static_cast<uint16_t*>(param->out->raw_data());
    const float* table = Fp8DecodeTable<dtype>();
    const __m256 scale = _mm256_set1_ps(param->input->quantization_scale());
    const int64_t count = param->input->numel();
    int64_t index = 0;
    for (; index + 8 <= count; index += 8) {
        StoreBf16x8(_mm256_mul_ps(LoadFp8x8Typed<dtype>(input + index), scale), output + index);
    }
    for (; index < count; ++index) {
        output[index] = FloatToBFloat16(table[input[index]] * param->input->quantization_scale());
    }
    param->out->set_data_type(DataType::BF16);
    return 0;
}

template <typename LoadFn>
int32_t EncodeFp8Vector(const feather::operators::CastParam* param, LoadFn&& load) {
    const int64_t count = param->input->numel();
    const float scale = param->out->quantization_scale();
    auto* output = static_cast<uint8_t*>(param->out->raw_data());
    const __m256 scale_vector = _mm256_set1_ps(scale);
    const auto encode_range = [&](int64_t begin, int64_t end) {
        alignas(32) float values[8];
        int64_t i = begin;
        for (; i + 8 <= end; i += 8) {
            for (int lane = 0; lane < 8; ++lane) {
                values[lane] = load(i + lane);
            }
            _mm256_store_ps(values, _mm256_div_ps(_mm256_load_ps(values), scale_vector));
            x86::EncodeFp8x8ForX86(param->to, values, output + i);
        }
        for (; i < end; ++i) {
            output[i] = x86::EncodeFp8ForX86(param->to, load(i) / scale);
        }
    };
#if defined(FEATHER_WITH_OPENMP)
    if (count >= (1 << 15) && !omp_in_parallel()) {
        const int workers = std::max(1, std::min<int>(4, omp_get_max_threads()));
        const int64_t chunk = (count / workers + 7) & ~int64_t{7};
#pragma omp parallel for schedule(static) num_threads(workers)
        for (int worker = 0; worker < workers; ++worker) {
            const int64_t begin = std::min<int64_t>(count, static_cast<int64_t>(worker) * chunk);
            const int64_t end = std::min<int64_t>(count, begin + chunk);
            encode_range(begin, end);
        }
    } else
#endif
    {
        encode_range(0, count);
    }
    param->out->set_data_type(param->to);
    return 0;
}

bool HasValidCastBuffers(const feather::operators::CastParam* param) {
    return param != nullptr && param->input != nullptr && param->out != nullptr && param->input->IsInitialized() &&
           param->out->IsInitialized() && param->input->dims().data() == param->out->dims().data() &&
           param->input->numel() == param->out->numel() &&
           param->input->numel() >= 0 && DataTypeBytes(param->input->data_type()) != 0 && DataTypeBytes(param->to) != 0 &&
           (param->out->data_type() == DataType::UNKNOWN || param->out->data_type() == param->to) &&
           static_cast<uint64_t>(param->input->numel()) <=
               std::numeric_limits<size_t>::max() / DataTypeBytes(param->input->data_type()) &&
           static_cast<uint64_t>(param->input->numel()) <= std::numeric_limits<size_t>::max() / DataTypeBytes(param->to) &&
           param->input->memory_size() >=
               static_cast<size_t>(param->input->numel()) * DataTypeBytes(param->input->data_type()) &&
           param->out->memory_size() >= static_cast<size_t>(param->input->numel()) * DataTypeBytes(param->to) &&
           (!IsFp8DataType(param->input->data_type()) || HasValidFp8Quantization(param->input.get())) &&
           (!IsFp8DataType(param->to) || HasValidFp8Quantization(param->out.get()));
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
            case DataType::FP8E4M3:
                static_cast<Fp8E4M3*>(param->out->raw_data())[i].bits =
                    x86::EncodeFp8ForX86(DataType::FP8E4M3, value / param->out->quantization_scale());
                break;
            case DataType::FP8E5M2:
                static_cast<Fp8E5M2*>(param->out->raw_data())[i].bits =
                    x86::EncodeFp8ForX86(DataType::FP8E5M2, value / param->out->quantization_scale());
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

template <DataType dtype>
int32_t ComputeFp8InputCast(feather::operators::CastParam* param) {
    if (param == nullptr || param->input == nullptr || param->input->data_type() != dtype) {
        return -1;
    }
    if (!HasValidCastBuffers(param)) return -1;
    if (param->to == DataType::FP32) return ComputeFp8ToFp32<dtype>(param);
    if (param->to == DataType::BF16) return ComputeFp8ToBf16<dtype>(param);
    return ComputeCastScalar(param);
}

int32_t ComputeBf16OutputFp8Cast(feather::operators::CastParam* param) {
    if (!HasValidCastBuffers(param) || param->input->data_type() != DataType::BF16 ||
        !IsFp8DataType(param->to)) {
        return -1;
    }
    const auto* input = static_cast<const uint16_t*>(param->input->raw_data());
    return EncodeFp8Vector(param, [input](int64_t index) { return BFloat16ToFloat(input[index]); });
}

int32_t ComputeFp32OutputFp8Cast(feather::operators::CastParam* param) {
    if (!HasValidCastBuffers(param) || param->input->data_type() != DataType::FP32 ||
        !IsFp8DataType(param->to)) {
        return -1;
    }
    const auto* input = static_cast<const float*>(param->input->raw_data());
    return EncodeFp8Vector(param, [input](int64_t index) { return input[index]; });
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
    if (param != nullptr && param->to != DataType::FP8E4M3 && param->to != DataType::FP8E5M2) {
        return ComputeCastScalar(param);
    }
    return ComputeBf16OutputFp8Cast(param);
}

template <>
int32_t CastKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::Cast::FP32");
    auto* param = static_cast<feather::operators::CastParam*>(param_);
    if (param != nullptr && param->input != nullptr && param->input->data_type() == DataType::FP32 &&
        param->to == DataType::BF16) {
        return ComputeFp32ToBf16(param);
    }
    if (param != nullptr && param->to != DataType::FP8E4M3 && param->to != DataType::FP8E5M2) {
        return ComputeCastScalar(param);
    }
    return ComputeFp32OutputFp8Cast(param);
}

template <>
int32_t CastKernel<DeviceType::X86, DataType::FP8E4M3>::compute() {
    AutoTimer timer("X86::Cast::FP8E4M3");
    return ComputeFp8InputCast<DataType::FP8E4M3>(static_cast<feather::operators::CastParam*>(param_));
}

template <>
int32_t CastKernel<DeviceType::X86, DataType::FP8E5M2>::compute() {
    AutoTimer timer("X86::Cast::FP8E5M2");
    return ComputeFp8InputCast<DataType::FP8E5M2>(static_cast<feather::operators::CastParam*>(param_));
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
        dispatcher.registerKernel(DeviceType::X86, DataType::FP8E4M3, "Cast", []() {
            return std::make_unique<CastKernel<DeviceType::X86, DataType::FP8E4M3>>();
        });
        dispatcher.registerKernel(DeviceType::X86, DataType::FP8E5M2, "Cast", []() {
            return std::make_unique<CastKernel<DeviceType::X86, DataType::FP8E5M2>>();
        });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
