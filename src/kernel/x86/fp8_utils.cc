#include "src/kernel/x86/fp8_utils.h"

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

namespace feather {
namespace kernel {
namespace x86 {

namespace {

template <DataType dtype>
void EncodeFp8x8Impl(__m256 values, uint8_t* output) {
    static_assert(dtype == DataType::FP8E4M3 || dtype == DataType::FP8E5M2,
                  "EncodeFp8x8Impl requires an FP8 data type");
    if (output == nullptr) {
        return;
    }

    const __m256i bits = _mm256_castps_si256(values);
    const __m256i sign = _mm256_and_si256(_mm256_srli_epi32(bits, 24), _mm256_set1_epi32(0x80));
    const __m256i exponent = _mm256_and_si256(_mm256_srli_epi32(bits, 23), _mm256_set1_epi32(0xff));
    const __m256i fraction = _mm256_and_si256(bits, _mm256_set1_epi32(0x007fffff));
    constexpr int kMantissaBits = dtype == DataType::FP8E4M3 ? 3 : 2;
    constexpr int kRoundingShift = 23 - kMantissaBits;
    constexpr int kMinimumNormalFloatExponent = dtype == DataType::FP8E4M3 ? 121 : 113;
    constexpr int kExponentBias = dtype == DataType::FP8E4M3 ? 7 : 15;

    const __m256i exponent_is_special = _mm256_cmpeq_epi32(exponent, _mm256_set1_epi32(0xff));
    const __m256i normal_candidate =
        _mm256_cmpgt_epi32(exponent, _mm256_set1_epi32(kMinimumNormalFloatExponent - 1));
    const __m256i normal_mask = _mm256_andnot_si256(exponent_is_special, normal_candidate);
    const __m256i exponent_field = _mm256_add_epi32(
        _mm256_sub_epi32(exponent, _mm256_set1_epi32(127)), _mm256_set1_epi32(kExponentBias));
    const __m256i mantissa = _mm256_srli_epi32(fraction, kRoundingShift);
    const __m256i remainder =
        _mm256_and_si256(fraction, _mm256_set1_epi32((1U << kRoundingShift) - 1U));
    const __m256i halfway = _mm256_set1_epi32(1U << (kRoundingShift - 1));
    const __m256i round_up = _mm256_or_si256(
        _mm256_cmpgt_epi32(remainder, halfway),
        _mm256_and_si256(_mm256_cmpeq_epi32(remainder, halfway),
                         _mm256_cmpgt_epi32(_mm256_and_si256(mantissa, _mm256_set1_epi32(1)),
                                            _mm256_setzero_si256())));
    const __m256i rounded_mantissa =
        _mm256_add_epi32(mantissa, _mm256_and_si256(round_up, _mm256_set1_epi32(1)));
    const __m256i mantissa_carry =
        _mm256_cmpeq_epi32(rounded_mantissa, _mm256_set1_epi32(1 << kMantissaBits));
    const __m256i final_mantissa = _mm256_andnot_si256(mantissa_carry, rounded_mantissa);
    const __m256i final_exponent_field = _mm256_add_epi32(
        exponent_field, _mm256_and_si256(mantissa_carry, _mm256_set1_epi32(1)));

    __m256i normal_code;
    __m256i special_code;
    if constexpr (dtype == DataType::FP8E4M3) {
        const __m256i mantissa_clamp = _mm256_and_si256(
            _mm256_cmpeq_epi32(final_exponent_field, _mm256_set1_epi32(15)),
            _mm256_cmpgt_epi32(final_mantissa, _mm256_set1_epi32(6)));
        const __m256i clamped_mantissa =
            _mm256_blendv_epi8(final_mantissa, _mm256_set1_epi32(6), mantissa_clamp);
        const __m256i base_code = _mm256_or_si256(
            _mm256_slli_epi32(final_exponent_field, 3), clamped_mantissa);
        const __m256i saturated = _mm256_or_si256(
            _mm256_cmpgt_epi32(final_exponent_field, _mm256_set1_epi32(15)), mantissa_clamp);
        normal_code = _mm256_blendv_epi8(base_code, _mm256_set1_epi32(0x7e), saturated);
        const __m256i has_nan_payload = _mm256_cmpgt_epi32(fraction, _mm256_setzero_si256());
        special_code = _mm256_or_si256(
            sign, _mm256_blendv_epi8(_mm256_set1_epi32(0x7e), _mm256_set1_epi32(0x7f), has_nan_payload));
    } else {
        const __m256i base_code = _mm256_or_si256(
            _mm256_slli_epi32(final_exponent_field, 2), final_mantissa);
        const __m256i saturated = _mm256_cmpgt_epi32(final_exponent_field, _mm256_set1_epi32(30));
        normal_code = _mm256_blendv_epi8(base_code, _mm256_set1_epi32(0x7b), saturated);
        const __m256i has_nan_payload = _mm256_cmpgt_epi32(fraction, _mm256_setzero_si256());
        special_code = _mm256_or_si256(
            sign, _mm256_blendv_epi8(_mm256_set1_epi32(0x7c), _mm256_set1_epi32(0x7d), has_nan_payload));
    }
    normal_code = _mm256_or_si256(normal_code, sign);
    const __m256i encoded = _mm256_blendv_epi8(normal_code, special_code, exponent_is_special);
    const __m256i fast_mask = _mm256_or_si256(normal_mask, exponent_is_special);

    alignas(32) uint32_t encoded_lanes[8];
    alignas(32) uint32_t fast_lanes[8];
    alignas(32) float input_lanes[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(encoded_lanes), encoded);
    _mm256_store_si256(reinterpret_cast<__m256i*>(fast_lanes), fast_mask);
    _mm256_store_ps(input_lanes, values);
    for (size_t lane = 0; lane < 8; ++lane) {
        output[lane] = fast_lanes[lane] != 0
                           ? static_cast<uint8_t>(encoded_lanes[lane])
                           : EncodeFp8ForX86<dtype>(input_lanes[lane]);
    }
}

}  // namespace

void EncodeFp8x8ForX86(DataType dtype, const float* input, uint8_t* output) {
    if (input == nullptr || output == nullptr) {
        return;
    }
    const __m256 values = _mm256_loadu_ps(input);
    if (dtype == DataType::FP8E4M3) {
        EncodeFp8x8Impl<DataType::FP8E4M3>(values, output);
    } else if (dtype == DataType::FP8E5M2) {
        EncodeFp8x8Impl<DataType::FP8E5M2>(values, output);
    }
}

}  // namespace x86
}  // namespace kernel
}  // namespace feather
