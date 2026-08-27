#ifndef FEATHER_KERNEL_X86_FP8_UTILS_H
#define FEATHER_KERNEL_X86_FP8_UTILS_H

#include <cstdint>

#include "util/fp8.h"
#include "util/types.h"

namespace feather {
namespace kernel {
namespace x86 {

// The canonical encoder is shared with CUDA and favors portability.  On x86,
// normal FP8 values can be rounded directly from IEEE-754 fields; retain the
// canonical path for subnormal values to preserve every boundary behavior.
template <DataType dtype>
inline uint8_t EncodeFp8ForX86(float value) {
    static_assert(dtype == DataType::FP8E4M3 || dtype == DataType::FP8E5M2,
                  "EncodeFp8ForX86 requires an FP8 data type");
    const uint32_t bits = fp8_detail::FloatBits(value);
    const uint8_t sign = static_cast<uint8_t>((bits >> 24) & 0x80U);
    const uint32_t exponent = (bits >> 23) & 0xffU;
    const uint32_t fraction = bits & 0x007fffffU;
    if (exponent == 0xffU) {
        if constexpr (dtype == DataType::FP8E4M3) {
            return static_cast<uint8_t>(sign | (fraction == 0U ? 0x7eU : 0x7fU));
        }
        return static_cast<uint8_t>(sign | (fraction == 0U ? 0x7cU : 0x7dU));
    }

    constexpr int kMantissaBits = dtype == DataType::FP8E4M3 ? 3 : 2;
    constexpr int kRoundingShift = 23 - kMantissaBits;
    constexpr int kMinimumNormalFloatExponent = dtype == DataType::FP8E4M3 ? 121 : 113;
    if (exponent < static_cast<uint32_t>(kMinimumNormalFloatExponent)) {
        if constexpr (dtype == DataType::FP8E4M3) {
            return FloatToFp8E4M3(value);
        }
        return FloatToFp8E5M2(value);
    }

    int exponent_field = static_cast<int>(exponent) - 127 + (dtype == DataType::FP8E4M3 ? 7 : 15);
    uint32_t mantissa = fraction >> kRoundingShift;
    constexpr uint32_t kRemainderMask = (1U << kRoundingShift) - 1U;
    constexpr uint32_t kHalfway = 1U << (kRoundingShift - 1);
    const uint32_t remainder = fraction & kRemainderMask;
    if (remainder > kHalfway || (remainder == kHalfway && (mantissa & 1U) != 0U)) {
        ++mantissa;
    }
    if (mantissa == (1U << kMantissaBits)) {
        mantissa = 0;
        ++exponent_field;
    }

    if constexpr (dtype == DataType::FP8E4M3) {
        if (exponent_field > 15) {
            return static_cast<uint8_t>(sign | 0x7eU);
        }
        if (exponent_field == 15 && mantissa > 6U) {
            mantissa = 6U;
        }
    } else if (exponent_field >= 31) {
        return static_cast<uint8_t>(sign | 0x7bU);
    }
    return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exponent_field) << kMantissaBits) | mantissa);
}

inline uint8_t EncodeFp8ForX86(DataType dtype, float value) {
    if (dtype == DataType::FP8E4M3) {
        return EncodeFp8ForX86<DataType::FP8E4M3>(value);
    }
    return dtype == DataType::FP8E5M2 ? EncodeFp8ForX86<DataType::FP8E5M2>(value) : 0U;
}

// Encode eight FP32 values at once. The implementation is compiled inside the
// x86 backend's AVX2 translation unit; callers only need a regular pointer
// interface and therefore do not inherit an AVX2 compiler requirement.
void EncodeFp8x8ForX86(DataType dtype, const float* input, uint8_t* output);

}  // namespace x86
}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_X86_FP8_UTILS_H
