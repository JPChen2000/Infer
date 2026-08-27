#ifndef FEATHER_INFER_UTIL_FP8_H
#define FEATHER_INFER_UTIL_FP8_H

#include <cstdint>
#include <cstring>
#include <cmath>

#if defined(__CUDACC__)
#define FEATHER_FP8_HD __host__ __device__
#else
#define FEATHER_FP8_HD
#endif

namespace feather {

// The wrappers deliberately remain distinct from UINT8.  Their one-byte
// payload is an encoded floating-point value, not an integer tensor.
struct Fp8E4M3 {
    uint8_t bits{0};
};

struct Fp8E5M2 {
    uint8_t bits{0};
};

static_assert(sizeof(Fp8E4M3) == 1, "FP8 E4M3 storage must be one byte");
static_assert(sizeof(Fp8E5M2) == 1, "FP8 E5M2 storage must be one byte");

namespace fp8_detail {

FEATHER_FP8_HD inline uint32_t FloatBits(float value) {
#if defined(__CUDA_ARCH__)
    return __float_as_uint(value);
#else
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
#endif
}

FEATHER_FP8_HD inline float FloatFromBits(uint32_t bits) {
#if defined(__CUDA_ARCH__)
    return __uint_as_float(bits);
#else
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
#endif
}

FEATHER_FP8_HD inline int32_t RoundToNearestEven(float value) {
    const int32_t lower = static_cast<int32_t>(value);
    const float fraction = value - static_cast<float>(lower);
    if (fraction > 0.5f || (fraction == 0.5f && (lower & 1) != 0)) {
        return lower + 1;
    }
    return lower;
}

FEATHER_FP8_HD inline uint8_t SignBit(float value) {
    return static_cast<uint8_t>((FloatBits(value) >> 24) & 0x80u);
}

FEATHER_FP8_HD inline float QuietNaN() { return FloatFromBits(0x7fc00000u); }

FEATHER_FP8_HD inline uint8_t EncodeFinite(float absolute_value, uint8_t sign, int mantissa_bits, int bias,
                                             int max_normal_exponent, int max_normal_mantissa, bool has_infinity) {
    const float min_normal = has_infinity ? 0.00006103515625f : 0.015625f;
    const float quantum = has_infinity ? 0.0000152587890625f : 0.001953125f;
    const float max_finite = has_infinity ? 57344.0f : 448.0f;
    const uint8_t max_code = has_infinity ? 0x7bU : 0x7eU;
    const int exponent_mask = has_infinity ? 0x1f : 0x0f;

    if (absolute_value < min_normal) {
        const int32_t subnormal = RoundToNearestEven(absolute_value / quantum);
        if (subnormal <= 0) {
            return sign;
        }
        if (subnormal >= (1 << mantissa_bits)) {
            return static_cast<uint8_t>(sign | (1 << mantissa_bits));
        }
        return static_cast<uint8_t>(sign | subnormal);
    }

    if (absolute_value > max_finite) {
        return static_cast<uint8_t>(sign | max_code);
    }

    int exponent = 0;
    float fraction = absolute_value;
    // frexpf is available in both the host and CUDA device math libraries.
    while (fraction >= 2.0f) {
        fraction *= 0.5f;
        ++exponent;
    }
    while (fraction < 1.0f) {
        fraction *= 2.0f;
        --exponent;
    }
    int32_t mantissa = RoundToNearestEven((fraction - 1.0f) * static_cast<float>(1 << mantissa_bits));
    if (mantissa >= (1 << mantissa_bits)) {
        mantissa = 0;
        ++exponent;
    }

    if (exponent > max_normal_exponent) {
        return static_cast<uint8_t>(sign | max_code);
    }
    int exponent_field = exponent + bias;
    if (exponent_field <= 0) {
        const int32_t subnormal = RoundToNearestEven(absolute_value / quantum);
        if (subnormal <= 0) {
            return sign;
        }
        if (subnormal >= (1 << mantissa_bits)) {
            return static_cast<uint8_t>(sign | (1 << mantissa_bits));
        }
        return static_cast<uint8_t>(sign | subnormal);
    }
    if (exponent_field >= exponent_mask) {
        exponent_field = exponent_mask;
        if (mantissa > max_normal_mantissa) {
            mantissa = max_normal_mantissa;
        }
    }
    return static_cast<uint8_t>(sign | (exponent_field << mantissa_bits) | mantissa);
}

FEATHER_FP8_HD inline uint8_t EncodeE4M3(float value) {
    const uint8_t sign = SignBit(value);
    const uint32_t bits = FloatBits(value);
    const uint32_t exponent = (bits >> 23) & 0xffu;
    const uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu) {
        if (mantissa != 0) {
            return static_cast<uint8_t>(sign | 0x7fU);
        }
        return static_cast<uint8_t>(sign | 0x7eU);
    }
    const float absolute_value = value < 0.0f ? -value : value;
    return EncodeFinite(absolute_value, sign, 3, 7, 8, 6, false);
}

FEATHER_FP8_HD inline uint8_t EncodeE5M2(float value) {
    const uint8_t sign = SignBit(value);
    const uint32_t bits = FloatBits(value);
    const uint32_t exponent = (bits >> 23) & 0xffu;
    const uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu) {
        return static_cast<uint8_t>(sign | (mantissa == 0 ? 0x7cU : 0x7dU));
    }
    const float absolute_value = value < 0.0f ? -value : value;
    return EncodeFinite(absolute_value, sign, 2, 15, 15, 3, true);
}

FEATHER_FP8_HD inline float DecodeE4M3(uint8_t bits) {
    const uint8_t sign = bits & 0x80u;
    const int exponent = (bits >> 3) & 0x0f;
    const int mantissa = bits & 0x07;
    if (exponent == 0 && mantissa == 0) {
        return sign == 0 ? 0.0f : -0.0f;
    }
    if (exponent == 15 && mantissa == 7) {
        return QuietNaN();
    }
    const float value = exponent == 0 ? static_cast<float>(mantissa) * 0.001953125f
                                      : (1.0f + static_cast<float>(mantissa) * 0.125f) *
                                            ldexpf(1.0f, exponent - 7);
    return sign == 0 ? value : -value;
}

FEATHER_FP8_HD inline float DecodeE5M2(uint8_t bits) {
    const uint8_t sign = bits & 0x80u;
    const int exponent = (bits >> 2) & 0x1f;
    const int mantissa = bits & 0x03;
    if (exponent == 0 && mantissa == 0) {
        return sign == 0 ? 0.0f : -0.0f;
    }
    if (exponent == 31) {
        if (mantissa == 0) {
            return sign == 0 ? FloatFromBits(0x7f800000u) : FloatFromBits(0xff800000u);
        }
        return QuietNaN();
    }
    const float value = exponent == 0 ? static_cast<float>(mantissa) * 0.0000152587890625f
                                      : (1.0f + static_cast<float>(mantissa) * 0.25f) *
                                            ldexpf(1.0f, exponent - 15);
    return sign == 0 ? value : -value;
}

}  // namespace fp8_detail

FEATHER_FP8_HD inline float Fp8E4M3ToFloat(uint8_t bits) { return fp8_detail::DecodeE4M3(bits); }
FEATHER_FP8_HD inline uint8_t FloatToFp8E4M3(float value) { return fp8_detail::EncodeE4M3(value); }
FEATHER_FP8_HD inline float Fp8E5M2ToFloat(uint8_t bits) { return fp8_detail::DecodeE5M2(bits); }
FEATHER_FP8_HD inline uint8_t FloatToFp8E5M2(float value) { return fp8_detail::EncodeE5M2(value); }

}  // namespace feather

#undef FEATHER_FP8_HD

#endif  // FEATHER_INFER_UTIL_FP8_H
