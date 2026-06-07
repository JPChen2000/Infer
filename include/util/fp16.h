#ifndef FEATHER_UTIL_FP16_H
#define FEATHER_UTIL_FP16_H

#include <cmath>
#include <cstdint>
#include <cstring>

namespace feather {

inline float HalfToFloat(uint16_t bits) {
    const uint32_t sign = (static_cast<uint32_t>(bits & 0x8000u)) << 16;
    uint32_t exp = (bits >> 10) & 0x1fu;
    uint32_t mantissa = bits & 0x03ffu;

    uint32_t out = 0;
    if (exp == 0) {
        if (mantissa == 0) {
            out = sign;
        } else {
            exp = 1;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --exp;
            }
            mantissa &= 0x03ffu;
            out = sign | ((exp + (127 - 15)) << 23) | (mantissa << 13);
        }
    } else if (exp == 0x1fu) {
        out = sign | 0x7f800000u | (mantissa << 13);
    } else {
        out = sign | ((exp + (127 - 15)) << 23) | (mantissa << 13);
    }

    float value = 0.0f;
    std::memcpy(&value, &out, sizeof(value));
    return value;
}

inline uint16_t FloatToHalf(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
    const uint32_t exp = (bits >> 23) & 0xffu;
    const uint32_t mantissa = bits & 0x007fffffu;

    if (exp == 0xffu) {
        if (mantissa == 0) {
            return static_cast<uint16_t>(sign | 0x7c00u);
        }
        return static_cast<uint16_t>(sign | 0x7c00u | (mantissa >> 13) | 1u);
    }

    const int32_t unbiased_exp = static_cast<int32_t>(exp) - 127;
    if (unbiased_exp > 15) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }

    if (unbiased_exp < -14) {
        if (unbiased_exp < -24) {
            return sign;
        }

        uint32_t subnormal = mantissa | 0x00800000u;
        const int32_t shift = -unbiased_exp - 14;
        const uint32_t rounded = subnormal + (1u << (shift + 12));
        return static_cast<uint16_t>(sign | (rounded >> (shift + 13)));
    }

    uint32_t rounded_mantissa = mantissa + 0x00001000u;
    uint16_t half_exp = static_cast<uint16_t>(unbiased_exp + 15);
    if (rounded_mantissa & 0x00800000u) {
        rounded_mantissa = 0;
        ++half_exp;
        if (half_exp >= 0x1fu) {
            return static_cast<uint16_t>(sign | 0x7c00u);
        }
    }

    return static_cast<uint16_t>(sign | (half_exp << 10) | (rounded_mantissa >> 13));
}

}  // namespace feather

#endif  // FEATHER_UTIL_FP16_H
