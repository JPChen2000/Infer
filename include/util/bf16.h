// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FEATHER_UTIL_BF16_H
#define FEATHER_UTIL_BF16_H

#include <cstdint>
#include <cstring>

namespace feather {

struct BFloat16 {
    uint16_t bits{};
};

inline float BFloat16ToFloat(uint16_t bits) {
    const uint32_t float_bits = static_cast<uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &float_bits, sizeof(value));
    return value;
}

inline uint16_t FloatToBFloat16(float value) {
    uint32_t float_bits = 0;
    std::memcpy(&float_bits, &value, sizeof(float_bits));

    const uint32_t exponent = float_bits & 0x7f800000u;
    const uint32_t mantissa = float_bits & 0x007fffffu;
    if (exponent == 0x7f800000u && mantissa != 0) {
        return static_cast<uint16_t>((float_bits >> 16) | 0x0040u);
    }
    const uint32_t round_bias = 0x7fffu + ((float_bits >> 16) & 1u);
    return static_cast<uint16_t>((float_bits + round_bias) >> 16);
}

}  // namespace feather

#endif  // FEATHER_UTIL_BF16_H
