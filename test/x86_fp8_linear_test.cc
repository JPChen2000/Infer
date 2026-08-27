#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <cstdint>
#include <limits>
#include <vector>

#include "src/kernel/x86/fp8_utils.h"
#include "src/kernel/x86/linear_fp8.h"
#include "util/fp8.h"
#include "util/types.h"

namespace {

using feather::DataType;
using feather::FloatToFp8E4M3;
using feather::FloatToFp8E5M2;
using feather::Fp8E4M3ToFloat;
using feather::Fp8E5M2ToFloat;
using feather::kernel::x86::ComputeLinearRowMajorX86Fp8;
using feather::kernel::x86::ComputeLinearRowMajorX86Fp8TransposedRhs;
using feather::kernel::x86::ComputeLinearRowMajorX86Fp8PackedRhs;
using feather::kernel::x86::ComputeLinearRowMajorX86Fp8PackedTransposedRhs;
using feather::kernel::x86::LinearBiasType;
using feather::kernel::x86::PackedFp8Rhs;
using feather::kernel::x86::PackedFp8TransposedRhs;

template <DataType dtype>
uint8_t Encode(float value, float scale) {
    if constexpr (dtype == DataType::FP8E4M3) {
        return FloatToFp8E4M3(value / scale);
    }
    return FloatToFp8E5M2(value / scale);
}

template <DataType dtype>
float Decode(uint8_t value, float scale) {
    if constexpr (dtype == DataType::FP8E4M3) {
        return Fp8E4M3ToFloat(value) * scale;
    }
    return Fp8E5M2ToFloat(value) * scale;
}

template <DataType dtype>
void CheckRowMajorPath() {
    constexpr int64_t m = 2;
    constexpr int64_t k = 17;
    constexpr int64_t n = 64;
    constexpr float lhs_scale = 0.25f;
    constexpr float rhs_scale = 0.5f;
    constexpr float bias_scale = 0.25f;
    constexpr float out_scale = 0.25f;

    std::vector<uint8_t> lhs(static_cast<size_t>(m * k));
    std::vector<uint8_t> rhs(static_cast<size_t>(k * n));
    std::vector<uint8_t> bias(static_cast<size_t>(n));
    std::vector<uint8_t> out(static_cast<size_t>(m * n), 0);
    for (int64_t i = 0; i < m * k; ++i) {
        lhs[static_cast<size_t>(i)] = Encode<dtype>(static_cast<float>((i % 7) - 3) * lhs_scale, lhs_scale);
    }
    for (int64_t i = 0; i < k * n; ++i) {
        rhs[static_cast<size_t>(i)] = Encode<dtype>(static_cast<float>((i % 9) - 4) * rhs_scale, rhs_scale);
    }
    for (int64_t i = 0; i < n; ++i) {
        bias[static_cast<size_t>(i)] = Encode<dtype>(static_cast<float>((i % 5) - 2) * bias_scale, bias_scale);
    }

    ASSERT_EQ(ComputeLinearRowMajorX86Fp8(
                  dtype, lhs.data(), lhs_scale, rhs.data(), rhs_scale, bias.data(), bias_scale, m, k, n,
                  LinearBiasType::kVector, out.data(), out_scale),
              0);
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t col = 0; col < n; ++col) {
            float expected = Decode<dtype>(bias[static_cast<size_t>(col)], bias_scale);
            for (int64_t inner = 0; inner < k; ++inner) {
                expected += Decode<dtype>(lhs[static_cast<size_t>(row * k + inner)], lhs_scale) *
                            Decode<dtype>(rhs[static_cast<size_t>(inner * n + col)], rhs_scale);
            }
            EXPECT_NEAR(Decode<dtype>(out[static_cast<size_t>(row * n + col)], out_scale), expected, 2.0f);
        }
    }
}

TEST(x86_fp8_linear_test, RowMajorVectorizedPathPreservesScaledValues) {
    CheckRowMajorPath<DataType::FP8E4M3>();
    CheckRowMajorPath<DataType::FP8E5M2>();
}

template <DataType dtype>
void CheckTransposedPath() {
    constexpr int64_t m = 1;
    constexpr int64_t k = 33;
    constexpr int64_t n = 128;
    constexpr float lhs_scale = 0.5f;
    constexpr float rhs_scale = 0.25f;
    constexpr float out_scale = 0.5f;
    std::vector<uint8_t> lhs(static_cast<size_t>(k));
    std::vector<uint8_t> rhs(static_cast<size_t>(n * k));
    std::vector<uint8_t> out(static_cast<size_t>(n), 0);
    for (int64_t i = 0; i < k; ++i) {
        lhs[static_cast<size_t>(i)] = Encode<dtype>(static_cast<float>((i % 11) - 5) * lhs_scale, lhs_scale);
    }
    for (int64_t i = 0; i < n * k; ++i) {
        rhs[static_cast<size_t>(i)] = Encode<dtype>(static_cast<float>((i % 13) - 6) * rhs_scale, rhs_scale);
    }

    ASSERT_EQ(ComputeLinearRowMajorX86Fp8TransposedRhs(
                  dtype, lhs.data(), lhs_scale, rhs.data(), rhs_scale, nullptr, 1.0f, 1.0f, 1.0f, m, k, n,
                  LinearBiasType::kNone, out.data(), out_scale),
              0);
    for (int64_t col = 0; col < n; ++col) {
        float expected = 0.0f;
        for (int64_t inner = 0; inner < k; ++inner) {
            expected += Decode<dtype>(lhs[static_cast<size_t>(inner)], lhs_scale) *
                        Decode<dtype>(rhs[static_cast<size_t>(col * k + inner)], rhs_scale);
        }
        EXPECT_NEAR(Decode<dtype>(out[static_cast<size_t>(col)], out_scale), expected, 4.0f);
    }
}

TEST(x86_fp8_linear_test, TransposedRhsVectorizedPathPreservesScaledValues) {
    CheckTransposedPath<DataType::FP8E4M3>();
    CheckTransposedPath<DataType::FP8E5M2>();
}

TEST(x86_fp8_linear_test, E5M2VectorPathPreservesInfinity) {
    constexpr int64_t k = 8;
    constexpr int64_t n = 8;
    std::vector<uint8_t> lhs(static_cast<size_t>(k), FloatToFp8E5M2(1.0f));
    std::vector<uint8_t> rhs(static_cast<size_t>(k * n), FloatToFp8E5M2(0.0f));
    for (int64_t row = 0; row < k; ++row) {
        rhs[static_cast<size_t>(row * n)] = 0x7cU;
    }
    std::vector<uint8_t> out(static_cast<size_t>(n), 0);
    ASSERT_EQ(ComputeLinearRowMajorX86Fp8(
                  DataType::FP8E5M2, lhs.data(), 1.0f, rhs.data(), 1.0f, nullptr, 1.0f, 1, k, n,
                  LinearBiasType::kNone, out.data(), 1.0f),
              0);
    EXPECT_EQ(out[0], 0x7cU);
}

TEST(x86_fp8_linear_test, PackedRhsMatchesUnpackedResult) {
    constexpr int64_t m = 1;
    constexpr int64_t k = 17;
    constexpr int64_t n = 128;
    std::vector<uint8_t> lhs(static_cast<size_t>(m * k));
    std::vector<uint8_t> rhs(static_cast<size_t>(k * n));
    for (int64_t index = 0; index < m * k; ++index) {
        lhs[static_cast<size_t>(index)] = FloatToFp8E4M3(static_cast<float>((index % 7) - 3));
    }
    for (int64_t index = 0; index < k * n; ++index) {
        rhs[static_cast<size_t>(index)] = FloatToFp8E4M3(static_cast<float>((index % 11) - 5) * 0.25f);
    }
    std::vector<uint8_t> expected(static_cast<size_t>(m * n), 0);
    std::vector<uint8_t> actual(static_cast<size_t>(m * n), 0);
    ASSERT_EQ(ComputeLinearRowMajorX86Fp8(
                  DataType::FP8E4M3, lhs.data(), 1.0f, rhs.data(), 1.0f, nullptr, 1.0f, m, k, n,
                  LinearBiasType::kNone, expected.data(), 1.0f),
              0);

    PackedFp8Rhs packed;
    ASSERT_TRUE(packed.Pack(DataType::FP8E4M3, rhs.data(), k, n));
    ASSERT_EQ(ComputeLinearRowMajorX86Fp8PackedRhs(
                  DataType::FP8E4M3, lhs.data(), 1.0f, rhs.data(), 1.0f, packed, nullptr, 1.0f, m, k, n,
                  LinearBiasType::kNone, actual.data(), 1.0f),
              0);
    EXPECT_EQ(actual, expected);
}

TEST(x86_fp8_linear_test, PackedRhsHandlesNonAlignedVectorBias) {
    constexpr int64_t m = 1;
    constexpr int64_t k = 17;
    constexpr int64_t n = 5;
    constexpr float lhs_scale = 0.25f;
    constexpr float rhs_scale = 0.5f;
    constexpr float bias_scale = 0.25f;
    constexpr float out_scale = 0.25f;

    std::vector<uint8_t> lhs(static_cast<size_t>(k));
    std::vector<uint8_t> rhs(static_cast<size_t>(k * n));
    std::vector<uint8_t> bias(static_cast<size_t>(n));
    std::vector<uint8_t> expected(static_cast<size_t>(n), 0);
    std::vector<uint8_t> actual(static_cast<size_t>(n), 0);
    for (int64_t index = 0; index < k; ++index) {
        lhs[static_cast<size_t>(index)] = Encode<DataType::FP8E4M3>(static_cast<float>((index % 7) - 3) * lhs_scale,
                                                                       lhs_scale);
    }
    for (int64_t index = 0; index < k * n; ++index) {
        rhs[static_cast<size_t>(index)] = Encode<DataType::FP8E4M3>(static_cast<float>((index % 11) - 5) * rhs_scale,
                                                                       rhs_scale);
    }
    for (int64_t index = 0; index < n; ++index) {
        bias[static_cast<size_t>(index)] = Encode<DataType::FP8E4M3>(static_cast<float>((index % 5) - 2) * bias_scale,
                                                                       bias_scale);
    }

    ASSERT_EQ(ComputeLinearRowMajorX86Fp8(
                  DataType::FP8E4M3, lhs.data(), lhs_scale, rhs.data(), rhs_scale, bias.data(), bias_scale, m, k, n,
                  LinearBiasType::kVector, expected.data(), out_scale),
              0);
    PackedFp8Rhs packed;
    ASSERT_TRUE(packed.Pack(DataType::FP8E4M3, rhs.data(), k, n));
    ASSERT_EQ(ComputeLinearRowMajorX86Fp8PackedRhs(
                  DataType::FP8E4M3, lhs.data(), lhs_scale, rhs.data(), rhs_scale, packed, bias.data(), bias_scale, m,
                  k, n, LinearBiasType::kVector, actual.data(), out_scale),
              0);
    for (int64_t col = 0; col < n; ++col) {
        EXPECT_NEAR(Decode<DataType::FP8E4M3>(actual[static_cast<size_t>(col)], out_scale),
                    Decode<DataType::FP8E4M3>(expected[static_cast<size_t>(col)], out_scale), 2.0f);
    }
}

TEST(x86_fp8_linear_test, PackedTransposedRhsMatchesUnpackedResult) {
    constexpr int64_t m = 1;
    constexpr int64_t k = 17;
    constexpr int64_t n = 64;
    std::vector<uint8_t> lhs(static_cast<size_t>(k));
    std::vector<uint8_t> rhs(static_cast<size_t>(n * k));
    std::vector<uint8_t> expected(static_cast<size_t>(n), 0);
    std::vector<uint8_t> actual(static_cast<size_t>(n), 0);
    for (int64_t index = 0; index < k; ++index) {
        lhs[static_cast<size_t>(index)] = FloatToFp8E5M2(static_cast<float>((index % 7) - 3));
    }
    for (int64_t index = 0; index < n * k; ++index) {
        rhs[static_cast<size_t>(index)] = FloatToFp8E5M2(static_cast<float>((index % 11) - 5) * 0.25f);
    }
    ASSERT_EQ(ComputeLinearRowMajorX86Fp8TransposedRhs(
                  DataType::FP8E5M2, lhs.data(), 1.0f, rhs.data(), 1.0f, nullptr, 1.0f, 1.0f, 1.0f, m, k, n,
                  LinearBiasType::kNone, expected.data(), 1.0f),
              0);
    PackedFp8TransposedRhs packed;
    ASSERT_TRUE(packed.Pack(DataType::FP8E5M2, rhs.data(), k, n));
    ASSERT_EQ(ComputeLinearRowMajorX86Fp8PackedTransposedRhs(
                  DataType::FP8E5M2, lhs.data(), 1.0f, rhs.data(), 1.0f, packed, nullptr, 1.0f, m, k, n,
                  LinearBiasType::kNone, actual.data(), 1.0f),
              0);
    EXPECT_EQ(actual, expected);
}

TEST(x86_fp8_linear_test, RejectsInvalidScaleOrPointers) {
    uint8_t value = 0;
    EXPECT_EQ(ComputeLinearRowMajorX86Fp8(
                  DataType::FP8E4M3, &value, 0.0f, &value, 1.0f, nullptr, 1.0f, 1, 1, 1,
                  LinearBiasType::kNone, &value, 1.0f),
              -1);
    EXPECT_EQ(ComputeLinearRowMajorX86Fp8(
                  DataType::FP8E5M2, nullptr, 1.0f, &value, 1.0f, nullptr, 1.0f, 1, 1, 1,
                  LinearBiasType::kNone, &value, 1.0f),
              -1);
}

TEST(x86_fp8_linear_test, FastFp8EncoderMatchesCanonicalEncoding) {
    const std::array<uint32_t, 22> boundary_bits = {
        0x00000000U, 0x80000000U, 0x00000001U, 0x80000001U, 0x387fffffU, 0x38800000U,
        0x477fffffU, 0x47800000U, 0x43dfffffU, 0x43e00000U, 0x47600000U, 0x47600001U,
        0x7f7fffffU, 0xff7fffffU, 0x7f800000U, 0xff800000U, 0x7fc00000U, 0xffc00000U,
        0x00800000U, 0x80800000U, 0x3f800000U, 0xbf800000U,
    };
    uint32_t state = 0x12345678U;
    for (const auto dtype : {DataType::FP8E4M3, DataType::FP8E5M2}) {
        const auto check = [dtype](uint32_t bits) {
            float value = 0.0f;
            std::memcpy(&value, &bits, sizeof(value));
            const uint8_t expected = dtype == DataType::FP8E4M3 ? FloatToFp8E4M3(value) : FloatToFp8E5M2(value);
            EXPECT_EQ(feather::kernel::x86::EncodeFp8ForX86(dtype, value), expected)
                << "dtype=" << static_cast<int>(dtype) << " bits=" << bits;
        };
        for (const uint32_t bits : boundary_bits) {
            check(bits);
        }
        for (int index = 0; index < 100000; ++index) {
            state = state * 1664525U + 1013904223U;
            check(state);
        }
    }
}

TEST(x86_fp8_linear_test, FastFp8VectorEncoderMatchesCanonicalEncoding) {
    const std::array<float, 24> values = {
        0.0f,          -0.0f,          0.001953125f,  -0.001953125f,  0.015625f,    -0.015625f,
        0.0625f,       -0.0625f,       1.0f,          -1.0f,           448.0f,       -448.0f,
        57344.0f,      -57344.0f,      1.0e-30f,      -1.0e-30f,       1.0e30f,      -1.0e30f,
        std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity() * -1.0f,
        std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(),
        0.333251953125f, -0.333251953125f,
    };
    for (const auto dtype : {DataType::FP8E4M3, DataType::FP8E5M2}) {
        for (size_t begin = 0; begin < values.size(); begin += 8) {
            std::array<uint8_t, 8> actual{};
            feather::kernel::x86::EncodeFp8x8ForX86(dtype, values.data() + begin, actual.data());
            for (size_t lane = 0; lane < actual.size(); ++lane) {
                const float value = values[begin + lane];
                const uint8_t expected = dtype == DataType::FP8E4M3 ? FloatToFp8E4M3(value) : FloatToFp8E5M2(value);
                EXPECT_EQ(actual[lane], expected)
                    << "dtype=" << static_cast<int>(dtype) << " lane=" << lane << " value=" << value;
            }
        }
    }
}

}  // namespace
