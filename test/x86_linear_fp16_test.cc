#include <gtest/gtest.h>

#include <vector>

#include "src/kernel/x86/linear_fp16.h"
#include "util/fp16.h"

namespace {

using feather::FloatToHalf;
using feather::HalfToFloat;
using feather::kernel::x86::ComputeLinearRowMajorX86Fp16;
using feather::kernel::x86::LinearBiasType;

std::vector<float> ReferenceMatMul(const std::vector<float>& lhs, const std::vector<float>& rhs, int64_t m, int64_t k,
                                   int64_t n) {
    std::vector<float> out(static_cast<size_t>(m * n), 0.0f);
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t t = 0; t < k; ++t) {
                out[static_cast<size_t>(i * n + j)] += lhs[static_cast<size_t>(i * k + t)] *
                                                       rhs[static_cast<size_t>(t * n + j)];
            }
        }
    }
    return out;
}

std::vector<uint16_t> ToHalf(const std::vector<float>& values) {
    std::vector<uint16_t> out(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        out[i] = FloatToHalf(values[i]);
    }
    return out;
}

TEST(x86_linear_fp16_test, MatMulHandlesColumnTailWithoutPadding) {
    const int64_t m = 2;
    const int64_t k = 5;
    const int64_t n = 10;
    const std::vector<float> lhs = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
    };
    const std::vector<float> rhs = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
        5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    };
    std::vector<uint16_t> out(static_cast<size_t>(m * n), 0);

    ASSERT_EQ(ComputeLinearRowMajorX86Fp16(ToHalf(lhs).data(), ToHalf(rhs).data(), nullptr, m, k, n,
                                           LinearBiasType::kNone, out.data()),
              0);

    const std::vector<float> expected = ReferenceMatMul(lhs, rhs, m, k, n);
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(HalfToFloat(out[i]), expected[i], 3e-2f);
    }
}

TEST(x86_linear_fp16_test, GemmSupportsVectorAndMatrixBias) {
    const int64_t m = 2;
    const int64_t k = 3;
    const int64_t n = 9;
    const std::vector<float> lhs = {
        1, 2, 3,
        4, 5, 6,
    };
    const std::vector<float> rhs = {
        1, 2, 3, 4, 5, 6, 7, 8, 9,
        2, 3, 4, 5, 6, 7, 8, 9, 10,
        3, 4, 5, 6, 7, 8, 9, 10, 11,
    };
    const std::vector<float> product = ReferenceMatMul(lhs, rhs, m, k, n);
    const std::vector<float> vector_bias = {1, 0, -1, 2, -2, 3, -3, 4, -4};
    const std::vector<float> matrix_bias = {
        0, 1, 2, 3, 4, 5, 6, 7, 8,
        9, 8, 7, 6, 5, 4, 3, 2, 1,
    };

    std::vector<uint16_t> vector_out(static_cast<size_t>(m * n), 0);
    ASSERT_EQ(ComputeLinearRowMajorX86Fp16(ToHalf(lhs).data(), ToHalf(rhs).data(), ToHalf(vector_bias).data(), m, k, n,
                                           LinearBiasType::kVector, vector_out.data()),
              0);
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            EXPECT_NEAR(HalfToFloat(vector_out[static_cast<size_t>(i * n + j)]),
                        product[static_cast<size_t>(i * n + j)] + vector_bias[static_cast<size_t>(j)], 3e-2f);
        }
    }

    std::vector<uint16_t> matrix_out(static_cast<size_t>(m * n), 0);
    ASSERT_EQ(ComputeLinearRowMajorX86Fp16(ToHalf(lhs).data(), ToHalf(rhs).data(), ToHalf(matrix_bias).data(), m, k, n,
                                           LinearBiasType::kMatrix, matrix_out.data()),
              0);
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            EXPECT_NEAR(HalfToFloat(matrix_out[static_cast<size_t>(i * n + j)]),
                        product[static_cast<size_t>(i * n + j)] +
                            matrix_bias[static_cast<size_t>(i * n + j)],
                        3e-2f);
        }
    }
}

TEST(x86_linear_fp16_test, MatMulHandlesSixteenWideBlockEightWideBlockAndTail) {
    const int64_t m = 3;
    const int64_t k = 7;
    const int64_t n = 25;

    std::vector<float> lhs(static_cast<size_t>(m * k), 0.0f);
    std::vector<float> rhs(static_cast<size_t>(k * n), 0.0f);
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t t = 0; t < k; ++t) {
            lhs[static_cast<size_t>(i * k + t)] = 0.5f + static_cast<float>(i * 3 + t) * 0.25f;
        }
    }
    for (int64_t t = 0; t < k; ++t) {
        for (int64_t j = 0; j < n; ++j) {
            rhs[static_cast<size_t>(t * n + j)] = -0.75f + static_cast<float>((t * 5 + j * 2) % 19) * 0.125f;
        }
    }

    std::vector<uint16_t> out(static_cast<size_t>(m * n), 0);
    ASSERT_EQ(ComputeLinearRowMajorX86Fp16(ToHalf(lhs).data(), ToHalf(rhs).data(), nullptr, m, k, n,
                                           LinearBiasType::kNone, out.data()),
              0);

    const std::vector<float> expected = ReferenceMatMul(lhs, rhs, m, k, n);
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(HalfToFloat(out[i]), expected[i], 3e-2f);
    }
}

TEST(x86_linear_fp16_test, RejectsInvalidArguments) {
    uint16_t out = 0;
    EXPECT_EQ(ComputeLinearRowMajorX86Fp16(nullptr, nullptr, nullptr, 1, 1, 1, LinearBiasType::kNone, &out), -1);
}

}  // namespace
