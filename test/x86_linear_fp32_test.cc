#include <gtest/gtest.h>

#include <vector>

#include "src/kernel/x86/linear_fp32.h"

namespace {

using feather::kernel::x86::ComputeLinearRowMajorX86Fp32;
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

TEST(x86_linear_fp32_test, MatMulHandlesColumnTailWithoutPadding) {
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
    std::vector<float> out(static_cast<size_t>(m * n), 0.0f);

    ASSERT_EQ(ComputeLinearRowMajorX86Fp32(lhs.data(), rhs.data(), nullptr, m, k, n, LinearBiasType::kNone,
                                           out.data()),
              0);

    const std::vector<float> expected = ReferenceMatMul(lhs, rhs, m, k, n);
    EXPECT_EQ(out, expected);
}

TEST(x86_linear_fp32_test, GemmSupportsVectorAndMatrixBias) {
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

    std::vector<float> vector_out(static_cast<size_t>(m * n), 0.0f);
    ASSERT_EQ(ComputeLinearRowMajorX86Fp32(lhs.data(), rhs.data(), vector_bias.data(), m, k, n,
                                           LinearBiasType::kVector, vector_out.data()),
              0);
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            EXPECT_FLOAT_EQ(vector_out[static_cast<size_t>(i * n + j)],
                            product[static_cast<size_t>(i * n + j)] + vector_bias[static_cast<size_t>(j)]);
        }
    }

    std::vector<float> matrix_out(static_cast<size_t>(m * n), 0.0f);
    ASSERT_EQ(ComputeLinearRowMajorX86Fp32(lhs.data(), rhs.data(), matrix_bias.data(), m, k, n,
                                           LinearBiasType::kMatrix, matrix_out.data()),
              0);
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            EXPECT_FLOAT_EQ(matrix_out[static_cast<size_t>(i * n + j)],
                            product[static_cast<size_t>(i * n + j)] +
                                matrix_bias[static_cast<size_t>(i * n + j)]);
        }
    }
}

TEST(x86_linear_fp32_test, RejectsInvalidArguments) {
    float out = 0.0f;
    EXPECT_EQ(ComputeLinearRowMajorX86Fp32(nullptr, nullptr, nullptr, 1, 1, 1, LinearBiasType::kNone, &out), -1);
}

}  // namespace
