#include <gtest/gtest.h>

#include <vector>

#include "src/kernel/x86/linear_fp32.h"
#include "util/threading.h"

#ifdef FEATHER_WITH_OPENMP
#include <omp.h>
#endif

namespace {

using feather::kernel::x86::ComputeLinearRowMajorX86Fp32;
using feather::kernel::x86::Fp32LinearUsesBlockedKernel;
using feather::kernel::x86::Fp32LinearWorkerCount;
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

TEST(x86_linear_fp32_test, WorkerPolicyParallelizesTransformerRows) {
#ifdef FEATHER_WITH_OPENMP
    const int previous_dynamic = omp_get_dynamic();
    const int previous_threads = omp_get_max_threads();
    omp_set_dynamic(0);
    omp_set_num_threads(static_cast<int>(feather::DefaultThreadCount()));
    EXPECT_GT(Fp32LinearWorkerCount(197, 768, 3072), 1U);
    omp_set_num_threads(previous_threads);
    omp_set_dynamic(previous_dynamic);
#else
    GTEST_SKIP() << "OpenMP is disabled for this build";
#endif
}

TEST(x86_linear_fp32_test, BlockedKernelPolicyTargetsTransformerMatrices) {
    EXPECT_TRUE(Fp32LinearUsesBlockedKernel(197, 768, 768));
    EXPECT_TRUE(Fp32LinearUsesBlockedKernel(197, 768, 3072));
    EXPECT_FALSE(Fp32LinearUsesBlockedKernel(1, 768, 3072));
    EXPECT_FALSE(Fp32LinearUsesBlockedKernel(2, 3, 4));
}

TEST(x86_linear_fp32_test, BlockedMatMulPreservesNumericalResults) {
    const int64_t m = 5;
    const int64_t k = 64;
    const int64_t n = 4096;
    std::vector<float> lhs(static_cast<size_t>(m * k));
    std::vector<float> rhs(static_cast<size_t>(k * n));
    for (size_t index = 0; index < lhs.size(); ++index) {
        lhs[index] = static_cast<float>((index % 17) - 8) * 0.125f;
    }
    for (size_t index = 0; index < rhs.size(); ++index) {
        rhs[index] = static_cast<float>((index % 23) - 11) * 0.0625f;
    }
    std::vector<float> out(static_cast<size_t>(m * n), 0.0f);
    ASSERT_TRUE(Fp32LinearUsesBlockedKernel(m, k, n));
    ASSERT_EQ(ComputeLinearRowMajorX86Fp32(lhs.data(), rhs.data(), nullptr, m, k, n, LinearBiasType::kNone,
                                            out.data()),
              0);

    const auto expected = ReferenceMatMul(lhs, rhs, m, k, n);
    for (size_t index = 0; index < out.size(); ++index) {
        EXPECT_FLOAT_EQ(out[index], expected[index]);
    }
}

}  // namespace
