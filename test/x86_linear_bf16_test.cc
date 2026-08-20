#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "core/kernel.h"
#include "core/operator.h"
#include "core/tensor.h"
#include "src/kernel/gemm.h"
#include "src/kernel/matmul.h"
#include "src/operator/matmul_op.h"
#include "src/operator/params.h"
#include "src/kernel/x86/linear.h"
#include "util/bf16.h"
#include "util/threading.h"

namespace {

using feather::BFloat16ToFloat;
using feather::FloatToBFloat16;
using feather::kernel::x86::ComputeLinearRowMajorX86Bf16;
using feather::kernel::x86::ComputeLinearRowMajorX86Bf16PackedRhs;
using feather::kernel::x86::ComputeLinearRowMajorX86Bf16PackedTransposedRhs;
using feather::kernel::x86::ComputeLinearRowMajorX86Bf16TransposedRhs;
using feather::kernel::x86::PackedBf16Rhs;
using feather::kernel::x86::PackedBf16TransposedRhs;
using feather::kernel::x86::LinearBiasType;
using feather::kernel::x86::Bf16LinearWorkerCount;

class ScopedEnvironmentVariable {
   public:
    ScopedEnvironmentVariable(const char* name, const char* value)
        : name_(name), previous_(std::getenv(name) == nullptr ? "" : std::getenv(name)),
          existed_(std::getenv(name) != nullptr) {
        setenv(name, value, 1);
    }

    ~ScopedEnvironmentVariable() {
        if (!existed_) {
            unsetenv(name_.c_str());
        } else {
            setenv(name_.c_str(), previous_.c_str(), 1);
        }
    }

   private:
    std::string name_;
    std::string previous_;
    bool existed_;
};

class DestructionTrackingKernel final : public feather::KernelBase {
   public:
    explicit DestructionTrackingKernel(bool* destroyed) : destroyed_(destroyed) {}
    ~DestructionTrackingKernel() override {
        if (destroyed_ != nullptr) {
            *destroyed_ = true;
        }
    }
    int32_t compute() override { return 0; }

   private:
    bool* destroyed_;
};

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

std::vector<uint16_t> ToBf16(const std::vector<float>& values) {
    std::vector<uint16_t> out(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        out[i] = FloatToBFloat16(values[i]);
    }
    return out;
}

TEST(x86_linear_bf16_test, MatMulHandlesVectorBlocksAndColumnTail) {
    const int64_t m = 2;
    const int64_t k = 5;
    const int64_t n = 25;
    std::vector<float> lhs(static_cast<size_t>(m * k));
    std::vector<float> rhs(static_cast<size_t>(k * n));
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t t = 0; t < k; ++t) {
            lhs[static_cast<size_t>(i * k + t)] = 0.25f * static_cast<float>(i + t + 1);
        }
    }
    for (int64_t t = 0; t < k; ++t) {
        for (int64_t j = 0; j < n; ++j) {
            rhs[static_cast<size_t>(t * n + j)] = -0.5f + 0.125f * static_cast<float>((t * 3 + j) % 17);
        }
    }

    const auto lhs_bf16 = ToBf16(lhs);
    const auto rhs_bf16 = ToBf16(rhs);
    std::vector<uint16_t> out(static_cast<size_t>(m * n), 0);
    ASSERT_EQ(ComputeLinearRowMajorX86Bf16(lhs_bf16.data(), rhs_bf16.data(), nullptr, m, k, n,
                                           LinearBiasType::kNone, out.data()),
              0);

    const auto expected = ReferenceMatMul(lhs, rhs, m, k, n);
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(BFloat16ToFloat(out[i]), expected[i], 8e-2f);
    }
}

TEST(x86_linear_bf16_test, WorkerPolicyUsesAllDefaultThreadsForLogitsAndKeepsSmallGemmSerial) {
    EXPECT_EQ(Bf16LinearWorkerCount(1, 1024, 248320), feather::DefaultThreadCount());
    EXPECT_EQ(Bf16LinearWorkerCount(1, 1024, 16), 1U);

    ScopedEnvironmentVariable worker_limit("FEATHER_X86_BF16_THREADS", "4");
    EXPECT_EQ(Bf16LinearWorkerCount(1, 1024, 248320), 4U);
}

TEST(x86_linear_bf16_test, Bf16OutputColumnParallelismBuildsWithOpenMP) {
#ifdef FEATHER_WITH_OPENMP
    SUCCEED();
#else
    GTEST_SKIP() << "OpenMP is disabled for this build";
#endif
}

TEST(x86_linear_bf16_test, ParallelWideBf16OutputMatchesSingleThreadBitForBit) {
    constexpr int64_t m = 1;
    constexpr int64_t k = 1024;
    constexpr int64_t n = 1024;
    std::vector<float> lhs(static_cast<size_t>(m * k));
    std::vector<float> rhs(static_cast<size_t>(k * n));
    for (int64_t t = 0; t < k; ++t) {
        lhs[static_cast<size_t>(t)] = 0.015625f * static_cast<float>((t % 37) - 18);
    }
    for (int64_t t = 0; t < k; ++t) {
        for (int64_t col = 0; col < n; ++col) {
            rhs[static_cast<size_t>(t * n + col)] =
                0.0078125f * static_cast<float>(((t * 13 + col * 7) % 53) - 26);
        }
    }

    const auto lhs_bf16 = ToBf16(lhs);
    const auto rhs_bf16 = ToBf16(rhs);
    std::vector<uint16_t> single_thread_out(static_cast<size_t>(n));
    std::vector<uint16_t> parallel_out(static_cast<size_t>(n));
    {
        ScopedEnvironmentVariable worker_limit("FEATHER_X86_BF16_THREADS", "1");
        ASSERT_EQ(ComputeLinearRowMajorX86Bf16(lhs_bf16.data(), rhs_bf16.data(), nullptr, m, k, n,
                                               LinearBiasType::kNone, single_thread_out.data()),
                  0);
    }
    {
        ScopedEnvironmentVariable worker_limit("FEATHER_X86_BF16_THREADS", "4");
        ASSERT_EQ(ComputeLinearRowMajorX86Bf16(lhs_bf16.data(), rhs_bf16.data(), nullptr, m, k, n,
                                               LinearBiasType::kNone, parallel_out.data()),
                  0);
    }

    EXPECT_EQ(parallel_out, single_thread_out);
}

TEST(x86_linear_bf16_test, MatMulSingleRowWideOutputPreservesBiasAndTail) {
    const int64_t m = 1;
    const int64_t k = 9;
    const int64_t n = 40;
    std::vector<float> lhs(static_cast<size_t>(m * k));
    std::vector<float> rhs(static_cast<size_t>(k * n));
    std::vector<float> vector_bias(static_cast<size_t>(n));
    std::vector<float> matrix_bias(static_cast<size_t>(m * n));
    for (int64_t t = 0; t < k; ++t) lhs[static_cast<size_t>(t)] = 0.1f * static_cast<float>(t - 3);
    for (int64_t t = 0; t < k; ++t) {
        for (int64_t j = 0; j < n; ++j) {
            rhs[static_cast<size_t>(t * n + j)] = 0.03f * static_cast<float>((t * 11 + j * 7) % 19 - 9);
        }
    }
    for (int64_t j = 0; j < n; ++j) {
        vector_bias[static_cast<size_t>(j)] = 0.05f * static_cast<float>(j - 20);
        matrix_bias[static_cast<size_t>(j)] = -0.025f * static_cast<float>(j - 11);
    }

    const auto lhs_bf16 = ToBf16(lhs);
    const auto rhs_bf16 = ToBf16(rhs);
    const auto vector_bias_bf16 = ToBf16(vector_bias);
    const auto matrix_bias_bf16 = ToBf16(matrix_bias);
    const auto product = ReferenceMatMul(lhs, rhs, m, k, n);

    std::vector<uint16_t> vector_out(static_cast<size_t>(n));
    std::vector<uint16_t> matrix_out(static_cast<size_t>(n));
    ASSERT_EQ(ComputeLinearRowMajorX86Bf16(lhs_bf16.data(), rhs_bf16.data(), vector_bias_bf16.data(), m, k, n,
                                           LinearBiasType::kVector, vector_out.data()),
              0);
    ASSERT_EQ(ComputeLinearRowMajorX86Bf16(lhs_bf16.data(), rhs_bf16.data(), matrix_bias_bf16.data(), m, k, n,
                                           LinearBiasType::kMatrix, matrix_out.data()),
              0);
    for (int64_t j = 0; j < n; ++j) {
        EXPECT_NEAR(BFloat16ToFloat(vector_out[static_cast<size_t>(j)]),
                    product[static_cast<size_t>(j)] + vector_bias[static_cast<size_t>(j)], 0.04f);
        EXPECT_NEAR(BFloat16ToFloat(matrix_out[static_cast<size_t>(j)]),
                    product[static_cast<size_t>(j)] + matrix_bias[static_cast<size_t>(j)], 0.04f);
    }
}

TEST(x86_linear_bf16_test, GemmSupportsVectorAndMatrixBias) {
    const int64_t m = 2;
    const int64_t k = 3;
    const int64_t n = 9;
    const std::vector<float> lhs = {1, 2, 3, 4, 5, 6};
    const std::vector<float> rhs = {1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                    3, 4, 5, 6, 7, 8, 9, 10, 11};
    const std::vector<float> vector_bias = {1, 0, -1, 2, -2, 3, -3, 4, -4};
    const std::vector<float> matrix_bias = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    const auto product = ReferenceMatMul(lhs, rhs, m, k, n);
    const auto lhs_bf16 = ToBf16(lhs);
    const auto rhs_bf16 = ToBf16(rhs);
    const auto vector_bias_bf16 = ToBf16(vector_bias);
    const auto matrix_bias_bf16 = ToBf16(matrix_bias);

    std::vector<uint16_t> vector_out(static_cast<size_t>(m * n), 0);
    ASSERT_EQ(ComputeLinearRowMajorX86Bf16(lhs_bf16.data(), rhs_bf16.data(), vector_bias_bf16.data(), m, k, n,
                                           LinearBiasType::kVector, vector_out.data()),
              0);
    std::vector<uint16_t> matrix_out(static_cast<size_t>(m * n), 0);
    ASSERT_EQ(ComputeLinearRowMajorX86Bf16(lhs_bf16.data(), rhs_bf16.data(), matrix_bias_bf16.data(), m, k, n,
                                           LinearBiasType::kMatrix, matrix_out.data()),
              0);

    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            const size_t offset = static_cast<size_t>(i * n + j);
            EXPECT_NEAR(BFloat16ToFloat(vector_out[offset]), product[offset] + vector_bias[static_cast<size_t>(j)],
                        1e-1f);
            EXPECT_NEAR(BFloat16ToFloat(matrix_out[offset]), product[offset] + matrix_bias[offset], 1e-1f);
        }
    }
}

TEST(x86_linear_bf16_test, RejectsInvalidArguments) {
    uint16_t out = 0;
    EXPECT_EQ(ComputeLinearRowMajorX86Bf16(nullptr, nullptr, nullptr, 1, 1, 1, LinearBiasType::kNone, &out), -1);
}

TEST(x86_linear_bf16_test, WideSingleRowMatchesReference) {
    const int64_t m = 1;
    const int64_t k = 257;
    const int64_t n = 4097;
    std::vector<float> lhs(static_cast<size_t>(k));
    std::vector<float> rhs(static_cast<size_t>(k * n));
    for (int64_t t = 0; t < k; ++t) {
        lhs[static_cast<size_t>(t)] = 0.03125f * static_cast<float>(t % 17 - 8);
        for (int64_t col = 0; col < n; ++col) {
            rhs[static_cast<size_t>(t * n + col)] =
                0.015625f * static_cast<float>((t * 13 + col * 5) % 31 - 15);
        }
    }

    const auto lhs_bf16 = ToBf16(lhs);
    const auto rhs_bf16 = ToBf16(rhs);
    std::vector<uint16_t> out(static_cast<size_t>(n));
    ASSERT_EQ(ComputeLinearRowMajorX86Bf16(lhs_bf16.data(), rhs_bf16.data(), nullptr, m, k, n,
                                           LinearBiasType::kNone, out.data()),
              0);

    const auto expected = ReferenceMatMul(lhs, rhs, m, k, n);
    for (int64_t col = 0; col < n; ++col) {
        EXPECT_NEAR(BFloat16ToFloat(out[static_cast<size_t>(col)]), expected[static_cast<size_t>(col)], 0.08f);
    }
}

TEST(x86_linear_bf16_test, PackedRhsMatchesWideProjectionWithBias) {
    const int64_t m = 1;
    const int64_t k = 13;
    const int64_t n = 64;
    std::vector<float> lhs(static_cast<size_t>(m * k));
    std::vector<float> rhs(static_cast<size_t>(k * n));
    std::vector<float> bias(static_cast<size_t>(n));
    for (int64_t t = 0; t < k; ++t) {
        lhs[static_cast<size_t>(t)] = 0.0625f * static_cast<float>(t - 6);
        for (int64_t col = 0; col < n; ++col) {
            rhs[static_cast<size_t>(t * n + col)] =
                0.03125f * static_cast<float>((t * 7 + col * 11) % 29 - 14);
        }
    }
    for (int64_t col = 0; col < n; ++col) bias[static_cast<size_t>(col)] = 0.015625f * static_cast<float>(col - 32);

    const auto lhs_bf16 = ToBf16(lhs);
    const auto rhs_bf16 = ToBf16(rhs);
    const auto bias_bf16 = ToBf16(bias);
    std::vector<uint16_t> unpacked_out(static_cast<size_t>(n));
    std::vector<uint16_t> packed_out(static_cast<size_t>(n));
    ASSERT_EQ(ComputeLinearRowMajorX86Bf16(lhs_bf16.data(), rhs_bf16.data(), bias_bf16.data(), m, k, n,
                                           LinearBiasType::kVector, unpacked_out.data()),
              0);

    PackedBf16Rhs packed_rhs;
    ASSERT_TRUE(packed_rhs.Pack(rhs_bf16.data(), k, n));
    ASSERT_TRUE(packed_rhs.Matches(rhs_bf16.data(), k, n));
    ASSERT_EQ(ComputeLinearRowMajorX86Bf16PackedRhs(lhs_bf16.data(), rhs_bf16.data(), packed_rhs,
                                                    bias_bf16.data(), m, k, n, LinearBiasType::kVector,
                                                    packed_out.data()),
              0);

    EXPECT_EQ(packed_out, unpacked_out);
}

TEST(x86_linear_bf16_test, PackedRhsSupportsUnalignedOutputWidth) {
    constexpr int64_t k = 2;
    constexpr int64_t n = 40;
    std::vector<uint16_t> rhs(static_cast<size_t>(k * n), FloatToBFloat16(1.0f));
    PackedBf16Rhs packed_rhs;

    EXPECT_TRUE(packed_rhs.Pack(rhs.data(), k, n));
    EXPECT_TRUE(packed_rhs.Matches(rhs.data(), k, n));
}

TEST(x86_linear_bf16_test, PackedRhsUsesSixtyFourColumnBlocksForTailOutputs) {
    constexpr int64_t k = 3;
    constexpr int64_t n = 65;
    std::vector<uint16_t> rhs(static_cast<size_t>(k * n), FloatToBFloat16(1.0f));
    PackedBf16Rhs packed_rhs;

    ASSERT_TRUE(packed_rhs.Pack(rhs.data(), k, n));
    EXPECT_EQ(packed_rhs.size(), static_cast<size_t>(2 * 64 * k));

    const std::vector<uint16_t> lhs = ToBf16({0.5f, -1.0f, 2.0f});
    std::vector<uint16_t> out(static_cast<size_t>(n));
    ASSERT_EQ(ComputeLinearRowMajorX86Bf16PackedRhs(lhs.data(), rhs.data(), packed_rhs, nullptr, 1, k, n,
                                                    LinearBiasType::kNone, out.data()),
              0);
    for (const uint16_t value : out) {
        EXPECT_NEAR(BFloat16ToFloat(value), 1.5f, 0.02f);
    }
}

TEST(x86_linear_bf16_test, PackedTransposedRhsMatchesWideProjectionWithBiasAndScaling) {
    constexpr int64_t m = 1;
    constexpr int64_t k = 13;
    constexpr int64_t n = 65;
    std::vector<float> lhs(static_cast<size_t>(m * k));
    std::vector<float> rhs_transposed(static_cast<size_t>(n * k));
    std::vector<float> bias(static_cast<size_t>(n));
    for (int64_t t = 0; t < k; ++t) {
        lhs[static_cast<size_t>(t)] = 0.0625f * static_cast<float>(t - 6);
    }
    for (int64_t col = 0; col < n; ++col) {
        bias[static_cast<size_t>(col)] = 0.015625f * static_cast<float>(col - 32);
        for (int64_t t = 0; t < k; ++t) {
            rhs_transposed[static_cast<size_t>(col * k + t)] =
                0.03125f * static_cast<float>((col * 7 + t * 11) % 29 - 14);
        }
    }

    const auto lhs_bf16 = ToBf16(lhs);
    const auto rhs_bf16 = ToBf16(rhs_transposed);
    const auto bias_bf16 = ToBf16(bias);
    std::vector<uint16_t> reference_out(static_cast<size_t>(n));
    std::vector<uint16_t> packed_out(static_cast<size_t>(n));
    ASSERT_EQ(ComputeLinearRowMajorX86Bf16TransposedRhs(lhs_bf16.data(), rhs_bf16.data(), bias_bf16.data(), m, k,
                                                         n, LinearBiasType::kVector, 0.75f, 1.25f,
                                                         reference_out.data()),
              0);

    PackedBf16TransposedRhs packed_rhs;
    ASSERT_TRUE(packed_rhs.Pack(rhs_bf16.data(), k, n));
    ASSERT_TRUE(packed_rhs.Matches(rhs_bf16.data(), k, n));
    ASSERT_EQ(ComputeLinearRowMajorX86Bf16PackedTransposedRhs(
                  lhs_bf16.data(), rhs_bf16.data(), packed_rhs, bias_bf16.data(), m, k, n,
                  LinearBiasType::kVector, 0.75f, 1.25f, packed_out.data()),
              0);

    for (int64_t col = 0; col < n; ++col) {
        EXPECT_NEAR(BFloat16ToFloat(packed_out[static_cast<size_t>(col)]),
                    BFloat16ToFloat(reference_out[static_cast<size_t>(col)]), 0.02f);
    }
}

TEST(x86_linear_bf16_test, TransposedRhsUsesVectorKernelAndScaling) {
    const int64_t m = 2;
    const int64_t k = 5;
    const int64_t n = 7;
    std::vector<float> lhs(static_cast<size_t>(m * k));
    std::vector<float> rhs_transposed(static_cast<size_t>(n * k));
    std::vector<float> bias(static_cast<size_t>(n));
    for (int64_t i = 0; i < m * k; ++i) lhs[static_cast<size_t>(i)] = 0.2f * static_cast<float>(i + 1);
    for (int64_t i = 0; i < n * k; ++i) rhs_transposed[static_cast<size_t>(i)] = -0.15f + 0.07f * static_cast<float>(i);
    for (int64_t i = 0; i < n; ++i) bias[static_cast<size_t>(i)] = 0.3f * static_cast<float>(i - 2);

    const auto lhs_bf16 = ToBf16(lhs);
    const auto rhs_bf16 = ToBf16(rhs_transposed);
    const auto bias_bf16 = ToBf16(bias);
    std::vector<uint16_t> out(static_cast<size_t>(m * n), 0);
    ASSERT_EQ(ComputeLinearRowMajorX86Bf16TransposedRhs(lhs_bf16.data(), rhs_bf16.data(), bias_bf16.data(), m, k, n,
                                                       LinearBiasType::kVector, 0.75f, 1.25f, out.data()),
              0);

    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            float expected = 1.25f * bias[static_cast<size_t>(j)];
            for (int64_t t = 0; t < k; ++t) {
                expected += 0.75f * lhs[static_cast<size_t>(i * k + t)] *
                            rhs_transposed[static_cast<size_t>(j * k + t)];
            }
            EXPECT_NEAR(BFloat16ToFloat(out[static_cast<size_t>(i * n + j)]), expected, 0.12f);
        }
    }
}

TEST(x86_linear_bf16_test, X86Bf16GemmPreservesDtypeForTransposedB) {
    auto lhs = std::make_shared<feather::Tensor>();
    lhs->Assign<feather::BFloat16>(
        {feather::BFloat16{FloatToBFloat16(1.0f)}, feather::BFloat16{FloatToBFloat16(2.0f)},
         feather::BFloat16{FloatToBFloat16(3.0f)}, feather::BFloat16{FloatToBFloat16(4.0f)}},
        {2, 2});
    auto rhs = std::make_shared<feather::Tensor>();
    rhs->Assign<feather::BFloat16>(
        {feather::BFloat16{FloatToBFloat16(1.0f)}, feather::BFloat16{FloatToBFloat16(3.0f)},
         feather::BFloat16{FloatToBFloat16(2.0f)}, feather::BFloat16{FloatToBFloat16(4.0f)}},
        {2, 2});
    auto out = std::make_shared<feather::Tensor>(std::vector<int64_t>{2, 2});

    feather::operators::GemmParam param{};
    param.a = lhs;
    param.b = rhs;
    param.out = out;
    param.trans_b = true;
    auto kernel = feather::KernelDispatcher::instance().create(feather::DeviceType::X86, feather::DataType::BF16,
                                                               "Gemm");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);
    EXPECT_EQ(out->data_type(), feather::DataType::BF16);
    ASSERT_EQ(out->numel(), 4);
    EXPECT_NEAR(out->data<feather::BFloat16>()[0].bits == 0 ? 0.0f
                                                            : BFloat16ToFloat(out->data<feather::BFloat16>()[0].bits),
                7.0f, 0.08f);
    EXPECT_NEAR(BFloat16ToFloat(out->data<feather::BFloat16>()[3].bits), 22.0f, 0.12f);
}

TEST(x86_linear_bf16_test, X86Bf16GemmCachesImmutableWideTransposedRhs) {
    constexpr int64_t m = 1;
    constexpr int64_t k = 1024;
    constexpr int64_t n = 1024;

    std::vector<feather::BFloat16> lhs_values(static_cast<size_t>(m * k));
    std::vector<feather::BFloat16> rhs_values(static_cast<size_t>(n * k));
    std::vector<feather::BFloat16> bias_values(static_cast<size_t>(n));
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t col = 0; col < k; ++col) {
            lhs_values[static_cast<size_t>(row * k + col)] = {
                FloatToBFloat16(0.0078125f * static_cast<float>(col % 17 - 8))};
        }
    }
    for (int64_t col = 0; col < n; ++col) {
        bias_values[static_cast<size_t>(col)] = {FloatToBFloat16(0.015625f * static_cast<float>(col % 13 - 6))};
        for (int64_t row = 0; row < k; ++row) {
            rhs_values[static_cast<size_t>(col * k + row)] = {
                FloatToBFloat16(0.00390625f * static_cast<float>((col * 3 + row * 5) % 31 - 15))};
        }
    }

    auto lhs = std::make_shared<feather::Tensor>();
    lhs->Assign<feather::BFloat16>(lhs_values, {m, k});
    auto rhs = std::make_shared<feather::Tensor>();
    rhs->Assign<feather::BFloat16>(rhs_values, {n, k});
    rhs->set_immutable(true);
    auto bias = std::make_shared<feather::Tensor>();
    bias->Assign<feather::BFloat16>(bias_values, {n});
    auto out = std::make_shared<feather::Tensor>(std::vector<int64_t>{m, n});

    feather::operators::GemmParam param{};
    param.a = lhs;
    param.b = rhs;
    param.bias = bias;
    param.out = out;
    param.trans_b = true;
    param.alpha = 0.75f;
    param.beta = 1.25f;
    auto kernel = feather::KernelDispatcher::instance().create(feather::DeviceType::X86, feather::DataType::BF16,
                                                               "Gemm");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);

    ASSERT_EQ(kernel->compute(), 0);
    std::vector<uint16_t> first_output(static_cast<size_t>(n));
    for (int64_t col = 0; col < n; ++col) {
        first_output[static_cast<size_t>(col)] = out->data<feather::BFloat16>()[col].bits;
    }
    ASSERT_EQ(kernel->compute(), 0);
    for (int64_t col = 0; col < n; ++col) {
        EXPECT_EQ(out->data<feather::BFloat16>()[col].bits, first_output[static_cast<size_t>(col)]);
    }

    for (const int64_t col : {int64_t{0}, n / 2, n - 1}) {
        float expected = param.beta * BFloat16ToFloat(bias_values[static_cast<size_t>(col)].bits);
        for (int64_t row = 0; row < k; ++row) {
            expected += param.alpha * BFloat16ToFloat(lhs_values[static_cast<size_t>(row)].bits) *
                        BFloat16ToFloat(rhs_values[static_cast<size_t>(col * k + row)].bits);
        }
        EXPECT_NEAR(BFloat16ToFloat(first_output[static_cast<size_t>(col)]), expected, 0.08f);
    }
}

TEST(x86_linear_bf16_test, X86Bf16GemmPreparesImmutableLogitsBeforeCompute) {
    constexpr int64_t k = 1;
    constexpr int64_t n = 32768;
    auto lhs = std::make_shared<feather::Tensor>();
    lhs->Assign<feather::BFloat16>({feather::BFloat16{FloatToBFloat16(2.0f)}}, {1, k});
    auto rhs = std::make_shared<feather::Tensor>();
    rhs->Assign<feather::BFloat16>(std::vector<feather::BFloat16>(static_cast<size_t>(n),
                                                                  feather::BFloat16{FloatToBFloat16(3.0f)}),
                                   {n, k});
    rhs->set_immutable(true);
    auto out = std::make_shared<feather::Tensor>(std::vector<int64_t>{1, n});

    feather::operators::GemmParam param{};
    param.a = lhs;
    param.b = rhs;
    param.out = out;
    param.trans_b = true;
    auto kernel = feather::KernelDispatcher::instance().create(feather::DeviceType::X86, feather::DataType::BF16,
                                                               "Gemm");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->Prepare(), 0);
    ASSERT_EQ(kernel->compute(), 0);
    EXPECT_EQ(out->data_type(), feather::DataType::BF16);
    EXPECT_NEAR(BFloat16ToFloat(out->data<feather::BFloat16>()[0].bits), 6.0f, 0.05f);
    EXPECT_NEAR(BFloat16ToFloat(out->data<feather::BFloat16>()[n - 1].bits), 6.0f, 0.05f);
}

TEST(x86_linear_bf16_test, X86Bf16GemmRejectsZeroKBeforeShapeArithmetic) {
    auto lhs = std::make_shared<feather::Tensor>();
    lhs->Resize({1, 0});
    lhs->set_data_type(feather::DataType::BF16);
    auto rhs = std::make_shared<feather::Tensor>();
    rhs->Resize({32768, 0});
    rhs->set_data_type(feather::DataType::BF16);
    rhs->set_immutable(true);
    auto out = std::make_shared<feather::Tensor>();
    out->Resize({1, 32768});
    out->set_data_type(feather::DataType::BF16);

    feather::operators::GemmParam param{};
    param.a = lhs;
    param.b = rhs;
    param.out = out;
    param.trans_b = true;
    auto kernel = feather::KernelDispatcher::instance().create(feather::DeviceType::X86, feather::DataType::BF16,
                                                               "Gemm");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    EXPECT_EQ(kernel->Prepare(), 0);
    EXPECT_EQ(kernel->compute(), -1);
}

TEST(x86_linear_bf16_test, X86Bf16MatMulRejectsUninitializedRhsBeforePacking) {
    auto lhs = std::make_shared<feather::Tensor>();
    lhs->Assign<feather::BFloat16>({feather::BFloat16{FloatToBFloat16(1.0f)}}, {1, 1});
    auto rhs = std::make_shared<feather::Tensor>();
    rhs->Resize({1, 32768});
    rhs->set_data_type(feather::DataType::BF16);
    rhs->set_immutable(true);
    auto out = std::make_shared<feather::Tensor>(std::vector<int64_t>{1, 32768});

    feather::operators::MatMulParam param{};
    param.a = lhs;
    param.b = rhs;
    param.out = out;
    auto kernel = feather::KernelDispatcher::instance().create(feather::DeviceType::X86, feather::DataType::BF16,
                                                               "MatMul");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    EXPECT_EQ(kernel->compute(), -1);
}

TEST(x86_linear_bf16_test, X86Bf16MatMulValidatesBatchBroadcastBeforeSingletonShortcut) {
    auto lhs = std::make_shared<feather::Tensor>();
    lhs->Assign<feather::BFloat16>(std::vector<feather::BFloat16>(6, feather::BFloat16{FloatToBFloat16(1.0f)}),
                                   {2, 1, 3});
    auto rhs = std::make_shared<feather::Tensor>();
    rhs->Assign<feather::BFloat16>(std::vector<feather::BFloat16>(24, feather::BFloat16{FloatToBFloat16(1.0f)}),
                                   {2, 3, 4});
    auto out = std::make_shared<feather::Tensor>(std::vector<int64_t>{1, 1, 4});

    feather::operators::MatMulParam param{};
    param.a = lhs;
    param.b = rhs;
    param.out = out;
    auto kernel = feather::KernelDispatcher::instance().create(feather::DeviceType::X86, feather::DataType::BF16,
                                                               "MatMul");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    EXPECT_EQ(kernel->compute(), -1);
}

TEST(x86_linear_bf16_test, X86Bf16GemmInvalidatesPackedRhsAfterTensorMutation) {
    constexpr int64_t k = 1;
    constexpr int64_t n = 32768;
    auto lhs = std::make_shared<feather::Tensor>();
    lhs->Assign<feather::BFloat16>({feather::BFloat16{FloatToBFloat16(2.0f)}}, {1, k});
    auto rhs = std::make_shared<feather::Tensor>();
    rhs->Assign<feather::BFloat16>(std::vector<feather::BFloat16>(static_cast<size_t>(n),
                                                                  feather::BFloat16{FloatToBFloat16(3.0f)}),
                                   {n, k});
    rhs->set_immutable(true);
    auto out = std::make_shared<feather::Tensor>(std::vector<int64_t>{1, n});

    feather::operators::GemmParam param{};
    param.a = lhs;
    param.b = rhs;
    param.out = out;
    param.trans_b = true;
    auto kernel = feather::KernelDispatcher::instance().create(feather::DeviceType::X86, feather::DataType::BF16,
                                                               "Gemm");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->Prepare(), 0);
    ASSERT_EQ(kernel->compute(), 0);
    EXPECT_NEAR(BFloat16ToFloat(out->data<feather::BFloat16>()[0].bits), 6.0f, 0.05f);

    rhs->mutable_data<feather::BFloat16>()[0].bits = FloatToBFloat16(5.0f);
    ASSERT_EQ(kernel->compute(), 0);
    EXPECT_NEAR(BFloat16ToFloat(out->data<feather::BFloat16>()[0].bits), 10.0f, 0.05f);
}

TEST(x86_linear_bf16_test, KernelBaseDestroysDerivedKernelThroughBasePointer) {
    bool destroyed = false;
    {
        std::unique_ptr<feather::KernelBase> kernel = std::make_unique<DestructionTrackingKernel>(&destroyed);
    }
    EXPECT_TRUE(destroyed);
}

TEST(x86_linear_bf16_test, X86Bf16BatchedMatMulBroadcastsBatchDimensions) {
    auto lhs = std::make_shared<feather::Tensor>();
    std::vector<feather::BFloat16> lhs_values;
    for (float value : {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}) {
        lhs_values.push_back({FloatToBFloat16(value)});
    }
    lhs->Assign<feather::BFloat16>(lhs_values, {2, 2, 2});
    auto rhs = std::make_shared<feather::Tensor>();
    std::vector<feather::BFloat16> rhs_values;
    for (float value : {1.0f, 2.0f, 3.0f, 4.0f}) rhs_values.push_back({FloatToBFloat16(value)});
    rhs->Assign<feather::BFloat16>(rhs_values, {1, 2, 2});
    auto out = std::make_shared<feather::Tensor>(std::vector<int64_t>{2, 2, 2});

    feather::operators::MatMulParam param{};
    param.a = lhs;
    param.b = rhs;
    param.out = out;
    auto kernel = feather::KernelDispatcher::instance().create(feather::DeviceType::X86, feather::DataType::BF16,
                                                               "MatMul");
    ASSERT_NE(kernel, nullptr);
    ASSERT_NE((dynamic_cast<feather::kernel::MatMulKernel<feather::DeviceType::X86, feather::DataType::BF16>*>(
                  kernel.get())),
              nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);
    EXPECT_EQ(out->data_type(), feather::DataType::BF16);
    const std::vector<float> expected = {7.0f, 10.0f, 15.0f, 22.0f, 23.0f, 34.0f, 31.0f, 46.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(BFloat16ToFloat(out->data<feather::BFloat16>()[i].bits), expected[i], 0.15f);
    }
}

}  // namespace
