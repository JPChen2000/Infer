#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/kernel.h"
#include "core/operator.h"
#include "core/tensor.h"
#include "src/kernel/gemm.h"
#include "src/operator/params.h"
#include "src/operator/gemm_op.h"
#include "util/bf16.h"
#include "util/fp16.h"

using feather::KernelDispatcher;
using feather::OpBase;
using feather::Tensor;
using feather::DeviceType;
using feather::DataType;
using feather::operators::GemmParam;

TEST(gemm_op_test, X86Bf16AcceptsSingletonLeadingBroadcastBias) {
    auto lhs = std::make_shared<Tensor>();
    lhs->Assign<feather::BFloat16>(
        {feather::BFloat16{feather::FloatToBFloat16(1.0f)}, feather::BFloat16{feather::FloatToBFloat16(2.0f)},
         feather::BFloat16{feather::FloatToBFloat16(3.0f)}},
        {1, 1, 3});

    auto rhs = std::make_shared<Tensor>();
    rhs->Assign<feather::BFloat16>(
        {feather::BFloat16{feather::FloatToBFloat16(1.0f)}, feather::BFloat16{feather::FloatToBFloat16(2.0f)},
         feather::BFloat16{feather::FloatToBFloat16(3.0f)}, feather::BFloat16{feather::FloatToBFloat16(4.0f)},
         feather::BFloat16{feather::FloatToBFloat16(5.0f)}, feather::BFloat16{feather::FloatToBFloat16(6.0f)}},
        {3, 2});

    auto bias = std::make_shared<Tensor>();
    bias->Assign<feather::BFloat16>(
        {feather::BFloat16{feather::FloatToBFloat16(0.5f)}, feather::BFloat16{feather::FloatToBFloat16(-1.0f)}},
        {1, 1, 2});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 2});
    GemmParam param;
    param.a = lhs;
    param.b = rhs;
    param.bias = bias;
    param.out = out;
    feather::operators::GemmOp op("broadcast_bias", param);
    ASSERT_EQ(op.CheckShape(), 0);
    ASSERT_EQ(op.InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::BF16, "Gemm");
    ASSERT_NE(kernel, nullptr);
    op.AttachKernel(std::move(kernel));
    ASSERT_EQ(op.Run(), 0);

    const auto* values = out->data<feather::BFloat16>();
    EXPECT_NEAR(feather::BFloat16ToFloat(values[0].bits), 22.5f, 0.2f);
    EXPECT_NEAR(feather::BFloat16ToFloat(values[1].bits), 27.0f, 0.2f);
}

TEST(gemm_op_test, GemmRunsOnX86) {
    auto lhs = std::make_shared<Tensor>();
    lhs->Assign<float>({1, 2, 3, 4, 5, 6}, {2, 3});

    auto rhs = std::make_shared<Tensor>();
    rhs->Assign<float>({1, 2, 3, 4, 5, 6}, {3, 2});

    auto bias = std::make_shared<Tensor>();
    bias->Assign<float>({1, 1, 2, 2}, {2, 2});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});

    GemmParam param;
    param.a = lhs;
    param.b = rhs;
    param.bias = bias;
    param.out = out;

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::GemmOp>("gemm0", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "Gemm");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    std::vector<float> expected = {
        23, 29,
        51, 66,
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(out->data<float>()[i], expected[i]);
    }
}

TEST(gemm_op_test, GemmReusesStaticShapeInferenceAcrossRuns) {
    auto lhs = std::make_shared<Tensor>();
    lhs->Assign<float>({1, 2, 3, 4}, {2, 2});
    auto rhs = std::make_shared<Tensor>();
    rhs->Assign<float>({1, 2, 3, 4}, {2, 2});
    auto bias = std::make_shared<Tensor>();
    bias->Assign<float>({1, 1}, {1, 2});
    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});

    GemmParam param{};
    param.a = lhs;
    param.b = rhs;
    param.bias = bias;
    param.out = out;
    feather::operators::GemmOp op("static_shape_gemm", param);
    ASSERT_EQ(op.InferOutputShapes(), 0);
    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "Gemm");
    ASSERT_NE(kernel, nullptr);
    op.AttachKernel(std::move(kernel));

    ASSERT_EQ(op.Run(), 0);
    ASSERT_EQ(op.Run(), 0);
    EXPECT_EQ(op.shape_inference_count(), 1);
}

TEST(gemm_op_test, CommonFp16KernelRunsCorrectly) {
    auto lhs = std::make_shared<Tensor>();
    lhs->Assign<uint16_t>({feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f),
                           feather::FloatToHalf(4.0f), feather::FloatToHalf(5.0f), feather::FloatToHalf(6.0f)},
                          {2, 3});

    auto rhs = std::make_shared<Tensor>();
    rhs->Assign<uint16_t>({feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f),
                           feather::FloatToHalf(4.0f), feather::FloatToHalf(5.0f), feather::FloatToHalf(6.0f)},
                          {3, 2});

    auto bias = std::make_shared<Tensor>();
    bias->Assign<uint16_t>({feather::FloatToHalf(1.0f), feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f),
                            feather::FloatToHalf(2.0f)},
                           {2, 2});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});
    out->mutable_data<uint16_t>();

    GemmParam param{};
    param.a = lhs;
    param.b = rhs;
    param.bias = bias;
    param.out = out;

    auto kernel = KernelDispatcher::instance().create(DeviceType::COMMON, DataType::FP16, "Gemm");
    ASSERT_NE(kernel, nullptr);
    auto* common_kernel = dynamic_cast<feather::kernel::GemmKernel<DeviceType::COMMON, DataType::FP16>*>(kernel.get());
    ASSERT_NE(common_kernel, nullptr);

    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    const std::vector<float> expected = {23.0f, 29.0f, 51.0f, 66.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(out->data<uint16_t>()[i]), expected[i], 2e-2f);
    }
}

TEST(gemm_op_test, X86Fp16KernelIsRegisteredAndRuns) {
    auto lhs = std::make_shared<Tensor>();
    lhs->Assign<uint16_t>({feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f),
                           feather::FloatToHalf(4.0f), feather::FloatToHalf(5.0f), feather::FloatToHalf(6.0f)},
                          {2, 3});

    auto rhs = std::make_shared<Tensor>();
    rhs->Assign<uint16_t>({feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f),
                           feather::FloatToHalf(4.0f), feather::FloatToHalf(5.0f), feather::FloatToHalf(6.0f)},
                          {3, 2});

    auto bias = std::make_shared<Tensor>();
    bias->Assign<uint16_t>({feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f)}, {2});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});
    out->mutable_data<uint16_t>();

    GemmParam param{};
    param.a = lhs;
    param.b = rhs;
    param.bias = bias;
    param.out = out;

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP16, "Gemm");
    ASSERT_NE(kernel, nullptr);
    auto* x86_kernel = dynamic_cast<feather::kernel::GemmKernel<DeviceType::X86, DataType::FP16>*>(kernel.get());
    ASSERT_NE(x86_kernel, nullptr);

    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    const std::vector<float> expected = {23.0f, 30.0f, 50.0f, 66.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(out->data<uint16_t>()[i]), expected[i], 2e-2f);
    }
}

TEST(gemm_op_test, X86SupportsTransposedBAndScalingAttributes) {
    auto lhs = std::make_shared<Tensor>();
    lhs->Assign<float>({1, 2, 3, 4, 5, 6}, {2, 3});

    auto rhs = std::make_shared<Tensor>();
    rhs->Assign<float>({1, 2, 3, 4, 5, 6}, {2, 3});

    auto bias = std::make_shared<Tensor>();
    bias->Assign<float>({2, 4}, {2});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});
    out->mutable_data<float>();

    GemmParam param{};
    param.a = lhs;
    param.b = rhs;
    param.bias = bias;
    param.out = out;
    param.alpha = 2.0f;
    param.beta = 0.5f;
    param.trans_b = true;

    feather::operators::GemmOp op("gemm_transposed_b", param);
    ASSERT_EQ(op.CheckShape(), 0);
    ASSERT_EQ(op.InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "Gemm");
    ASSERT_NE(kernel, nullptr);
    op.AttachKernel(std::move(kernel));
    ASSERT_EQ(op.Run(), 0);

    const std::vector<float> expected = {29.0f, 66.0f, 65.0f, 156.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(out->data<float>()[i], expected[i]);
    }
}
