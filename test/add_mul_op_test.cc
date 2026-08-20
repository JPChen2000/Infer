#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/kernel.h"
#include "core/operator.h"
#include "core/tensor.h"
#include "util/bf16.h"
#include "util/fp16.h"
#include "src/operator/add_op.h"
#include "src/operator/mul_op.h"
#include "src/operator/params.h"

using feather::DataType;
using feather::DeviceType;
using feather::KernelDispatcher;
using feather::OpBase;
using feather::Tensor;
using feather::operators::BinaryParam;

TEST(add_mul_op_test, AddRunsOnX86) {
    auto lhs = std::make_shared<Tensor>();
    lhs->Assign<float>({1, 2, 3, 4}, {2, 2});

    auto rhs = std::make_shared<Tensor>();
    rhs->Assign<float>({10, 20, 30, 40}, {2, 2});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});

    BinaryParam param{};
    param.lhs = lhs;
    param.rhs = rhs;
    param.out = out;

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::AddOp>("add0", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "Add");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    const std::vector<float> expected = {11, 22, 33, 44};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(out->data<float>()[i], expected[i]);
    }
}

TEST(add_mul_op_test, MulRunsOnX86) {
    auto lhs = std::make_shared<Tensor>();
    lhs->Assign<float>({1, 2, 3, 4}, {2, 2});

    auto rhs = std::make_shared<Tensor>();
    rhs->Assign<float>({10, 20, 30, 40}, {2, 2});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});

    BinaryParam param{};
    param.lhs = lhs;
    param.rhs = rhs;
    param.out = out;

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::MulOp>("mul0", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "Mul");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    const std::vector<float> expected = {10, 40, 90, 160};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(out->data<float>()[i], expected[i]);
    }
}

TEST(add_mul_op_test, AddBroadcastsRowVectorOnX86) {
    auto lhs = std::make_shared<Tensor>();
    lhs->Assign<float>({1, 2, 3, 4, 5, 6}, {2, 3});

    auto rhs = std::make_shared<Tensor>();
    rhs->Assign<float>({10, 20, 30}, {3});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 3});

    BinaryParam param{};
    param.lhs = lhs;
    param.rhs = rhs;
    param.out = out;

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::AddOp>("add_broadcast0", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "Add");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    EXPECT_EQ(out->dims().data(), std::vector<int64_t>({2, 3}));
    const std::vector<float> expected = {11, 22, 33, 14, 25, 36};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(out->data<float>()[i], expected[i]);
    }
}

TEST(add_mul_op_test, MulBroadcastsColumnVectorOnX86) {
    auto lhs = std::make_shared<Tensor>();
    lhs->Assign<float>({1, 2, 3, 4, 5, 6}, {2, 3});

    auto rhs = std::make_shared<Tensor>();
    rhs->Assign<float>({10, 20}, {2, 1});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 3});

    BinaryParam param{};
    param.lhs = lhs;
    param.rhs = rhs;
    param.out = out;

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::MulOp>("mul_broadcast0", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "Mul");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    EXPECT_EQ(out->dims().data(), std::vector<int64_t>({2, 3}));
    const std::vector<float> expected = {10, 20, 30, 80, 100, 120};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(out->data<float>()[i], expected[i]);
    }
}

TEST(add_mul_op_test, AddRunsOnX86FP16) {
    auto lhs = std::make_shared<Tensor>();
    lhs->Assign<uint16_t>({feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f),
                           feather::FloatToHalf(4.0f)},
                          {2, 2});

    auto rhs = std::make_shared<Tensor>();
    rhs->Assign<uint16_t>({feather::FloatToHalf(10.0f), feather::FloatToHalf(20.0f), feather::FloatToHalf(30.0f),
                           feather::FloatToHalf(40.0f)},
                          {2, 2});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});
    out->mutable_data<uint16_t>();

    BinaryParam param{};
    param.lhs = lhs;
    param.rhs = rhs;
    param.out = out;

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::AddOp>("add_fp16", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP16, "Add");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    const std::vector<float> expected = {11, 22, 33, 44};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(out->data<uint16_t>()[i]), expected[i], 1e-3f);
    }
}

TEST(add_mul_op_test, MulRunsOnX86FP16) {
    auto lhs = std::make_shared<Tensor>();
    lhs->Assign<uint16_t>({feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f),
                           feather::FloatToHalf(4.0f)},
                          {2, 2});

    auto rhs = std::make_shared<Tensor>();
    rhs->Assign<uint16_t>({feather::FloatToHalf(10.0f), feather::FloatToHalf(20.0f), feather::FloatToHalf(30.0f),
                           feather::FloatToHalf(40.0f)},
                          {2, 2});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});
    out->mutable_data<uint16_t>();

    BinaryParam param{};
    param.lhs = lhs;
    param.rhs = rhs;
    param.out = out;

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::MulOp>("mul_fp16", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP16, "Mul");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    const std::vector<float> expected = {10, 40, 90, 160};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(out->data<uint16_t>()[i]), expected[i], 1e-3f);
    }
}

TEST(add_mul_op_test, AddRunsOnX86BF16WithBroadcastAndTail) {
    auto lhs = std::make_shared<Tensor>();
    lhs->Assign<feather::BFloat16>(
        {{feather::FloatToBFloat16(1.0f)}, {feather::FloatToBFloat16(2.0f)}, {feather::FloatToBFloat16(3.0f)},
         {feather::FloatToBFloat16(4.0f)}, {feather::FloatToBFloat16(5.0f)}, {feather::FloatToBFloat16(6.0f)},
         {feather::FloatToBFloat16(7.0f)}, {feather::FloatToBFloat16(8.0f)}, {feather::FloatToBFloat16(9.0f)},
         {feather::FloatToBFloat16(10.0f)}, {feather::FloatToBFloat16(11.0f)}, {feather::FloatToBFloat16(12.0f)},
         {feather::FloatToBFloat16(13.0f)}, {feather::FloatToBFloat16(14.0f)}, {feather::FloatToBFloat16(15.0f)},
         {feather::FloatToBFloat16(16.0f)}, {feather::FloatToBFloat16(17.0f)}, {feather::FloatToBFloat16(18.0f)},
         {feather::FloatToBFloat16(19.0f)}, {feather::FloatToBFloat16(20.0f)}, {feather::FloatToBFloat16(21.0f)},
         {feather::FloatToBFloat16(22.0f)}, {feather::FloatToBFloat16(23.0f)}, {feather::FloatToBFloat16(24.0f)},
         {feather::FloatToBFloat16(25.0f)}, {feather::FloatToBFloat16(26.0f)}, {feather::FloatToBFloat16(27.0f)},
         {feather::FloatToBFloat16(28.0f)}, {feather::FloatToBFloat16(29.0f)}, {feather::FloatToBFloat16(30.0f)},
         {feather::FloatToBFloat16(31.0f)}, {feather::FloatToBFloat16(32.0f)}, {feather::FloatToBFloat16(33.0f)},
         {feather::FloatToBFloat16(34.0f)}},
        {2, 17});
    auto rhs = std::make_shared<Tensor>();
    rhs->Assign<feather::BFloat16>(
        {{feather::FloatToBFloat16(0.5f)}, {feather::FloatToBFloat16(1.5f)}, {feather::FloatToBFloat16(2.5f)},
         {feather::FloatToBFloat16(3.5f)}, {feather::FloatToBFloat16(4.5f)}, {feather::FloatToBFloat16(5.5f)},
         {feather::FloatToBFloat16(6.5f)}, {feather::FloatToBFloat16(7.5f)}, {feather::FloatToBFloat16(8.5f)},
         {feather::FloatToBFloat16(9.5f)}, {feather::FloatToBFloat16(10.5f)}, {feather::FloatToBFloat16(11.5f)},
         {feather::FloatToBFloat16(12.5f)}, {feather::FloatToBFloat16(13.5f)}, {feather::FloatToBFloat16(14.5f)},
         {feather::FloatToBFloat16(15.5f)}, {feather::FloatToBFloat16(16.5f)}},
        {17});
    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 17});
    out->mutable_data<feather::BFloat16>();

    BinaryParam param{};
    param.lhs = lhs;
    param.rhs = rhs;
    param.out = out;
    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::BF16, "Add");
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(kernel->device(), DeviceType::X86);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);
    ASSERT_EQ(out->data_type(), DataType::BF16);
    for (int64_t index = 0; index < 34; ++index) {
        const float expected = feather::BFloat16ToFloat(lhs->data<feather::BFloat16>()[index].bits) +
                               feather::BFloat16ToFloat(rhs->data<feather::BFloat16>()[index % 17].bits);
        EXPECT_NEAR(feather::BFloat16ToFloat(out->data<feather::BFloat16>()[index].bits), expected, 0.08f);
    }
}

TEST(add_mul_op_test, MulRunsOnX86BF16WithBroadcastAndTail) {
    auto lhs = std::make_shared<Tensor>();
    lhs->Assign<feather::BFloat16>(
        {{feather::FloatToBFloat16(1.0f)}, {feather::FloatToBFloat16(2.0f)}, {feather::FloatToBFloat16(3.0f)},
         {feather::FloatToBFloat16(4.0f)}, {feather::FloatToBFloat16(5.0f)}, {feather::FloatToBFloat16(6.0f)},
         {feather::FloatToBFloat16(7.0f)}, {feather::FloatToBFloat16(8.0f)}, {feather::FloatToBFloat16(9.0f)},
         {feather::FloatToBFloat16(10.0f)}, {feather::FloatToBFloat16(11.0f)}, {feather::FloatToBFloat16(12.0f)},
         {feather::FloatToBFloat16(13.0f)}, {feather::FloatToBFloat16(14.0f)}, {feather::FloatToBFloat16(15.0f)},
         {feather::FloatToBFloat16(16.0f)}, {feather::FloatToBFloat16(17.0f)}, {feather::FloatToBFloat16(18.0f)},
         {feather::FloatToBFloat16(19.0f)}, {feather::FloatToBFloat16(20.0f)}, {feather::FloatToBFloat16(21.0f)},
         {feather::FloatToBFloat16(22.0f)}, {feather::FloatToBFloat16(23.0f)}, {feather::FloatToBFloat16(24.0f)},
         {feather::FloatToBFloat16(25.0f)}, {feather::FloatToBFloat16(26.0f)}, {feather::FloatToBFloat16(27.0f)},
         {feather::FloatToBFloat16(28.0f)}, {feather::FloatToBFloat16(29.0f)}, {feather::FloatToBFloat16(30.0f)},
         {feather::FloatToBFloat16(31.0f)}, {feather::FloatToBFloat16(32.0f)}, {feather::FloatToBFloat16(33.0f)},
         {feather::FloatToBFloat16(34.0f)}},
        {2, 17});
    auto rhs = std::make_shared<Tensor>();
    rhs->Assign<feather::BFloat16>({{feather::FloatToBFloat16(2.0f)}, {feather::FloatToBFloat16(3.0f)}}, {2, 1});
    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 17});
    out->mutable_data<feather::BFloat16>();

    BinaryParam param{};
    param.lhs = lhs;
    param.rhs = rhs;
    param.out = out;
    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::BF16, "Mul");
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(kernel->device(), DeviceType::X86);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);
    ASSERT_EQ(out->data_type(), DataType::BF16);
    for (int64_t row = 0; row < 2; ++row) {
        for (int64_t col = 0; col < 17; ++col) {
            const float expected = feather::BFloat16ToFloat(lhs->data<feather::BFloat16>()[row * 17 + col].bits) *
                                   feather::BFloat16ToFloat(rhs->data<feather::BFloat16>()[row].bits);
            EXPECT_NEAR(feather::BFloat16ToFloat(out->data<feather::BFloat16>()[row * 17 + col].bits), expected,
                        0.12f);
        }
    }
}
