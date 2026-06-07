#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/kernel.h"
#include "core/operator.h"
#include "core/tensor.h"
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
