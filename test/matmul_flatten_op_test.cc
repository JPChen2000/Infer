#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/kernel.h"
#include "core/operator.h"
#include "core/tensor.h"
#include "src/operator/flatten_op.h"
#include "src/operator/matmul_op.h"
#include "src/operator/params.h"

using feather::DataType;
using feather::DeviceType;
using feather::KernelDispatcher;
using feather::OpBase;
using feather::Tensor;
using feather::operators::FlattenParam;
using feather::operators::MatMulParam;

TEST(matmul_flatten_op_test, MatMulRunsOnX86) {
    auto lhs = std::make_shared<Tensor>();
    lhs->Assign<float>({1, 2, 3, 4, 5, 6}, {2, 3});

    auto rhs = std::make_shared<Tensor>();
    rhs->Assign<float>({1, 2, 3, 4, 5, 6}, {3, 2});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});

    MatMulParam param{};
    param.a = lhs;
    param.b = rhs;
    param.out = out;

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::MatMulOp>("matmul0", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "MatMul");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    const std::vector<float> expected = {22, 28, 49, 64};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(out->data<float>()[i], expected[i]);
    }
}

TEST(matmul_flatten_op_test, FlattenRunsOnX86) {
    auto input = std::make_shared<Tensor>();
    input->Assign<float>({1, 2, 3, 4, 5, 6}, {2, 3});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 3});

    FlattenParam param{};
    param.input = input;
    param.out = out;
    param.axis = 1;

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::FlattenOp>("flatten0", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "Flatten");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    EXPECT_EQ(out->dims().data(), std::vector<int64_t>({2, 3}));
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(out->data<float>()[i], input->data<float>()[i]);
    }
}

TEST(matmul_flatten_op_test, FlattenSupportsHigherRankAxisOnX86) {
    auto input = std::make_shared<Tensor>();
    input->Assign<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 2, 3});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{4, 3});

    FlattenParam param{};
    param.input = input;
    param.out = out;
    param.axis = 2;

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::FlattenOp>("flatten_axis2", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "Flatten");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    EXPECT_EQ(out->dims().data(), std::vector<int64_t>({4, 3}));
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(out->data<float>()[i], input->data<float>()[i]);
    }
}
