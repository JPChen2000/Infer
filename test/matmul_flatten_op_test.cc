#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/kernel.h"
#include "core/operator.h"
#include "core/tensor.h"
#include "src/kernel/flatten.h"
#include "src/kernel/matmul.h"
#include "src/operator/flatten_op.h"
#include "src/operator/matmul_op.h"
#include "src/operator/params.h"
#include "util/fp16.h"

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

TEST(matmul_flatten_op_test, CommonFp16MatMulKernelRunsCorrectly) {
    auto lhs = std::make_shared<Tensor>();
    lhs->Assign<uint16_t>({feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f),
                           feather::FloatToHalf(4.0f), feather::FloatToHalf(5.0f), feather::FloatToHalf(6.0f)},
                          {2, 3});

    auto rhs = std::make_shared<Tensor>();
    rhs->Assign<uint16_t>({feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f),
                           feather::FloatToHalf(4.0f), feather::FloatToHalf(5.0f), feather::FloatToHalf(6.0f)},
                          {3, 2});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});
    out->mutable_data<uint16_t>();

    MatMulParam param{};
    param.a = lhs;
    param.b = rhs;
    param.out = out;

    auto kernel = KernelDispatcher::instance().create(DeviceType::COMMON, DataType::FP16, "MatMul");
    ASSERT_NE(kernel, nullptr);
    auto* common_kernel =
        dynamic_cast<feather::kernel::MatMulKernel<DeviceType::COMMON, DataType::FP16>*>(kernel.get());
    ASSERT_NE(common_kernel, nullptr);

    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    const std::vector<float> expected = {22, 28, 49, 64};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(out->data<uint16_t>()[i]), expected[i], 2e-2f);
    }
}

TEST(matmul_flatten_op_test, X86Fp16MatMulKernelIsRegisteredAndRuns) {
    auto lhs = std::make_shared<Tensor>();
    lhs->Assign<uint16_t>({feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f),
                           feather::FloatToHalf(4.0f), feather::FloatToHalf(5.0f), feather::FloatToHalf(6.0f)},
                          {2, 3});

    auto rhs = std::make_shared<Tensor>();
    rhs->Assign<uint16_t>({feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f),
                           feather::FloatToHalf(4.0f), feather::FloatToHalf(5.0f), feather::FloatToHalf(6.0f)},
                          {3, 2});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});
    out->mutable_data<uint16_t>();

    MatMulParam param{};
    param.a = lhs;
    param.b = rhs;
    param.out = out;

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP16, "MatMul");
    ASSERT_NE(kernel, nullptr);
    auto* x86_kernel = dynamic_cast<feather::kernel::MatMulKernel<DeviceType::X86, DataType::FP16>*>(kernel.get());
    ASSERT_NE(x86_kernel, nullptr);

    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    const std::vector<float> expected = {22, 28, 49, 64};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(out->data<uint16_t>()[i]), expected[i], 2e-2f);
    }
}

TEST(matmul_flatten_op_test, X86Fp32FlattenUsesRegisteredKernel) {
    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "Flatten");
    ASSERT_NE(kernel, nullptr);
    auto* x86_kernel = dynamic_cast<feather::kernel::FlattenKernel<DeviceType::X86, DataType::FP32>*>(kernel.get());
    EXPECT_NE(x86_kernel, nullptr);
}

TEST(matmul_flatten_op_test, X86Fp16FlattenUsesRegisteredKernel) {
    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP16, "Flatten");
    ASSERT_NE(kernel, nullptr);
    auto* x86_kernel = dynamic_cast<feather::kernel::FlattenKernel<DeviceType::X86, DataType::FP16>*>(kernel.get());
    EXPECT_NE(x86_kernel, nullptr);
}

TEST(matmul_flatten_op_test, FlattenRunsOnX86FP16) {
    auto input = std::make_shared<Tensor>();
    input->Assign<uint16_t>({feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f),
                             feather::FloatToHalf(4.0f), feather::FloatToHalf(5.0f), feather::FloatToHalf(6.0f)},
                            {2, 3});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 3});
    out->mutable_data<uint16_t>();

    FlattenParam param{};
    param.input = input;
    param.out = out;
    param.axis = 1;

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::FlattenOp>("flatten_fp16", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP16, "Flatten");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    const std::vector<float> expected = {1, 2, 3, 4, 5, 6};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(out->data<uint16_t>()[i]), expected[i], 1e-3f);
    }
}
