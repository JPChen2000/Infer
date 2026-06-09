#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/kernel.h"
#include "core/operator.h"
#include "core/operator_registry.h"
#include "core/tensor.h"
#include "util/fp16.h"
#include "src/operator/params.h"
#include "src/operator/pool_op.h"

using feather::KernelDispatcher;
using feather::OpBase;
using feather::Tensor;
using feather::DeviceType;
using feather::DataType;
using feather::operators::PoolParam;

TEST(pool_op_test, AvgPoolRunsOnX86) {
    auto input = std::make_shared<Tensor>();
    input->Assign<float>({
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    }, {3, 3});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});

    PoolParam param{};
    param.input = input;
    param.out = out;
    param.kernel_h = 2;
    param.kernel_w = 2;
    param.stride_h = 1;
    param.stride_w = 1;
    param.pad_h = 0;
    param.pad_w = 0;

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::AvgPoolOp>("avgpool0", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "AvgPool");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    const std::vector<float> expected = {3.0f, 4.0f, 6.0f, 7.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(out->data<float>()[i], expected[i]);
    }
}

TEST(pool_op_test, MaxPoolRunsOnX86) {
    auto input = std::make_shared<Tensor>();
    input->Assign<float>({
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    }, {3, 3});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});

    PoolParam param{};
    param.input = input;
    param.out = out;
    param.kernel_h = 2;
    param.kernel_w = 2;
    param.stride_h = 1;
    param.stride_w = 1;
    param.pad_h = 0;
    param.pad_w = 0;

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::MaxPoolOp>("maxpool0", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "MaxPool");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    const std::vector<float> expected = {5.0f, 6.0f, 8.0f, 9.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(out->data<float>()[i], expected[i]);
    }
}

TEST(pool_op_test, MaxPoolRunsOnX86FP16) {
    auto input = std::make_shared<Tensor>();
    input->Assign<uint16_t>({feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f),
                             feather::FloatToHalf(4.0f), feather::FloatToHalf(5.0f), feather::FloatToHalf(6.0f),
                             feather::FloatToHalf(7.0f), feather::FloatToHalf(8.0f), feather::FloatToHalf(9.0f)},
                            {3, 3});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});
    out->mutable_data<uint16_t>();

    PoolParam param{};
    param.input = input;
    param.out = out;
    param.kernel_h = 2;
    param.kernel_w = 2;
    param.stride_h = 1;
    param.stride_w = 1;
    param.pad_h = 0;
    param.pad_w = 0;

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::MaxPoolOp>("maxpool_fp16", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP16, "MaxPool");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    const std::vector<float> expected = {5.0f, 6.0f, 8.0f, 9.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(out->data<uint16_t>()[i]), expected[i], 1e-3f);
    }
}

TEST(pool_op_test, MaxPoolRunsOnX86Nhwc) {
    auto input = std::make_shared<Tensor>();
    input->Assign<float>({
        1, 10,
        2, 20,
        3, 30,
        4, 40,
    }, {1, 2, 2, 2});
    input->set_layout(feather::DataLayout::NHWC);

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 1, 2});
    out->set_layout(feather::DataLayout::NHWC);

    PoolParam param{};
    param.input = input;
    param.out = out;
    param.kernel_h = 2;
    param.kernel_w = 2;
    param.stride_h = 1;
    param.stride_w = 1;
    param.pad_h = 0;
    param.pad_w = 0;

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::MaxPoolOp>("maxpool_nhwc", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = feather::CreateKernelForTensor(DeviceType::X86, "MaxPool", {input, out}, DataType::FP32);
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    EXPECT_EQ(out->dims().data(), std::vector<int64_t>({1, 1, 1, 2}));
    EXPECT_EQ(out->layout(), feather::DataLayout::NHWC);
    EXPECT_FLOAT_EQ(out->data<float>()[0], 4.0f);
    EXPECT_FLOAT_EQ(out->data<float>()[1], 40.0f);
}

TEST(pool_op_test, MaxPoolRunsOnCommonNhwc) {
    auto input = std::make_shared<Tensor>();
    input->Assign<float>({
        1, 10,
        2, 20,
        3, 30,
        4, 40,
    }, {1, 2, 2, 2});
    input->set_layout(feather::DataLayout::NHWC);

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 1, 2});
    out->set_layout(feather::DataLayout::NHWC);

    PoolParam param{};
    param.input = input;
    param.out = out;
    param.kernel_h = 2;
    param.kernel_w = 2;
    param.stride_h = 1;
    param.stride_w = 1;
    param.pad_h = 0;
    param.pad_w = 0;

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::MaxPoolOp>("maxpool_common_nhwc", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = feather::CreateKernelForTensor(DeviceType::COMMON, "MaxPool", {input, out}, DataType::FP32);
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    EXPECT_EQ(out->dims().data(), std::vector<int64_t>({1, 1, 1, 2}));
    EXPECT_EQ(out->layout(), feather::DataLayout::NHWC);
    EXPECT_FLOAT_EQ(out->data<float>()[0], 4.0f);
    EXPECT_FLOAT_EQ(out->data<float>()[1], 40.0f);
}
