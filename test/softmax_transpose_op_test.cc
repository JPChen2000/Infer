#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "core/kernel.h"
#include "core/operator.h"
#include "core/tensor.h"
#include "util/fp16.h"
#include "src/operator/params.h"
#include "src/operator/softmax_op.h"
#include "src/operator/transpose_op.h"

using feather::DataType;
using feather::DeviceType;
using feather::KernelDispatcher;
using feather::OpBase;
using feather::Tensor;
using feather::operators::SoftmaxParam;
using feather::operators::TransposeParam;

TEST(softmax_transpose_op_test, SoftmaxRunsOnX86) {
    auto input = std::make_shared<Tensor>();
    input->Assign<float>({1, 2, 3, 0, 0, 0}, {2, 3});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 3});

    SoftmaxParam param{};
    param.input = input;
    param.out = out;
    param.axis = 1;

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::SoftmaxOp>("softmax0", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "Softmax");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    const float e1 = std::exp(1.0f);
    const float e2 = std::exp(2.0f);
    const float e3 = std::exp(3.0f);
    const float sum = e1 + e2 + e3;
    EXPECT_NEAR(out->data<float>()[0], e1 / sum, 1e-6f);
    EXPECT_NEAR(out->data<float>()[1], e2 / sum, 1e-6f);
    EXPECT_NEAR(out->data<float>()[2], e3 / sum, 1e-6f);
    EXPECT_NEAR(out->data<float>()[3], 1.0f / 3.0f, 1e-6f);
    EXPECT_NEAR(out->data<float>()[4], 1.0f / 3.0f, 1e-6f);
    EXPECT_NEAR(out->data<float>()[5], 1.0f / 3.0f, 1e-6f);
}

TEST(softmax_transpose_op_test, TransposeRunsOnX86) {
    auto input = std::make_shared<Tensor>();
    input->Assign<float>({1, 2, 3, 4, 5, 6}, {2, 3});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{3, 2});

    TransposeParam param{};
    param.input = input;
    param.out = out;
    param.perm = {1, 0};

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::TransposeOp>("transpose0", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "Transpose");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    EXPECT_EQ(out->dims().data(), std::vector<int64_t>({3, 2}));
    const std::vector<float> expected = {1, 4, 2, 5, 3, 6};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(out->data<float>()[i], expected[i]);
    }
}

TEST(softmax_transpose_op_test, SoftmaxSupportsAxis0OnX86) {
    auto input = std::make_shared<Tensor>();
    input->Assign<float>({1, 2, 3, 4, 5, 6}, {2, 3});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 3});

    SoftmaxParam param{};
    param.input = input;
    param.out = out;
    param.axis = 0;

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::SoftmaxOp>("softmax_axis0", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "Softmax");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    for (int col = 0; col < 3; ++col) {
        const float sum = out->data<float>()[col] + out->data<float>()[3 + col];
        EXPECT_NEAR(sum, 1.0f, 1e-6f);
    }
}

TEST(softmax_transpose_op_test, TransposeSupports3DPermOnX86) {
    auto input = std::make_shared<Tensor>();
    input->Assign<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 3, 2});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{3, 2, 2});

    TransposeParam param{};
    param.input = input;
    param.out = out;
    param.perm = {1, 0, 2};

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::TransposeOp>("transpose3d0", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "Transpose");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    EXPECT_EQ(out->dims().data(), std::vector<int64_t>({3, 2, 2}));
    const std::vector<float> expected = {1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(out->data<float>()[i], expected[i]);
    }
}

TEST(softmax_transpose_op_test, TransposeRunsOnX86FP16) {
    auto input = std::make_shared<Tensor>();
    input->Assign<uint16_t>({feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f),
                             feather::FloatToHalf(4.0f), feather::FloatToHalf(5.0f), feather::FloatToHalf(6.0f)},
                            {2, 3});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{3, 2});
    out->mutable_data<uint16_t>();

    TransposeParam param{};
    param.input = input;
    param.out = out;
    param.perm = {1, 0};

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::TransposeOp>("transpose_fp16", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP16, "Transpose");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    const std::vector<float> expected = {1, 4, 2, 5, 3, 6};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(out->data<uint16_t>()[i]), expected[i], 1e-3f);
    }
}

TEST(softmax_transpose_op_test, TransposeSupportsYolov5HeadLayoutOnX86FP16) {
    auto input = std::make_shared<Tensor>();
    input->Assign<uint16_t>({feather::FloatToHalf(1.0f),  feather::FloatToHalf(2.0f),  feather::FloatToHalf(3.0f),
                             feather::FloatToHalf(4.0f),  feather::FloatToHalf(5.0f),  feather::FloatToHalf(6.0f),
                             feather::FloatToHalf(7.0f),  feather::FloatToHalf(8.0f),  feather::FloatToHalf(9.0f),
                             feather::FloatToHalf(10.0f), feather::FloatToHalf(11.0f), feather::FloatToHalf(12.0f),
                             feather::FloatToHalf(13.0f), feather::FloatToHalf(14.0f), feather::FloatToHalf(15.0f),
                             feather::FloatToHalf(16.0f), feather::FloatToHalf(17.0f), feather::FloatToHalf(18.0f),
                             feather::FloatToHalf(19.0f), feather::FloatToHalf(20.0f), feather::FloatToHalf(21.0f),
                             feather::FloatToHalf(22.0f), feather::FloatToHalf(23.0f), feather::FloatToHalf(24.0f)},
                            {1, 2, 3, 2, 2});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{1, 2, 2, 2, 3});
    out->mutable_data<uint16_t>();

    TransposeParam param{};
    param.input = input;
    param.out = out;
    param.perm = {0, 1, 3, 4, 2};

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::TransposeOp>("transpose_yolov5_head_fp16", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP16, "Transpose");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    const std::vector<float> expected = {1, 5,  9,  2, 6,  10, 3, 7,  11, 4, 8,  12,
                                         13, 17, 21, 14, 18, 22, 15, 19, 23, 16, 20, 24};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(out->data<uint16_t>()[i]), expected[i], 1e-3f);
    }
}
