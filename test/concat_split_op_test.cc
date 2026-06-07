#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/kernel.h"
#include "core/operator.h"
#include "core/tensor.h"
#include "util/fp16.h"
#include "src/operator/concat_op.h"
#include "src/operator/params.h"
#include "src/operator/split_op.h"

using feather::DataType;
using feather::DeviceType;
using feather::KernelDispatcher;
using feather::OpBase;
using feather::Tensor;
using feather::operators::ConcatParam;
using feather::operators::SplitParam;

TEST(concat_split_op_test, ConcatRunsOnX86) {
    auto lhs = std::make_shared<Tensor>();
    lhs->Assign<float>({1, 2, 3, 4}, {2, 2});

    auto rhs = std::make_shared<Tensor>();
    rhs->Assign<float>({5, 6, 7, 8}, {2, 2});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 4});

    ConcatParam param{};
    param.inputs = {lhs, rhs};
    param.out = out;
    param.axis = 1;

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::ConcatOp>("concat0", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "Concat");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    EXPECT_EQ(out->dims().data(), std::vector<int64_t>({2, 4}));
    const std::vector<float> expected = {
        1, 2, 5, 6,
        3, 4, 7, 8,
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(out->data<float>()[i], expected[i]);
    }
}

TEST(concat_split_op_test, SplitRunsOnX86) {
    auto input = std::make_shared<Tensor>();
    input->Assign<float>({
        1, 2, 5, 6,
        3, 4, 7, 8,
    }, {2, 4});

    auto out0 = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});
    auto out1 = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});

    SplitParam param{};
    param.input = input;
    param.outputs = {out0, out1};
    param.axis = 1;
    param.split_sizes = {2, 2};

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::SplitOp>("split0", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "Split");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    EXPECT_EQ(out0->dims().data(), std::vector<int64_t>({2, 2}));
    EXPECT_EQ(out1->dims().data(), std::vector<int64_t>({2, 2}));

    const std::vector<float> expected0 = {1, 2, 3, 4};
    const std::vector<float> expected1 = {5, 6, 7, 8};
    for (size_t i = 0; i < expected0.size(); ++i) {
        EXPECT_FLOAT_EQ(out0->data<float>()[i], expected0[i]);
        EXPECT_FLOAT_EQ(out1->data<float>()[i], expected1[i]);
    }
}

TEST(concat_split_op_test, ConcatRunsOnX86FP16) {
    auto lhs = std::make_shared<Tensor>();
    lhs->Assign<uint16_t>({feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f),
                           feather::FloatToHalf(4.0f)},
                          {2, 2});

    auto rhs = std::make_shared<Tensor>();
    rhs->Assign<uint16_t>({feather::FloatToHalf(5.0f), feather::FloatToHalf(6.0f), feather::FloatToHalf(7.0f),
                           feather::FloatToHalf(8.0f)},
                          {2, 2});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 4});
    out->mutable_data<uint16_t>();

    ConcatParam param{};
    param.inputs = {lhs, rhs};
    param.out = out;
    param.axis = 1;

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::ConcatOp>("concat_fp16", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP16, "Concat");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    const std::vector<float> expected = {1, 2, 5, 6, 3, 4, 7, 8};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(out->data<uint16_t>()[i]), expected[i], 1e-3f);
    }
}

TEST(concat_split_op_test, SplitRunsOnX86FP16) {
    auto input = std::make_shared<Tensor>();
    input->Assign<uint16_t>({feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(5.0f),
                             feather::FloatToHalf(6.0f), feather::FloatToHalf(3.0f), feather::FloatToHalf(4.0f),
                             feather::FloatToHalf(7.0f), feather::FloatToHalf(8.0f)},
                            {2, 4});

    auto out0 = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});
    auto out1 = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});
    out0->mutable_data<uint16_t>();
    out1->mutable_data<uint16_t>();

    SplitParam param{};
    param.input = input;
    param.outputs = {out0, out1};
    param.axis = 1;
    param.split_sizes = {2, 2};

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::SplitOp>("split_fp16", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP16, "Split");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    const std::vector<float> expected0 = {1, 2, 3, 4};
    const std::vector<float> expected1 = {5, 6, 7, 8};
    for (size_t i = 0; i < expected0.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(out0->data<uint16_t>()[i]), expected0[i], 1e-3f);
        EXPECT_NEAR(feather::HalfToFloat(out1->data<uint16_t>()[i]), expected1[i], 1e-3f);
    }
}
