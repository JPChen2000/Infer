#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "core/kernel.h"
#include "core/operator.h"
#include "core/tensor.h"
#include "util/fp16.h"
#include "src/operator/params.h"
#include "src/operator/sigmoid_op.h"

using feather::KernelDispatcher;
using feather::OpBase;
using feather::Tensor;
using feather::DeviceType;
using feather::DataType;
using feather::operators::UnaryParam;

TEST(sigmoid_op_test, SigmoidRunsOnX86) {
    auto input = std::make_shared<Tensor>();
    input->Assign<float>({-2.0f, 0.0f, 2.0f}, {3});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{3});

    UnaryParam param;
    param.input = input;
    param.out = out;

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::SigmoidOp>("sigmoid0", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "Sigmoid");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    const std::vector<float> expected = {
        1.0f / (1.0f + std::exp(2.0f)),
        0.5f,
        1.0f / (1.0f + std::exp(-2.0f)),
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(out->data<float>()[i], expected[i], 1e-6f);
    }
}

TEST(sigmoid_op_test, SigmoidRunsOnX86FP16) {
    auto input = std::make_shared<Tensor>();
    input->Assign<uint16_t>({feather::FloatToHalf(-2.0f), feather::FloatToHalf(0.0f), feather::FloatToHalf(2.0f)}, {3});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{3});
    out->mutable_data<uint16_t>();

    UnaryParam param;
    param.input = input;
    param.out = out;

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::SigmoidOp>("sigmoid_fp16", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP16, "Sigmoid");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    const std::vector<float> expected = {
        1.0f / (1.0f + std::exp(2.0f)),
        0.5f,
        1.0f / (1.0f + std::exp(-2.0f)),
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(out->data<uint16_t>()[i]), expected[i], 1e-3f);
    }
}

TEST(sigmoid_op_test, SigmoidProcessesVectorBlocksAndTailOnX86FP16) {
    auto input = std::make_shared<Tensor>();
    input->Assign<uint16_t>(
        {feather::FloatToHalf(-4.0f), feather::FloatToHalf(-3.0f), feather::FloatToHalf(-2.0f),
         feather::FloatToHalf(-1.0f), feather::FloatToHalf(0.0f),  feather::FloatToHalf(1.0f),
         feather::FloatToHalf(2.0f),  feather::FloatToHalf(3.0f),  feather::FloatToHalf(4.0f)},
        {9});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{9});
    out->mutable_data<uint16_t>();

    UnaryParam param;
    param.input = input;
    param.out = out;

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::SigmoidOp>("sigmoid_fp16_blocks", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP16, "Sigmoid");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    const std::vector<float> expected = {
        1.0f / (1.0f + std::exp(4.0f)),  1.0f / (1.0f + std::exp(3.0f)),
        1.0f / (1.0f + std::exp(2.0f)),  1.0f / (1.0f + std::exp(1.0f)),
        0.5f,                            1.0f / (1.0f + std::exp(-1.0f)),
        1.0f / (1.0f + std::exp(-2.0f)), 1.0f / (1.0f + std::exp(-3.0f)),
        1.0f / (1.0f + std::exp(-4.0f)),
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(out->data<uint16_t>()[i]), expected[i], 1e-3f);
    }
}
