#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/kernel.h"
#include "core/operator.h"
#include "core/operator_registry.h"
#include "core/tensor.h"
#include "src/operator/params.h"
#include "src/operator/resize_op.h"
#include "util/fp16.h"

using feather::DataType;
using feather::DeviceType;
using feather::KernelDispatcher;
using feather::OpBase;
using feather::Tensor;
using feather::operators::ResizeParam;

TEST(resize_op_test, ResizeRunsOnX86FP16) {
    auto input = std::make_shared<Tensor>();
    input->Assign<uint16_t>({feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f),
                             feather::FloatToHalf(4.0f)},
                            {1, 1, 2, 2});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 4, 4});
    out->mutable_data<uint16_t>();

    ResizeParam param{};
    param.input = input;
    param.out = out;
    param.scales = {1.0f, 1.0f, 2.0f, 2.0f};

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::ResizeOp>("resize_fp16", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP16, "Resize");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    const std::vector<float> expected = {
        1, 1, 2, 2,
        1, 1, 2, 2,
        3, 3, 4, 4,
        3, 3, 4, 4,
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(out->data<uint16_t>()[i]), expected[i], 1e-3f);
    }
}

TEST(resize_op_test, ResizeRunsOnX86FP32Nhwc) {
    auto input = std::make_shared<Tensor>();
    input->Assign<float>({1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f, 4.0f, 40.0f}, {1, 2, 2, 2});
    input->set_layout(feather::DataLayout::NHWC);

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{1, 4, 4, 2});
    out->set_layout(feather::DataLayout::NHWC);

    ResizeParam param{};
    param.input = input;
    param.out = out;
    param.scales = {1.0f, 2.0f, 2.0f, 1.0f};

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::ResizeOp>("resize_nhwc", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = feather::CreateKernelForTensor(DeviceType::X86, "Resize", {input, out}, DataType::FP32);
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    EXPECT_EQ(out->dims().data(), std::vector<int64_t>({1, 4, 4, 2}));
    EXPECT_EQ(out->layout(), feather::DataLayout::NHWC);
    const std::vector<float> expected = {
        1, 10, 1, 10, 2, 20, 2, 20,
        1, 10, 1, 10, 2, 20, 2, 20,
        3, 30, 3, 30, 4, 40, 4, 40,
        3, 30, 3, 30, 4, 40, 4, 40,
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(out->data<float>()[i], expected[i]);
    }
}

TEST(resize_op_test, ResizeRunsOnCommonFP32Nhwc) {
    auto input = std::make_shared<Tensor>();
    input->Assign<float>({1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f, 4.0f, 40.0f}, {1, 2, 2, 2});
    input->set_layout(feather::DataLayout::NHWC);

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{1, 4, 4, 2});
    out->set_layout(feather::DataLayout::NHWC);

    ResizeParam param{};
    param.input = input;
    param.out = out;
    param.scales = {1.0f, 2.0f, 2.0f, 1.0f};

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::ResizeOp>("resize_common_nhwc", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = feather::CreateKernelForTensor(DeviceType::COMMON, "Resize", {input, out}, DataType::FP32);
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    EXPECT_EQ(out->dims().data(), std::vector<int64_t>({1, 4, 4, 2}));
    EXPECT_EQ(out->layout(), feather::DataLayout::NHWC);
    const std::vector<float> expected = {
        1, 10, 1, 10, 2, 20, 2, 20,
        1, 10, 1, 10, 2, 20, 2, 20,
        3, 30, 3, 30, 4, 40, 4, 40,
        3, 30, 3, 30, 4, 40, 4, 40,
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(out->data<float>()[i], expected[i]);
    }
}
