#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/kernel.h"
#include "core/operator.h"
#include "core/tensor.h"
#include "util/fp16.h"
#include "src/operator/params.h"
#include "src/operator/reshape_op.h"

using feather::KernelDispatcher;
using feather::OpBase;
using feather::Tensor;
using feather::DeviceType;
using feather::DataType;
using feather::operators::ReshapeParam;

TEST(reshape_op_test, ReshapePreservesDataAndChangesShape) {
    auto input = std::make_shared<Tensor>();
    input->Assign<float>({1, 2, 3, 4, 5, 6}, {2, 3});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{3, 2});

    ReshapeParam param;
    param.input = input;
    param.out = out;
    param.target_shape = {3, 2};

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::ReshapeOp>("reshape0", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "Reshape");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    EXPECT_EQ(out->dims().data(), std::vector<int64_t>({3, 2}));
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(out->data<float>()[i], input->data<float>()[i]);
    }
    EXPECT_EQ(out->raw_data(), input->raw_data());
    input->mutable_data<float>()[0] = 42.0f;
    EXPECT_FLOAT_EQ(out->data<float>()[0], 42.0f);
}

TEST(reshape_op_test, ReshapePreservesDataAndChangesShapeFP16) {
    auto input = std::make_shared<Tensor>();
    input->Assign<uint16_t>({feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f),
                             feather::FloatToHalf(4.0f), feather::FloatToHalf(5.0f), feather::FloatToHalf(6.0f)},
                            {2, 3});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{3, 2});
    out->mutable_data<uint16_t>();

    ReshapeParam param;
    param.input = input;
    param.out = out;
    param.target_shape = {3, 2};

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::ReshapeOp>("reshape_fp16", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP16, "Reshape");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    EXPECT_EQ(out->dims().data(), std::vector<int64_t>({3, 2}));
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(feather::HalfToFloat(out->data<uint16_t>()[i]),
                    feather::HalfToFloat(input->data<uint16_t>()[i]), 1e-3f);
    }
}
