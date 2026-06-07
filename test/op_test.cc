#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <type_traits>
#include "core/operator.h"
#include "core/memory.h"
#include "core/tensor.h"
#include "core/kernel.h"
#include "core/dim.h"
#include "util/types.h"
#include "util/logger.h"
#include "src/operator/params.h"
#include "src/operator/fc_op.h"

using namespace feather;
using feather::operators::FcParam;

TEST(fcop_test, TestX86) {
    std::vector<float> data = {0, 1, 2, 3, 4, 5};
    std::vector<int64_t> shape = {3, 2};
    auto tensor = std::make_shared<Tensor>();
    tensor->Assign(data, shape);
    EXPECT_EQ(tensor->data_size(), 6);

    auto input = std::make_shared<Tensor>();
    std::vector<float> data1 = {0, 1, 2, 3, 4, 5};
    std::vector<int64_t> shape1 = {2, 3};
    input->Assign(data1, shape1);
    EXPECT_EQ(input->data_size(), 6);

    auto bais = std::make_shared<Tensor>();
    bais->Assign<float>({0.123, 1.45, 2.13, 2.22}, {2, 2});
    
    std::vector<int64_t> shape2 = {2, 2};
    auto out = std::make_shared<Tensor>(shape2);
    FcParam param;
    param.w = tensor;
    param.bias = bais;
    param.out = out;
    param.input = input;
    auto fc_op = std::make_shared<feather::operators::FcOp>(param);
    auto kernel = KernelDispatcher::instance().create(
                DeviceType::X86, DataType::FP32, "FC");
    fc_op->AttachKernel(std::move(kernel));
    EXPECT_EQ(fc_op->Run(), 0);
    std::cout << *input;
    std::cout << *tensor;
    std::cout << *bais;
    std::cout << *out;
}

TEST(fcop_test, FcOpImplementsUnifiedOperatorContract) {
    auto weight = std::make_shared<Tensor>();
    weight->Assign<float>({1, 2, 3, 4, 5, 6, 7, 8}, {2, 4});

    auto input = std::make_shared<Tensor>();
    input->Assign<float>({1, 2, 3, 4, 5, 6}, {3, 2});

    auto bias = std::make_shared<Tensor>();
    bias->Assign<float>({1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}, {3, 4});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{3, 4});

    FcParam param;
    param.w = weight;
    param.bias = bias;
    param.out = out;
    param.input = input;

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::FcOp>("fc_test", param);
    ASSERT_EQ(op->type(), "FC");
    ASSERT_EQ(op->name(), "fc_test");
    ASSERT_EQ(op->inputs().size(), 3U);
    ASSERT_EQ(op->outputs().size(), 1U);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);
    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "FC");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    std::vector<float> expected = {
        12, 15, 18, 21,
        25, 32, 39, 46,
        38, 49, 60, 71,
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(out->data<float>()[i], expected[i]);
    }
}

TEST(fcop_test, RunFailsWithoutGraphManagedKernelBinding) {
    auto weight = std::make_shared<Tensor>();
    weight->Assign<float>({1, 2, 3, 4}, {2, 2});

    auto input = std::make_shared<Tensor>();
    input->Assign<float>({1, 2}, {1, 2});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{1, 2});

    FcParam param;
    param.w = weight;
    param.bias = nullptr;
    param.out = out;
    param.input = input;

    feather::operators::FcOp op("fc_unbound", param);
    ASSERT_EQ(op.CheckShape(), 0);
    ASSERT_EQ(op.InferOutputShapes(), 0);
    EXPECT_EQ(op.Run(), -1);
}
