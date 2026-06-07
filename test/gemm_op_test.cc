#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/kernel.h"
#include "core/operator.h"
#include "core/tensor.h"
#include "src/operator/params.h"
#include "src/operator/gemm_op.h"

using feather::KernelDispatcher;
using feather::OpBase;
using feather::Tensor;
using feather::DeviceType;
using feather::DataType;
using feather::operators::GemmParam;

TEST(gemm_op_test, GemmRunsOnX86) {
    auto lhs = std::make_shared<Tensor>();
    lhs->Assign<float>({1, 2, 3, 4, 5, 6}, {2, 3});

    auto rhs = std::make_shared<Tensor>();
    rhs->Assign<float>({1, 2, 3, 4, 5, 6}, {3, 2});

    auto bias = std::make_shared<Tensor>();
    bias->Assign<float>({1, 1, 2, 2}, {2, 2});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});

    GemmParam param;
    param.a = lhs;
    param.b = rhs;
    param.bias = bias;
    param.out = out;

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::GemmOp>("gemm0", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "Gemm");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    std::vector<float> expected = {
        23, 29,
        51, 66,
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(out->data<float>()[i], expected[i]);
    }
}
