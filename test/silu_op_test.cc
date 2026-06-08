#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "core/kernel.h"
#include "core/operator.h"
#include "core/tensor.h"
#include "src/operator/params.h"
#include "src/operator/silu_op.h"
#include "util/fp16.h"

using feather::DataType;
using feather::DeviceType;
using feather::KernelDispatcher;
using feather::OpBase;
using feather::Tensor;
using feather::operators::UnaryParam;

namespace {

float Silu(float value) {
    return value / (1.0f + std::exp(-value));
}

}  // namespace

TEST(silu_op_test, SiluRunsOnX86FP32) {
    auto input = std::make_shared<Tensor>();
    input->Assign<float>({-2.0f, -1.0f, 0.0f, 2.0f}, {4});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{4});

    UnaryParam param;
    param.input = input;
    param.out = out;

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::SiluOp>("silu0", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "SiLU");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    for (int64_t i = 0; i < input->numel(); ++i) {
        EXPECT_NEAR(out->data<float>()[i], Silu(input->data<float>()[i]), 1e-6f);
    }
}

TEST(silu_op_test, SiluRunsOnX86FP16) {
    auto input = std::make_shared<Tensor>();
    input->Assign<uint16_t>({feather::FloatToHalf(-2.0f), feather::FloatToHalf(-1.0f),
                             feather::FloatToHalf(0.0f), feather::FloatToHalf(2.0f)},
                            {4});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{4});
    out->mutable_data<uint16_t>();

    UnaryParam param;
    param.input = input;
    param.out = out;

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::SiluOp>("silu_fp16", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP16, "SiLU");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    for (int64_t i = 0; i < input->numel(); ++i) {
        const float expected = Silu(feather::HalfToFloat(input->data<uint16_t>()[i]));
        EXPECT_NEAR(feather::HalfToFloat(out->data<uint16_t>()[i]), expected, 1e-3f);
    }
}
