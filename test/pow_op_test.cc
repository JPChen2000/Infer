#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "core/kernel.h"
#include "core/operator.h"
#include "core/tensor.h"
#include "src/operator/params.h"
#include "src/operator/pow_op.h"
#include "util/fp16.h"

using feather::DataType;
using feather::DeviceType;
using feather::KernelDispatcher;
using feather::OpBase;
using feather::Tensor;
using feather::operators::PowParam;

TEST(pow_op_test, PowRunsOnX86FP16) {
    auto input = std::make_shared<Tensor>();
    input->Assign<uint16_t>({feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f),
                             feather::FloatToHalf(4.0f)},
                            {4});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{4});
    out->mutable_data<uint16_t>();

    PowParam param{};
    param.input = input;
    param.out = out;
    param.exponent = 2.0f;

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::PowOp>("pow_fp16", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP16, "Pow");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    const std::vector<float> expected = {1.0f, 4.0f, 9.0f, 16.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(out->data<uint16_t>()[i]), expected[i], 1e-3f);
    }
}
