#include <gtest/gtest.h>

#include <memory>

#include "core/kernel.h"
#include "src/kernel/div.h"
#include "src/operator/params.h"

namespace {

using feather::DataType;
using feather::DeviceType;
using feather::KernelBase;
using feather::KernelDispatcher;
using feather::Tensor;
using feather::operators::BinaryParam;

TEST(x86_div_test, Fp32LastDimensionScalarBroadcastUsesX86Kernel) {
    auto lhs = std::make_shared<Tensor>(sizeof(float) * 2 * 3 * 8);
    auto rhs = std::make_shared<Tensor>(sizeof(float) * 2 * 3);
    auto out = std::make_shared<Tensor>(sizeof(float) * 2 * 3 * 8);
    lhs->Resize({2, 3, 8});
    rhs->Resize({2, 3, 1});
    out->Resize({2, 3, 8});
    lhs->set_data_type(DataType::FP32);
    rhs->set_data_type(DataType::FP32);
    out->set_data_type(DataType::FP32);
    for (int64_t row = 0; row < 6; ++row) {
        rhs->mutable_data<float>()[row] = static_cast<float>(row + 1);
        for (int64_t col = 0; col < 8; ++col) {
            lhs->mutable_data<float>()[row * 8 + col] = static_cast<float>(row * 8 + col + 1);
        }
    }

    BinaryParam param;
    param.lhs = lhs;
    param.rhs = rhs;
    param.out = out;
    std::unique_ptr<KernelBase> kernel =
        KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "Div");
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(kernel->device(), DeviceType::X86);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);
    for (int64_t row = 0; row < 6; ++row) {
        for (int64_t col = 0; col < 8; ++col) {
            EXPECT_FLOAT_EQ(out->data<float>()[row * 8 + col],
                            lhs->data<float>()[row * 8 + col] / rhs->data<float>()[row]);
        }
    }
}

}  // namespace
