#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/kernel.h"
#include "src/kernel/div.h"
#if defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2")
#endif
#include "src/kernel/x86/elementwise.h"
#if defined(__GNUC__)
#pragma GCC pop_options
#endif
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

#if defined(__GNUC__)
__attribute__((target("avx2")))
#endif
void RunFp32SameShapeVectorDivisionTest() {
    auto lhs = std::make_shared<Tensor>();
    lhs->Assign<float>({8.0f, 12.0f, 18.0f, 25.0f, 32.0f, 45.0f, 63.0f, 80.0f}, {2, 4});
    auto rhs = std::make_shared<Tensor>();
    rhs->Assign<float>({2.0f, 3.0f, 6.0f, 5.0f, 4.0f, 9.0f, 7.0f, 8.0f}, {2, 4});
    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 4});
    out->mutable_data<float>();

    BinaryParam param;
    param.lhs = lhs;
    param.rhs = rhs;
    param.out = out;
    ASSERT_TRUE(feather::kernel::x86::elementwise_detail::TryComputeLastDimensionBroadcastFp32<
                 feather::kernel::x86::elementwise_detail::BinaryOperation::kDiv>(&param));

    const std::vector<float> expected = {4.0f, 4.0f, 3.0f, 5.0f, 8.0f, 5.0f, 9.0f, 10.0f};
    for (size_t index = 0; index < expected.size(); ++index) {
        EXPECT_FLOAT_EQ(out->data<float>()[index], expected[index]);
    }
}

TEST(x86_div_test, Fp32SameShapeUsesVectorElementwisePath) {
    RunFp32SameShapeVectorDivisionTest();
}

}  // namespace
