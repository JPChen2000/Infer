#include <gtest/gtest.h>

#include <memory>

#include "core/kernel.h"
#include "src/kernel/reduce_mean.h"
#include "src/operator/params.h"

namespace {

using feather::DataType;
using feather::DeviceType;
using feather::KernelBase;
using feather::KernelDispatcher;
using feather::Tensor;
using feather::operators::ReduceMeanParam;

TEST(x86_reduce_mean_test, Fp32LastAxisKeepdimsUsesX86Kernel) {
    constexpr int64_t kRows = 6;
    constexpr int64_t kColumns = 13;
    auto input = std::make_shared<Tensor>(sizeof(float) * kRows * kColumns);
    auto out = std::make_shared<Tensor>(sizeof(float) * kRows);
    input->Resize({2, 3, kColumns});
    out->Resize({2, 3, 1});
    input->set_data_type(DataType::FP32);
    out->set_data_type(DataType::FP32);
    for (int64_t row = 0; row < kRows; ++row) {
        for (int64_t column = 0; column < kColumns; ++column) {
            input->mutable_data<float>()[row * kColumns + column] =
                static_cast<float>(row * kColumns + column + 1) * 0.25f;
        }
    }

    ReduceMeanParam param{};
    param.input = input;
    param.out = out;
    param.axes = {-1};
    param.keepdims = true;
    std::unique_ptr<KernelBase> kernel =
        KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "ReduceMean");
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(kernel->device(), DeviceType::X86);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    for (int64_t row = 0; row < kRows; ++row) {
        float expected = 0.0f;
        for (int64_t column = 0; column < kColumns; ++column) {
            expected += input->data<float>()[row * kColumns + column];
        }
        EXPECT_NEAR(out->data<float>()[row], expected / static_cast<float>(kColumns), 1e-6f);
    }
}

}  // namespace
