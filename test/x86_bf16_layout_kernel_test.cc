#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/kernel.h"
#include "core/tensor.h"
#include "src/kernel/expand.h"
#include "src/kernel/identity.h"
#include "src/kernel/transpose.h"
#include "src/operator/params.h"
#include "util/bf16.h"

namespace {

using feather::BFloat16;
using feather::DataType;
using feather::DeviceType;
using feather::FloatToBFloat16;
using feather::Tensor;

std::shared_ptr<Tensor> MakeBf16Tensor(const std::vector<int64_t>& dims) {
    int64_t numel = 1;
    for (const int64_t dim : dims) numel *= dim;
    std::vector<BFloat16> values(static_cast<size_t>(numel));
    for (int64_t index = 0; index < numel; ++index) values[static_cast<size_t>(index)] = {FloatToBFloat16(static_cast<float>(index))};
    auto tensor = std::make_shared<Tensor>();
    tensor->Assign<BFloat16>(values, dims);
    return tensor;
}

std::shared_ptr<Tensor> MakeOutput(const std::vector<int64_t>& dims) {
    int64_t numel = 1;
    for (const int64_t dim : dims) numel *= dim;
    auto tensor = std::make_shared<Tensor>(static_cast<size_t>(numel) * sizeof(BFloat16));
    tensor->Resize(dims);
    return tensor;
}

TEST(x86_bf16_layout_kernel_test, ExpandsRepeatedContiguousBlockWithNativeKernel) {
    auto input = MakeBf16Tensor({1, 2, 1, 2, 3});
    auto shape = std::make_shared<Tensor>();
    shape->Assign<int64_t>({1, 2, 4, 2, 3}, {5});
    auto out = MakeOutput({1, 2, 4, 2, 3});

    feather::operators::ExpandParam param{};
    param.input = input;
    param.shape = shape;
    param.out = out;
    auto kernel = feather::KernelDispatcher::instance().create(DeviceType::X86, DataType::BF16, "Expand");
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->device(), DeviceType::X86);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);
    ASSERT_EQ(out->data_type(), DataType::BF16);

    for (int64_t outer = 0; outer < 2; ++outer) {
        for (int64_t repeat = 0; repeat < 4; ++repeat) {
            for (int64_t inner = 0; inner < 6; ++inner) {
                const int64_t output_offset = (outer * 4 + repeat) * 6 + inner;
                const int64_t input_offset = outer * 6 + inner;
                EXPECT_EQ(out->data<BFloat16>()[output_offset].bits, input->data<BFloat16>()[input_offset].bits);
            }
        }
    }
}

TEST(x86_bf16_layout_kernel_test, TransposesLastTwoAxesWithNativeKernel) {
    auto input = MakeBf16Tensor({1, 2, 3, 4});
    auto out = MakeOutput({1, 2, 4, 3});

    feather::operators::TransposeParam param{};
    param.input = input;
    param.out = out;
    param.perm = {0, 1, 3, 2};
    auto kernel = feather::KernelDispatcher::instance().create(DeviceType::X86, DataType::BF16, "Transpose");
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->device(), DeviceType::X86);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);
    ASSERT_EQ(out->data_type(), DataType::BF16);

    for (int64_t head = 0; head < 2; ++head) {
        for (int64_t row = 0; row < 3; ++row) {
            for (int64_t column = 0; column < 4; ++column) {
                const int64_t input_offset = ((head * 3 + row) * 4) + column;
                const int64_t output_offset = ((head * 4 + column) * 3) + row;
                EXPECT_EQ(out->data<BFloat16>()[output_offset].bits, input->data<BFloat16>()[input_offset].bits);
            }
        }
    }
}

TEST(x86_bf16_layout_kernel_test, TransposesLastTwoAxesAcrossSimdTilesAndTails) {
    auto input = MakeBf16Tensor({1, 2, 9, 10});
    auto out = MakeOutput({1, 2, 10, 9});

    feather::operators::TransposeParam param{};
    param.input = input;
    param.out = out;
    param.perm = {0, 1, 3, 2};
    auto kernel = feather::KernelDispatcher::instance().create(DeviceType::X86, DataType::BF16, "Transpose");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    for (int64_t head = 0; head < 2; ++head) {
        for (int64_t row = 0; row < 9; ++row) {
            for (int64_t column = 0; column < 10; ++column) {
                const int64_t input_offset = ((head * 9 + row) * 10) + column;
                const int64_t output_offset = ((head * 10 + column) * 9) + row;
                EXPECT_EQ(out->data<BFloat16>()[output_offset].bits, input->data<BFloat16>()[input_offset].bits);
            }
        }
    }
}

TEST(x86_bf16_layout_kernel_test, IdentityKeepsBf16DataType) {
    auto input = MakeBf16Tensor({2, 4});
    auto out = MakeOutput({2, 4});

    feather::operators::UnaryParam param{};
    param.input = input;
    param.out = out;
    auto kernel = feather::KernelDispatcher::instance().create(DeviceType::X86, DataType::BF16, "Identity");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);
    EXPECT_EQ(out->data_type(), DataType::BF16);
    for (int64_t index = 0; index < input->numel(); ++index) {
        EXPECT_EQ(out->data<BFloat16>()[index].bits, input->data<BFloat16>()[index].bits);
    }
}

}  // namespace
