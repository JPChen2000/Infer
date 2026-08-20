#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/kernel.h"
#include "src/kernel/cast.h"
#include "src/operator/params.h"
#include "util/bf16.h"

namespace {

using feather::BFloat16ToFloat;
using feather::FloatToBFloat16;
using feather::DataType;
using feather::DeviceType;
using feather::KernelBase;
using feather::KernelDispatcher;
using feather::Tensor;
using feather::operators::CastParam;

std::shared_ptr<Tensor> MakeTensor(size_t bytes, const std::vector<int64_t>& dims, DataType dtype) {
    auto tensor = std::make_shared<Tensor>(bytes);
    tensor->Resize(dims);
    tensor->set_data_type(dtype);
    return tensor;
}

TEST(x86_cast_test, Bf16ToFp32UsesRegisteredKernel) {
    constexpr int64_t kCount = 17;
    const std::vector<float> expected = {0.0f, 1.0f, -2.5f, 3.25f, 7.0f, -9.5f, 0.125f, 12.0f,
                                         -16.0f, 0.75f, 2.0f, -4.0f, 5.5f, 8.0f, -11.0f, 1.5f, 6.25f};
    auto input = MakeTensor(sizeof(uint16_t) * kCount, {kCount}, DataType::BF16);
    auto output = MakeTensor(sizeof(float) * kCount, {kCount}, DataType::FP32);
    auto* input_data = static_cast<uint16_t*>(input->raw_data());
    for (int64_t i = 0; i < kCount; ++i) input_data[i] = FloatToBFloat16(expected[static_cast<size_t>(i)]);

    CastParam param;
    param.input = input;
    param.out = output;
    param.to = DataType::FP32;
    std::unique_ptr<KernelBase> kernel =
        KernelDispatcher::instance().create(DeviceType::X86, DataType::BF16, "Cast");
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(kernel->device(), DeviceType::X86);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);
    for (int64_t i = 0; i < kCount; ++i) {
        EXPECT_FLOAT_EQ(output->data<float>()[i], BFloat16ToFloat(input_data[i]));
    }
}

TEST(x86_cast_test, Fp32ToBf16UsesRegisteredKernel) {
    constexpr int64_t kCount = 17;
    auto input = MakeTensor(sizeof(float) * kCount, {kCount}, DataType::FP32);
    auto output = MakeTensor(sizeof(uint16_t) * kCount, {kCount}, DataType::BF16);
    auto* input_data = input->mutable_data<float>();
    for (int64_t i = 0; i < kCount; ++i) input_data[i] = static_cast<float>(i - 8) * 0.375f;

    CastParam param;
    param.input = input;
    param.out = output;
    param.to = DataType::BF16;
    std::unique_ptr<KernelBase> kernel =
        KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "Cast");
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(kernel->device(), DeviceType::X86);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);
    for (int64_t i = 0; i < kCount; ++i) {
        EXPECT_EQ(output->data<uint16_t>()[i], FloatToBFloat16(input_data[i]));
    }
}

}  // namespace
