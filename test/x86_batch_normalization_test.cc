#include <gtest/gtest.h>

#include <memory>
#include <typeinfo>

#include "core/kernel.h"
#include "src/kernel/batch_normalization.h"
#include "src/operator/params.h"

namespace {

using feather::DataType;
using feather::DataLayout;
using feather::DeviceType;
using feather::KernelDispatcher;
using feather::Tensor;
using feather::operators::BatchNormParam;

std::shared_ptr<Tensor> MakeFp32Tensor(const std::vector<float>& values, const std::vector<int64_t>& dims) {
    auto tensor = std::make_shared<Tensor>();
    tensor->Assign<float>(values, dims);
    tensor->set_layout(DataLayout::NCHW);
    return tensor;
}

TEST(x86_batch_normalization_test, RegistersFp32Kernel) {
    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "BatchNormalization");
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(typeid(*kernel),
              typeid(feather::kernel::BatchNormalizationKernel<DeviceType::X86, DataType::FP32>));
}

TEST(x86_batch_normalization_test, ComputesNchwFp32Channels) {
    auto input = MakeFp32Tensor({1.0f, 2.0f, 3.0f, 4.0f,
                                 10.0f, 20.0f, 30.0f, 40.0f},
                                {1, 2, 2, 2});
    auto scale = MakeFp32Tensor({2.0f, 3.0f}, {2});
    auto bias = MakeFp32Tensor({1.0f, -2.0f}, {2});
    auto mean = MakeFp32Tensor({1.0f, 20.0f}, {2});
    auto var = MakeFp32Tensor({1.0f, 4.0f}, {2});
    auto output = std::make_shared<Tensor>(std::vector<int64_t>{1, 2, 2, 2});
    output->set_layout(DataLayout::NCHW);
    output->mutable_data<float>();

    BatchNormParam param{};
    param.input = input;
    param.scale = scale;
    param.bias = bias;
    param.mean = mean;
    param.var = var;
    param.out = output;
    param.epsilon = 0.0f;

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "BatchNormalization");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    const std::vector<float> expected = {1.0f, 3.0f, 5.0f, 7.0f,
                                         -17.0f, -2.0f, 13.0f, 28.0f};
    for (size_t index = 0; index < expected.size(); ++index) {
        EXPECT_FLOAT_EQ(output->data<float>()[index], expected[index]);
    }
}

}  // namespace
