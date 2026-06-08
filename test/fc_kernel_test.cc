#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <type_traits>
#include "util/types.h"
#include "core/memory.h"
#include "core/tensor.h"
#include "core/dim.h"
#include "util/logger.h"
#include "core/kernel.h"
#include "src/kernel/fc.h"
#include "src/operator/params.h"
#include "util/fp16.h"


using namespace feather;
using feather::operators::FcParam;

TEST(fccompute_test, TestX86) {
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
    std::cout << *out;
    auto kernel = feather::kernel::FcKernel<DeviceType::COMMON, DataType::FP32>();
    FcParam param;
    param.w = tensor;
    param.bias = bais;
    param.out = out;
    param.input = input;

    kernel.SetParam((void *)&param);
    kernel.compute();
    std::cout << *out;
    std::cout << *input;
    std::cout << *tensor;
}

TEST(fccompute_test, DispatcherPrefersX86KernelWhenAvailable) {
    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "FC");
    ASSERT_NE(kernel, nullptr);
    auto* x86_kernel = dynamic_cast<feather::kernel::FcKernel<DeviceType::X86, DataType::FP32>*>(kernel.get());
    EXPECT_NE(x86_kernel, nullptr);
}

TEST(fccompute_test, CommonFp16KernelRunsCorrectly) {
    auto weight = std::make_shared<Tensor>();
    weight->Assign<uint16_t>({feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f),
                              feather::FloatToHalf(4.0f), feather::FloatToHalf(5.0f), feather::FloatToHalf(6.0f)},
                             {3, 2});

    auto input = std::make_shared<Tensor>();
    input->Assign<uint16_t>({feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f),
                             feather::FloatToHalf(4.0f), feather::FloatToHalf(5.0f), feather::FloatToHalf(6.0f)},
                            {2, 3});

    auto bias = std::make_shared<Tensor>();
    bias->Assign<uint16_t>({feather::FloatToHalf(0.5f), feather::FloatToHalf(-0.5f)}, {2});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});
    out->mutable_data<uint16_t>();

    FcParam param;
    param.w = weight;
    param.bias = bias;
    param.out = out;
    param.input = input;

    auto kernel = KernelDispatcher::instance().create(DeviceType::COMMON, DataType::FP16, "FC");
    ASSERT_NE(kernel, nullptr);
    auto* common_kernel = dynamic_cast<feather::kernel::FcKernel<DeviceType::COMMON, DataType::FP16>*>(kernel.get());
    ASSERT_NE(common_kernel, nullptr);

    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    const std::vector<float> expected = {22.5f, 27.5f, 49.5f, 63.5f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(out->data<uint16_t>()[i]), expected[i], 2e-2f);
    }
}

TEST(fccompute_test, X86Fp16KernelIsRegisteredAndRuns) {
    auto weight = std::make_shared<Tensor>();
    weight->Assign<uint16_t>({feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f),
                              feather::FloatToHalf(4.0f), feather::FloatToHalf(5.0f), feather::FloatToHalf(6.0f)},
                             {3, 2});

    auto input = std::make_shared<Tensor>();
    input->Assign<uint16_t>({feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f),
                             feather::FloatToHalf(4.0f), feather::FloatToHalf(5.0f), feather::FloatToHalf(6.0f)},
                            {2, 3});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});
    out->mutable_data<uint16_t>();

    FcParam param;
    param.w = weight;
    param.bias = nullptr;
    param.out = out;
    param.input = input;

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP16, "FC");
    ASSERT_NE(kernel, nullptr);
    auto* x86_kernel = dynamic_cast<feather::kernel::FcKernel<DeviceType::X86, DataType::FP16>*>(kernel.get());
    ASSERT_NE(x86_kernel, nullptr);

    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    const std::vector<float> expected = {22.0f, 28.0f, 49.0f, 64.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(out->data<uint16_t>()[i]), expected[i], 2e-2f);
    }
}
