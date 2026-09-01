#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "core/kernel.h"
#include "core/tensor.h"
#include "quant/quantization.h"
#include "src/operator/params.h"

namespace {

std::shared_ptr<feather::Tensor> MakeFloatTensor(const std::vector<float>& values,
                                                 const std::vector<int64_t>& dims) {
    auto tensor = std::make_shared<feather::Tensor>();
    tensor->Assign<float>(values, dims);
    return tensor;
}

std::shared_ptr<feather::Tensor> MakeInt32Tensor(const std::vector<int32_t>& values,
                                                 const std::vector<int64_t>& dims) {
    auto tensor = std::make_shared<feather::Tensor>();
    tensor->Assign<int32_t>(values, dims);
    return tensor;
}

TEST(quantize_linear_op_test, CommonQuantizeLinearUsesScaleAndZeroPointInputs) {
    auto input = MakeFloatTensor({-100.0f, -1.0f, 0.0f, 1.0f, 100.0f}, {1, 5});
    auto scale = MakeFloatTensor({0.5f}, {1});
    auto zero_point = MakeInt32Tensor({-3}, {1});
    auto output = std::make_shared<feather::Tensor>(std::vector<int64_t>{1, 5});

    auto kernel = feather::KernelDispatcher::instance().create(feather::DeviceType::COMMON,
                                                                 feather::DataType::FP32,
                                                                 "QuantizeLinear");
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(kernel->device(), feather::DeviceType::COMMON);
    EXPECT_EQ(kernel->data_type(), feather::DataType::FP32);

    feather::operators::QuantizeLinearParam param{};
    param.input = input;
    param.scale = scale;
    param.zero_point = zero_point;
    param.out = output;
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);
    EXPECT_EQ(output->data_type(), feather::DataType::INT8);
    EXPECT_EQ(output->data<int8_t>()[0], -128);
    EXPECT_EQ(output->data<int8_t>()[1], -5);
    EXPECT_EQ(output->data<int8_t>()[2], -3);
    EXPECT_EQ(output->data<int8_t>()[3], -1);
    EXPECT_EQ(output->data<int8_t>()[4], 127);
    EXPECT_FLOAT_EQ(output->quantization_scale(), 0.5f);
    EXPECT_EQ(output->quantization().zero_point, -3);
}

TEST(quantize_linear_op_test, CommonDequantizeLinearProducesFloatingPointValues) {
    auto input = std::make_shared<feather::Tensor>();
    input->Assign<int8_t>({-5, -3, -1, 1}, {2, 2});
    feather::QuantizationParams input_quantization;
    input_quantization.enabled = true;
    input_quantization.scale = 0.5f;
    input_quantization.zero_point = -3;
    input->set_quantization(input_quantization);
    auto scale = MakeFloatTensor({0.5f}, {1});
    auto zero_point = MakeInt32Tensor({-3}, {1});
    auto output = std::make_shared<feather::Tensor>(std::vector<int64_t>{2, 2});

    auto kernel = feather::KernelDispatcher::instance().create(feather::DeviceType::COMMON,
                                                                 feather::DataType::INT8,
                                                                 "DequantizeLinear");
    ASSERT_NE(kernel, nullptr);

    feather::operators::DequantizeLinearParam param{};
    param.input = input;
    param.scale = scale;
    param.zero_point = zero_point;
    param.out = output;
    param.to = feather::DataType::FP32;
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);
    EXPECT_EQ(output->data_type(), feather::DataType::FP32);
    EXPECT_FLOAT_EQ(output->data<float>()[0], -1.0f);
    EXPECT_FLOAT_EQ(output->data<float>()[1], 0.0f);
    EXPECT_FLOAT_EQ(output->data<float>()[2], 1.0f);
    EXPECT_FLOAT_EQ(output->data<float>()[3], 2.0f);
}

TEST(quantize_linear_op_test, CommonQuantizeLinearSupportsPerChannelParameters) {
    auto input = MakeFloatTensor({1.0f, 2.0f, 4.0f, -1.0f, -2.0f, -4.0f}, {2, 3});
    auto scale = MakeFloatTensor({0.5f, 1.0f, 2.0f}, {3});
    auto zero_point = MakeInt32Tensor({0, 1, -2}, {3});
    auto output = std::make_shared<feather::Tensor>(std::vector<int64_t>{2, 3});

    auto kernel = feather::KernelDispatcher::instance().create(feather::DeviceType::COMMON,
                                                                 feather::DataType::FP32,
                                                                 "QuantizeLinear");
    ASSERT_NE(kernel, nullptr);
    feather::operators::QuantizeLinearParam param{};
    param.input = input;
    param.scale = scale;
    param.zero_point = zero_point;
    param.axis = 1;
    param.out = output;
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);
    EXPECT_EQ(std::vector<int8_t>(output->data<int8_t>(), output->data<int8_t>() + 6),
              (std::vector<int8_t>{2, 3, 0, -2, -1, -4}));
}

TEST(quantize_linear_op_test, RejectsUnsupportedOutputAndMissingParameters) {
    auto input = MakeFloatTensor({1.0f}, {1});
    auto output = std::make_shared<feather::Tensor>(std::vector<int64_t>{1});
    auto kernel = feather::KernelDispatcher::instance().create(feather::DeviceType::COMMON,
                                                                 feather::DataType::FP32,
                                                                 "QuantizeLinear");
    ASSERT_NE(kernel, nullptr);
    feather::operators::QuantizeLinearParam param{};
    param.input = input;
    param.out = output;
    kernel->SetParam(&param);
    EXPECT_EQ(kernel->compute(), -1);
}

}  // namespace
