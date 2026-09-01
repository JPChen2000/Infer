#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "core/kernel.h"
#include "core/tensor.h"
#include "quant/quantization.h"
#include "src/operator/params.h"
#include "src/kernel/x86/int8_conv.h"
#include "util/types.h"

#ifdef FEATHER_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace {

using feather::DataType;
using feather::DeviceType;
using feather::KernelDispatcher;
using feather::QuantizationGranularity;
using feather::QuantizationParams;
using feather::Tensor;

QuantizationParams PerTensor(float scale, int32_t zero_point) {
    QuantizationParams params;
    params.enabled = true;
    params.scale = scale;
    params.zero_point = zero_point;
    params.scales = {scale};
    params.zero_points = {zero_point};
    return params;
}

QuantizationParams PerChannel(int64_t axis, std::vector<float> scales, std::vector<int32_t> zero_points) {
    QuantizationParams params;
    params.enabled = true;
    params.granularity = QuantizationGranularity::kPerChannel;
    params.axis = axis;
    params.scales = std::move(scales);
    params.zero_points = std::move(zero_points);
    params.scale = params.scales.front();
    params.zero_point = params.zero_points.front();
    return params;
}

std::shared_ptr<Tensor> MakeInt8(std::vector<int8_t> values, std::vector<int64_t> dims,
                                 const QuantizationParams& quantization) {
    auto tensor = std::make_shared<Tensor>();
    tensor->Assign<int8_t>(values, dims);
    tensor->set_quantization(quantization);
    return tensor;
}

std::shared_ptr<Tensor> MakeInt32(std::vector<int32_t> values, std::vector<int64_t> dims) {
    auto tensor = std::make_shared<Tensor>();
    tensor->Assign<int32_t>(values, dims);
    return tensor;
}

std::unique_ptr<feather::KernelBase> CreateCommonInt8Kernel(const char* op_type) {
    return KernelDispatcher::instance().create(DeviceType::COMMON, DataType::INT8, op_type);
}

std::unique_ptr<feather::KernelBase> CreateX86Int8Kernel(const char* op_type) {
    return KernelDispatcher::instance().create(DeviceType::X86, DataType::INT8, op_type);
}

TEST(CommonInt8KernelTest, RegistersMatMulKernel) {
    EXPECT_NE(CreateCommonInt8Kernel("MatMul"), nullptr);
}

TEST(CommonInt8KernelTest, RegistersGemmKernel) {
    EXPECT_NE(CreateCommonInt8Kernel("Gemm"), nullptr);
}

TEST(CommonInt8KernelTest, RegistersFullyConnectedKernel) {
    EXPECT_NE(CreateCommonInt8Kernel("FC"), nullptr);
}

TEST(CommonInt8KernelTest, RegistersConv2DKernel) {
    EXPECT_NE(CreateCommonInt8Kernel("Conv2D"), nullptr);
}

TEST(CommonInt8KernelTest, FcUsesInt32AccumulationZeroPointsAndPerChannelWeights) {
    auto input = MakeInt8({5, 1}, {1, 2}, PerTensor(0.5f, 3));
    auto weight = MakeInt8({2, 3, 6, -1}, {2, 2}, PerChannel(1, {0.25f, 0.5f}, {-2, 1}));
    auto bias = MakeInt32({4, -4}, {2});
    auto output = std::make_shared<Tensor>(std::vector<int64_t>{1, 2});
    output->set_data_type(DataType::INT8);
    output->set_quantization(PerTensor(0.25f, 10));

    feather::operators::FcParam param;
    param.input = input;
    param.w = weight;
    param.bias = bias;
    param.out = output;
    auto kernel = CreateCommonInt8Kernel("FC");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);
    EXPECT_EQ(output->data<int8_t>()[0], 8);
    EXPECT_EQ(output->data<int8_t>()[1], 14);
}

TEST(CommonInt8KernelTest, MatMulRequantizesAndSaturatesOutput) {
    auto lhs = MakeInt8({127}, {1, 1}, PerTensor(1.0f, 0));
    auto rhs = MakeInt8({127}, {1, 1}, PerTensor(1.0f, 0));
    auto output = std::make_shared<Tensor>(std::vector<int64_t>{1, 1});
    output->set_data_type(DataType::INT8);
    output->set_quantization(PerTensor(0.01f, 0));

    feather::operators::MatMulParam param;
    param.a = lhs;
    param.b = rhs;
    param.out = output;
    auto kernel = CreateCommonInt8Kernel("MatMul");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);
    EXPECT_EQ(output->data<int8_t>()[0], 127);
}

TEST(CommonInt8KernelTest, MatMulSupportsBroadcastedBatchDimensions) {
    auto lhs = MakeInt8({1, 2, 3, 4}, {2, 1, 2}, PerTensor(1.0f, 0));
    auto rhs = MakeInt8({1, 0, 0, 1}, {1, 2, 2}, PerTensor(1.0f, 0));
    auto output = std::make_shared<Tensor>(std::vector<int64_t>{2, 1, 2});
    output->set_data_type(DataType::INT8);
    output->set_quantization(PerTensor(1.0f, 0));

    feather::operators::MatMulParam param;
    param.a = lhs;
    param.b = rhs;
    param.out = output;
    auto kernel = CreateCommonInt8Kernel("MatMul");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);
    EXPECT_EQ(output->data<int8_t>()[0], 1);
    EXPECT_EQ(output->data<int8_t>()[1], 2);
    EXPECT_EQ(output->data<int8_t>()[2], 3);
    EXPECT_EQ(output->data<int8_t>()[3], 4);
}

TEST(CommonInt8KernelTest, GemmAcceptsInt32VectorBias) {
    auto lhs = MakeInt8({1, 2, 3, 4}, {2, 2}, PerTensor(1.0f, 0));
    auto rhs = MakeInt8({1, 0, 0, 1}, {2, 2}, PerTensor(1.0f, 0));
    auto bias = MakeInt32({1, 2}, {2});
    auto output = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});
    output->set_data_type(DataType::INT8);
    output->set_quantization(PerTensor(1.0f, 0));

    feather::operators::GemmParam param;
    param.a = lhs;
    param.b = rhs;
    param.bias = bias;
    param.out = output;
    auto kernel = CreateCommonInt8Kernel("Gemm");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);
    EXPECT_EQ(output->data<int8_t>()[0], 2);
    EXPECT_EQ(output->data<int8_t>()[1], 4);
    EXPECT_EQ(output->data<int8_t>()[2], 4);
    EXPECT_EQ(output->data<int8_t>()[3], 6);
}

TEST(CommonInt8KernelTest, Conv2DUsesInt32Bias) {
    auto input = MakeInt8({5}, {1, 1, 1, 1}, PerTensor(0.5f, 3));
    auto weight = MakeInt8({6}, {1, 1, 1, 1}, PerTensor(0.25f, 2));
    auto bias = MakeInt32({4}, {1});
    auto output = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 1, 1});
    output->set_data_type(DataType::INT8);
    output->set_quantization(PerTensor(0.25f, 0));

    feather::operators::Conv2dParam param;
    param.input = input;
    param.w = weight;
    param.bias = bias;
    param.out = output;
    param.stride_h = 1;
    param.stride_w = 1;
    param.pad_h = 0;
    param.pad_w = 0;
    param.dilation_h = 1;
    param.dilation_w = 1;
    param.group = 1;
    auto kernel = CreateCommonInt8Kernel("Conv2D");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);
    EXPECT_EQ(output->data<int8_t>()[0], 6);
}

TEST(CommonInt8KernelTest, RejectsPerBlockWeightsAndFloatingPointBias) {
    auto input = MakeInt8({1, 2}, {1, 2}, PerTensor(1.0f, 0));
    auto weight = MakeInt8({1, 0, 0, 1}, {2, 2}, PerTensor(1.0f, 0));
    auto block_quantization = PerTensor(1.0f, 0);
    block_quantization.granularity = QuantizationGranularity::kPerBlock;
    block_quantization.block_size = 2;
    weight->set_quantization(block_quantization);
    auto output = std::make_shared<Tensor>(std::vector<int64_t>{1, 2});
    output->set_data_type(DataType::INT8);
    output->set_quantization(PerTensor(1.0f, 0));
    auto floating_bias = std::make_shared<Tensor>();
    floating_bias->Assign<float>({1.0f, 2.0f}, {2});

    feather::operators::FcParam param;
    param.input = input;
    param.w = weight;
    param.bias = floating_bias;
    param.out = output;
    auto kernel = CreateCommonInt8Kernel("FC");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    EXPECT_EQ(kernel->compute(), -1);
}

TEST(X86Int8KernelTest, UsesNativeKernelsForAllSupportedOperators) {
    for (const char* op_type : {"FC", "Gemm", "MatMul", "Conv2D", "Resize", "Concat", "SiLU"}) {
        auto kernel = CreateX86Int8Kernel(op_type);
        ASSERT_NE(kernel, nullptr) << op_type;
        EXPECT_EQ(kernel->device(), DeviceType::X86) << op_type;
    }
}

TEST(X86Int8KernelTest, RunsFcWithTheInt8QuantizationContract) {
    auto input = MakeInt8({5, 1}, {1, 2}, PerTensor(0.5f, 3));
    auto weight = MakeInt8({2, 3, 6, -1}, {2, 2}, PerChannel(1, {0.25f, 0.5f}, {-2, 1}));
    auto bias = MakeInt32({4, -4}, {2});
    auto output = std::make_shared<Tensor>(std::vector<int64_t>{1, 2});
    output->set_data_type(DataType::INT8);
    output->set_quantization(PerTensor(0.25f, 10));
    feather::operators::FcParam param;
    param.input = input;
    param.w = weight;
    param.bias = bias;
    param.out = output;
    auto kernel = CreateX86Int8Kernel("FC");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);
    EXPECT_EQ(output->data<int8_t>()[0], 8);
    EXPECT_EQ(output->data<int8_t>()[1], 14);
}

TEST(X86Int8KernelTest, RunsGemmMatMulAndConv2D) {
    auto lhs = MakeInt8({1, 2, 3, 4}, {2, 2}, PerTensor(1.0f, 0));
    auto rhs = MakeInt8({1, 0, 0, 1}, {2, 2}, PerTensor(1.0f, 0));
    auto output = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});
    output->set_data_type(DataType::INT8);
    output->set_quantization(PerTensor(1.0f, 0));

    feather::operators::GemmParam gemm_param;
    gemm_param.a = lhs;
    gemm_param.b = rhs;
    gemm_param.out = output;
    auto gemm = CreateX86Int8Kernel("Gemm");
    ASSERT_NE(gemm, nullptr);
    gemm->SetParam(&gemm_param);
    ASSERT_EQ(gemm->compute(), 0);
    EXPECT_EQ(output->data<int8_t>()[0], 1);
    EXPECT_EQ(output->data<int8_t>()[1], 2);
    EXPECT_EQ(output->data<int8_t>()[2], 3);
    EXPECT_EQ(output->data<int8_t>()[3], 4);

    auto matmul_output = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});
    matmul_output->set_data_type(DataType::INT8);
    matmul_output->set_quantization(PerTensor(1.0f, 0));
    feather::operators::MatMulParam matmul_param;
    matmul_param.a = lhs;
    matmul_param.b = rhs;
    matmul_param.out = matmul_output;
    auto matmul = CreateX86Int8Kernel("MatMul");
    ASSERT_NE(matmul, nullptr);
    matmul->SetParam(&matmul_param);
    ASSERT_EQ(matmul->compute(), 0);
    EXPECT_EQ(matmul_output->data<int8_t>()[0], 1);
    EXPECT_EQ(matmul_output->data<int8_t>()[1], 2);
    EXPECT_EQ(matmul_output->data<int8_t>()[2], 3);
    EXPECT_EQ(matmul_output->data<int8_t>()[3], 4);

    auto conv_input = MakeInt8({5}, {1, 1, 1, 1}, PerTensor(0.5f, 3));
    auto conv_weight = MakeInt8({6}, {1, 1, 1, 1}, PerTensor(0.25f, 2));
    auto conv_bias = MakeInt32({4}, {1});
    auto conv_output = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 1, 1});
    conv_output->set_data_type(DataType::INT8);
    conv_output->set_quantization(PerTensor(0.25f, 0));
    feather::operators::Conv2dParam conv_param;
    conv_param.input = conv_input;
    conv_param.w = conv_weight;
    conv_param.bias = conv_bias;
    conv_param.out = conv_output;
    conv_param.stride_h = 1;
    conv_param.stride_w = 1;
    conv_param.pad_h = 0;
    conv_param.pad_w = 0;
    conv_param.dilation_h = 1;
    conv_param.dilation_w = 1;
    conv_param.group = 1;
    auto conv = CreateX86Int8Kernel("Conv2D");
    ASSERT_NE(conv, nullptr);
    conv->SetParam(&conv_param);
    ASSERT_EQ(conv->compute(), 0);
    EXPECT_EQ(conv_output->data<int8_t>()[0], 6);
}

TEST(X86Int8KernelTest, RunsBlockedInt8ConvAcrossInteriorAndPadding) {
    constexpr int64_t kInputChannels = 4;
    constexpr int64_t kOutputChannels = 8;
    constexpr int64_t kHeight = 5;
    constexpr int64_t kWidth = 11;
    constexpr int64_t kKernel = 3;
    std::vector<int8_t> input(static_cast<size_t>(kInputChannels * kHeight * kWidth));
    std::vector<int8_t> weight(static_cast<size_t>(kOutputChannels * kInputChannels * kKernel * kKernel));
    for (size_t index = 0; index < input.size(); ++index) input[index] = static_cast<int8_t>((index * 5) % 17 - 8);
    for (size_t index = 0; index < weight.size(); ++index) weight[index] = static_cast<int8_t>((index * 3) % 11 - 5);
    auto input_tensor = MakeInt8(std::move(input), {1, kInputChannels, kHeight, kWidth}, PerTensor(1.0f, 0));
    auto weight_tensor = MakeInt8(std::move(weight), {kOutputChannels, kInputChannels, kKernel, kKernel},
                                  PerTensor(1.0f, 0));
    auto output = std::make_shared<Tensor>(std::vector<int64_t>{1, kOutputChannels, kHeight, kWidth});
    output->set_data_type(DataType::INT8);
    output->set_quantization(PerTensor(1.0f, 0));

    feather::operators::Conv2dParam param;
    param.input = input_tensor;
    param.w = weight_tensor;
    param.out = output;
    param.stride_h = 1;
    param.stride_w = 1;
    param.pad_h = 1;
    param.pad_w = 1;
    param.dilation_h = 1;
    param.dilation_w = 1;
    param.group = 1;
    auto kernel = CreateX86Int8Kernel("Conv2D");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    const int8_t* input_data = input_tensor->data<int8_t>();
    const int8_t* weight_data = weight_tensor->data<int8_t>();
    const int8_t* output_data = output->data<int8_t>();
    for (int64_t output_channel = 0; output_channel < kOutputChannels; ++output_channel) {
        for (int64_t output_y = 0; output_y < kHeight; ++output_y) {
            for (int64_t output_x = 0; output_x < kWidth; ++output_x) {
                int32_t expected = 0;
                for (int64_t input_channel = 0; input_channel < kInputChannels; ++input_channel) {
                    for (int64_t kernel_y = 0; kernel_y < kKernel; ++kernel_y) {
                        for (int64_t kernel_x = 0; kernel_x < kKernel; ++kernel_x) {
                            const int64_t input_y = output_y + kernel_y - 1;
                            const int64_t input_x = output_x + kernel_x - 1;
                            if (input_y < 0 || input_y >= kHeight || input_x < 0 || input_x >= kWidth) continue;
                            const size_t input_index = static_cast<size_t>(
                                (input_channel * kHeight + input_y) * kWidth + input_x);
                            const size_t weight_index = static_cast<size_t>(
                                ((output_channel * kInputChannels + input_channel) * kKernel + kernel_y) *
                                kKernel + kernel_x);
                            expected += static_cast<int32_t>(input_data[input_index]) * weight_data[weight_index];
                        }
                    }
                }
                expected = std::max(-128, std::min(127, expected));
                const size_t output_index = static_cast<size_t>(
                    (output_channel * kHeight + output_y) * kWidth + output_x);
                EXPECT_EQ(static_cast<int32_t>(output_data[output_index]), expected)
                    << "oc=" << output_channel << " y=" << output_y << " x=" << output_x;
            }
        }
    }
}

TEST(X86Int8KernelTest, RunsPointwiseInt8ConvAcrossFourPixelTile) {
    constexpr int64_t kInputChannels = 3;
    constexpr int64_t kOutputChannels = 8;
    constexpr int64_t kHeight = 4;
    constexpr int64_t kWidth = 9;
    std::vector<int8_t> input(static_cast<size_t>(kInputChannels * kHeight * kWidth));
    std::vector<int8_t> weight(static_cast<size_t>(kOutputChannels * kInputChannels));
    for (size_t index = 0; index < input.size(); ++index) input[index] = static_cast<int8_t>((index * 7) % 23 - 11);
    for (size_t index = 0; index < weight.size(); ++index) weight[index] = static_cast<int8_t>((index * 5) % 9 - 4);
    auto input_tensor = MakeInt8(std::move(input), {1, kInputChannels, kHeight, kWidth}, PerTensor(1.0f, 0));
    auto weight_tensor = MakeInt8(std::move(weight), {kOutputChannels, kInputChannels, 1, 1}, PerTensor(1.0f, 0));
    auto output = std::make_shared<Tensor>(std::vector<int64_t>{1, kOutputChannels, kHeight, kWidth});
    output->set_data_type(DataType::INT8);
    output->set_quantization(PerTensor(1.0f, 0));

    feather::operators::Conv2dParam param;
    param.input = input_tensor;
    param.w = weight_tensor;
    param.out = output;
    param.stride_h = 1;
    param.stride_w = 1;
    param.pad_h = 0;
    param.pad_w = 0;
    param.dilation_h = 1;
    param.dilation_w = 1;
    param.group = 1;
    auto kernel = CreateX86Int8Kernel("Conv2D");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    const int8_t* input_data = input_tensor->data<int8_t>();
    const int8_t* weight_data = weight_tensor->data<int8_t>();
    const int8_t* output_data = output->data<int8_t>();
    for (int64_t output_channel = 0; output_channel < kOutputChannels; ++output_channel) {
        for (int64_t output_y = 0; output_y < kHeight; ++output_y) {
            for (int64_t output_x = 0; output_x < kWidth; ++output_x) {
                int32_t expected = 0;
                for (int64_t input_channel = 0; input_channel < kInputChannels; ++input_channel) {
                    const size_t input_index = static_cast<size_t>(
                        (input_channel * kHeight + output_y) * kWidth + output_x);
                    const size_t weight_index = static_cast<size_t>(output_channel * kInputChannels + input_channel);
                    expected += static_cast<int32_t>(input_data[input_index]) * weight_data[weight_index];
                }
                expected = std::max(-128, std::min(127, expected));
                const size_t output_index = static_cast<size_t>(
                    (output_channel * kHeight + output_y) * kWidth + output_x);
                EXPECT_EQ(static_cast<int32_t>(output_data[output_index]), expected)
                    << "oc=" << output_channel << " y=" << output_y << " x=" << output_x;
            }
        }
    }
}

TEST(X86Int8KernelTest, RunsSiluWithTheInt8QuantizationContract) {
    auto input = MakeInt8({-8, -4, -1, 0, 1, 4, 8, 16}, {1, 8}, PerTensor(0.25f, 0));
    auto output = std::make_shared<Tensor>(std::vector<int64_t>{1, 8});
    output->set_data_type(DataType::INT8);
    output->set_quantization(PerTensor(0.125f, 3));

    feather::operators::UnaryParam param;
    param.input = input;
    param.out = output;
    auto kernel = CreateX86Int8Kernel("SiLU");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    const int8_t* actual = output->data<int8_t>();
    for (size_t index = 0; index < 8; ++index) {
        const float input_real = static_cast<float>(input->data<int8_t>()[index]) * 0.25f;
        const float silu = input_real / (1.0f + std::exp(-input_real));
        const double quantized = std::round(static_cast<double>(silu) / 0.125) + 3.0;
        const int32_t expected = static_cast<int32_t>(std::max(-128.0, std::min(127.0, quantized)));
        EXPECT_EQ(static_cast<int32_t>(actual[index]), expected) << "index=" << index;
    }
}

TEST(X86Int8KernelTest, RunsMaxPoolWithTheInt8QuantizationContract) {
    auto input = MakeInt8({-8, -2, 1, 4, 7, -3, 5, 2, -6}, {1, 1, 3, 3}, PerTensor(0.5f, 4));
    auto output = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 2, 2});
    output->set_data_type(DataType::INT8);
    output->set_quantization(PerTensor(0.25f, -3));

    feather::operators::PoolParam param;
    param.input = input;
    param.out = output;
    param.kernel_h = 2;
    param.kernel_w = 2;
    param.stride_h = 1;
    param.stride_w = 1;
    param.pad_h = 0;
    param.pad_w = 0;
    auto kernel = CreateX86Int8Kernel("MaxPool");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    const int8_t* actual = output->data<int8_t>();
    const int8_t* input_data = input->data<int8_t>();
    for (int64_t output_y = 0; output_y < 2; ++output_y) {
        for (int64_t output_x = 0; output_x < 2; ++output_x) {
            int32_t max_quantized = -128;
            for (int64_t kernel_y = 0; kernel_y < 2; ++kernel_y) {
                for (int64_t kernel_x = 0; kernel_x < 2; ++kernel_x) {
                    const size_t index = static_cast<size_t>((output_y + kernel_y) * 3 + output_x + kernel_x);
                    max_quantized = std::max(max_quantized, static_cast<int32_t>(input_data[index]));
                }
            }
            const double real_value = (static_cast<double>(max_quantized) - 4.0) * 0.5;
            const double quantized = std::round(real_value / 0.25) - 3.0;
            const int32_t expected = static_cast<int32_t>(std::max(-128.0, std::min(127.0, quantized)));
            EXPECT_EQ(static_cast<int32_t>(actual[output_y * 2 + output_x]), expected)
                << "y=" << output_y << " x=" << output_x;
        }
    }
}

TEST(X86Int8KernelTest, RunsResizeAndConcatWithInt8TensorData) {
    auto lhs = MakeInt8({1, 2, 3, 4}, {1, 1, 2, 2}, PerTensor(0.5f, 0));
    auto rhs = MakeInt8({5, 6, 7, 8}, {1, 1, 2, 2}, PerTensor(0.5f, 0));
    auto concat_output = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 2, 4});
    concat_output->set_data_type(DataType::INT8);
    concat_output->set_quantization(PerTensor(0.5f, 0));
    feather::operators::ConcatParam concat_param;
    concat_param.inputs = {lhs, rhs};
    concat_param.out = concat_output;
    concat_param.axis = 3;
    auto concat = CreateX86Int8Kernel("Concat");
    ASSERT_NE(concat, nullptr);
    EXPECT_EQ(concat->device(), DeviceType::X86);
    concat->SetParam(&concat_param);
    ASSERT_EQ(concat->compute(), 0);
    EXPECT_EQ(std::vector<int8_t>(concat_output->data<int8_t>(), concat_output->data<int8_t>() + 8),
              (std::vector<int8_t>{1, 2, 5, 6, 3, 4, 7, 8}));

    auto resize_output = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 4, 4});
    resize_output->set_data_type(DataType::INT8);
    resize_output->set_quantization(PerTensor(0.25f, 0));
    feather::operators::ResizeParam resize_param;
    resize_param.input = lhs;
    resize_param.out = resize_output;
    resize_param.scales = {1.0f, 1.0f, 2.0f, 2.0f};
    auto resize = CreateX86Int8Kernel("Resize");
    ASSERT_NE(resize, nullptr);
    EXPECT_EQ(resize->device(), DeviceType::X86);
    resize->SetParam(&resize_param);
    ASSERT_EQ(resize->compute(), 0);
    const std::vector<int8_t> expected_resize = {2, 2, 4, 4, 2, 2, 4, 4,
                                                 6, 6, 8, 8, 6, 6, 8, 8};
    EXPECT_EQ(std::vector<int8_t>(resize_output->data<int8_t>(), resize_output->data<int8_t>() + 16),
              expected_resize);
}

TEST(X86Int8KernelTest, RequantizesConcatInputsWithDifferentScales) {
    auto lhs = MakeInt8({1, 2, 3, 4}, {1, 1, 2, 2}, PerTensor(0.5f, 0));
    auto rhs = MakeInt8({5, 6, 7, 8}, {1, 1, 2, 2}, PerTensor(1.0f, 0));
    auto output = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 2, 4});
    output->set_data_type(DataType::INT8);
    output->set_quantization(PerTensor(0.25f, 0));
    feather::operators::ConcatParam param;
    param.inputs = {lhs, rhs};
    param.out = output;
    param.axis = 3;
    auto kernel = CreateX86Int8Kernel("Concat");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);
    EXPECT_EQ(std::vector<int8_t>(output->data<int8_t>(), output->data<int8_t>() + 8),
              (std::vector<int8_t>{2, 4, 20, 24, 6, 8, 28, 32}));
}

TEST(X86Int8KernelTest, RequantizesConcatInputsAcrossVectorizedLookupAndTail) {
    constexpr int64_t kWidth = 9;
    std::vector<int8_t> lhs(static_cast<size_t>(kWidth));
    std::vector<int8_t> rhs(static_cast<size_t>(kWidth));
    for (int64_t index = 0; index < kWidth; ++index) {
        lhs[static_cast<size_t>(index)] = static_cast<int8_t>(index - 4);
        rhs[static_cast<size_t>(index)] = static_cast<int8_t>(4 - index);
    }
    auto lhs_tensor = MakeInt8(std::move(lhs), {1, 1, 1, kWidth}, PerTensor(0.5f, 0));
    auto rhs_tensor = MakeInt8(std::move(rhs), {1, 1, 1, kWidth}, PerTensor(1.0f, 0));
    auto output = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 1, 2 * kWidth});
    output->set_data_type(DataType::INT8);
    output->set_quantization(PerTensor(0.25f, 0));

    feather::operators::ConcatParam param;
    param.inputs = {lhs_tensor, rhs_tensor};
    param.out = output;
    param.axis = 3;
    auto kernel = CreateX86Int8Kernel("Concat");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    const int8_t* actual = output->data<int8_t>();
    for (int64_t index = 0; index < kWidth; ++index) {
        EXPECT_EQ(static_cast<int32_t>(actual[index]), (index - 4) * 2);
        EXPECT_EQ(static_cast<int32_t>(actual[kWidth + index]), (4 - index) * 4);
    }
}

TEST(X86Int8KernelTest, RunsDirectInt8UnaryAndBinaryKernels) {
    auto input = MakeInt8({-4, -1, 0, 3, 7, 12, 20, 31}, {1, 8}, PerTensor(0.25f, 0));
    auto rhs = MakeInt8({2, 4, 6, 8, 10, 12, 14, 16}, {1, 8}, PerTensor(0.25f, 0));
    auto unary_output = std::make_shared<Tensor>(std::vector<int64_t>{1, 8});
    unary_output->set_data_type(DataType::INT8);
    unary_output->set_quantization(PerTensor(0.25f, 0));
    feather::operators::UnaryParam unary_param;
    unary_param.input = input;
    unary_param.out = unary_output;
    auto relu = CreateX86Int8Kernel("ReLU");
    ASSERT_NE(relu, nullptr);
    relu->SetParam(&unary_param);
    ASSERT_EQ(relu->compute(), 0);
    EXPECT_EQ(std::vector<int8_t>(unary_output->data<int8_t>(), unary_output->data<int8_t>() + 8),
              (std::vector<int8_t>{0, 0, 0, 3, 7, 12, 20, 31}));

    auto binary_output = std::make_shared<Tensor>(std::vector<int64_t>{1, 8});
    binary_output->set_data_type(DataType::INT8);
    binary_output->set_quantization(PerTensor(0.25f, 0));
    feather::operators::BinaryParam binary_param;
    binary_param.lhs = input;
    binary_param.rhs = rhs;
    binary_param.out = binary_output;
    auto add = CreateX86Int8Kernel("Add");
    ASSERT_NE(add, nullptr);
    add->SetParam(&binary_param);
    ASSERT_EQ(add->compute(), 0);
    EXPECT_EQ(std::vector<int8_t>(binary_output->data<int8_t>(), binary_output->data<int8_t>() + 8),
              (std::vector<int8_t>{-2, 3, 6, 11, 17, 24, 34, 47}));
}

TEST(X86Int8KernelTest, PacksConvWeightsInEightOutputChannelBlocks) {
    constexpr int64_t kOutputChannels = 16;
    constexpr int64_t kPatchSize = 5;
    std::vector<int8_t> weights(static_cast<size_t>(kOutputChannels * kPatchSize));
    for (int64_t output_channel = 0; output_channel < kOutputChannels; ++output_channel) {
        for (int64_t patch_index = 0; patch_index < kPatchSize; ++patch_index) {
            weights[static_cast<size_t>(output_channel * kPatchSize + patch_index)] =
                static_cast<int8_t>(output_channel * 7 + patch_index - 40);
        }
    }

    std::vector<int8_t> packed;
    feather::kernel::x86::PackInt8ConvWeightsOc8(weights.data(), kOutputChannels, kPatchSize, &packed);

    ASSERT_EQ(packed.size(), weights.size());
    for (int64_t block = 0; block < kOutputChannels / 8; ++block) {
        for (int64_t patch_index = 0; patch_index < kPatchSize; ++patch_index) {
            for (int64_t lane = 0; lane < 8; ++lane) {
                const size_t packed_index = static_cast<size_t>((block * kPatchSize + patch_index) * 8 + lane);
                const size_t source_index =
                    static_cast<size_t>(((block * 8 + lane) * kPatchSize) + patch_index);
                EXPECT_EQ(packed[packed_index], weights[source_index])
                    << "block=" << block << " patch=" << patch_index << " lane=" << lane;
            }
        }
    }
}

TEST(X86Int8KernelTest, AccumulatesOc8Int8DotProductsMaddubs) {
  constexpr int64_t kPatchSize = 5;
  const int8_t input[kPatchSize] = {
      static_cast<int8_t>(-128), static_cast<int8_t>(127),
      static_cast<int8_t>(-3), static_cast<int8_t>(42),
      static_cast<int8_t>(-91)};
  const int8_t weight_values[8] = {
      static_cast<int8_t>(-128), static_cast<int8_t>(-127),
      static_cast<int8_t>(-1), static_cast<int8_t>(0),
      static_cast<int8_t>(1), static_cast<int8_t>(63),
      static_cast<int8_t>(127), static_cast<int8_t>(-64)};
  int8_t packed_weight[kPatchSize * 8];
  for (int64_t k = 0; k < kPatchSize; ++k) {
    for (int oc = 0; oc < 8; ++oc) {
      packed_weight[k * 8 + oc] = weight_values[(k * 3 + oc) % 8];
    }
  }

  int32_t accumulators[8] = {17, -23, 101, 7, -311, 409, 19, -5};
  int32_t expected[8];
  for (int oc = 0; oc < 8; ++oc) {
    expected[oc] = accumulators[oc];
    for (int64_t k = 0; k < kPatchSize; ++k) {
      expected[oc] += static_cast<int32_t>(input[k]) *
                      static_cast<int32_t>(packed_weight[k * 8 + oc]);
    }
  }

  feather::kernel::x86::AccumulateInt8Oc8Maddubs(input, packed_weight, kPatchSize, accumulators);

  for (int oc = 0; oc < 8; ++oc) {
    EXPECT_EQ(expected[oc], accumulators[oc]) << "oc=" << oc;
  }
}
TEST(X86Int8KernelTest, PacksInt8ConvWeightsForMaddubsPairs) {
    constexpr int64_t kOutputChannels = 8;
    constexpr int64_t kInputChannels = 1;
    constexpr int64_t kKernelH = 1;
    constexpr int64_t kKernelW = 5;
    std::vector<int8_t> weights(static_cast<size_t>(kOutputChannels * kKernelW));
    for (int64_t oc = 0; oc < kOutputChannels; ++oc) {
        for (int64_t x = 0; x < kKernelW; ++x) {
            weights[static_cast<size_t>(oc * kKernelW + x)] =
                static_cast<int8_t>(oc * 11 + x * 7 - 50);
        }
    }
    std::vector<int8_t> packed;
    feather::kernel::x86::PackInt8ConvWeightsMaddubsPair(
        weights.data(), kOutputChannels, kInputChannels, kKernelH, kKernelW, &packed);
    ASSERT_EQ(packed.size(), static_cast<size_t>(3 * 32));
    for (int64_t pair = 0; pair < 3; ++pair) {
        const int64_t first_x = pair * 2;
        const int64_t block_offset = pair * 32;
        for (int64_t x_offset = 0; x_offset < 2; ++x_offset) {
            const int64_t x = first_x + x_offset;
            for (int64_t oc = 0; oc < 8; ++oc) {
                const int8_t value = x < kKernelW
                    ? weights[static_cast<size_t>(oc * kKernelW + x)]
                    : static_cast<int8_t>(0);
                const size_t index = static_cast<size_t>(block_offset + x_offset * 16 + oc * 2);
                EXPECT_EQ(packed[index], value) << "pair=" << pair << " x_offset=" << x_offset << " oc=" << oc;
                EXPECT_EQ(static_cast<uint8_t>(packed[index + 1]),
                          static_cast<uint8_t>(value) ^ static_cast<uint8_t>(0xff));
            }
        }
    }
}

TEST(X86Int8KernelTest, AccumulatesOc8Int8DotProductsFromPackedMaddubsPairs) {
    constexpr int64_t kPairCount = 3;
    constexpr int64_t kPatchSize = kPairCount * 2;
    const std::array<int8_t, kPatchSize> input = {
        static_cast<int8_t>(-128), static_cast<int8_t>(127),
        static_cast<int8_t>(-3), static_cast<int8_t>(42),
        static_cast<int8_t>(-91), static_cast<int8_t>(64)};
    std::vector<int8_t> weights(static_cast<size_t>(8 * kPatchSize));
    for (int64_t output_channel = 0; output_channel < 8; ++output_channel) {
        for (int64_t patch_index = 0; patch_index < kPatchSize; ++patch_index) {
            weights[static_cast<size_t>(output_channel * kPatchSize + patch_index)] =
                patch_index == 0 && output_channel == 0
                    ? static_cast<int8_t>(-128)
                    : static_cast<int8_t>((output_channel * 17 + patch_index * 11) % 61 - 30);
        }
    }
    std::vector<int8_t> packed;
    feather::kernel::x86::PackInt8ConvWeightsMaddubsPair(
        weights.data(), 8, 1, 1, kPatchSize, &packed);

    std::array<int32_t, 8> actual = {17, -23, 101, 7, -311, 409, 19, -5};
    std::array<int32_t, 8> expected = actual;
    for (int output_channel = 0; output_channel < 8; ++output_channel) {
        for (int64_t patch_index = 0; patch_index < kPatchSize; ++patch_index) {
            expected[static_cast<size_t>(output_channel)] +=
                static_cast<int32_t>(input[static_cast<size_t>(patch_index)]) *
                static_cast<int32_t>(weights[static_cast<size_t>(output_channel * kPatchSize + patch_index)]);
        }
    }

    feather::kernel::x86::AccumulateInt8Oc8MaddubsPairPacked(
        input.data(), packed.data(), kPairCount, actual.data());
    EXPECT_EQ(actual, expected);
}

TEST(X86Int8KernelTest, PacksInt8PointwiseWeightsForMaddubsPairs) {
    constexpr int64_t kOutputChannels = 8;
    constexpr int64_t kInputChannels = 3;
    std::vector<int8_t> weights(static_cast<size_t>(kOutputChannels * kInputChannels));
    for (int64_t output_channel = 0; output_channel < kOutputChannels; ++output_channel) {
        for (int64_t input_channel = 0; input_channel < kInputChannels; ++input_channel) {
            weights[static_cast<size_t>(output_channel * kInputChannels + input_channel)] =
                static_cast<int8_t>(output_channel * 9 + input_channel * 5 - 31);
        }
    }

    std::vector<int8_t> packed;
    feather::kernel::x86::PackInt8PointwiseWeightsMaddubsPair(
        weights.data(), kOutputChannels, kInputChannels, &packed);
    ASSERT_EQ(packed.size(), static_cast<size_t>(2 * 32));
    for (int64_t pair = 0; pair < 2; ++pair) {
        for (int64_t input_offset = 0; input_offset < 2; ++input_offset) {
            const int64_t input_channel = pair * 2 + input_offset;
            for (int64_t output_channel = 0; output_channel < 8; ++output_channel) {
                const int8_t value = input_channel < kInputChannels
                    ? weights[static_cast<size_t>(output_channel * kInputChannels + input_channel)]
                    : static_cast<int8_t>(0);
                const size_t index = static_cast<size_t>(pair * 32 + input_offset * 16 + output_channel * 2);
                EXPECT_EQ(packed[index], value)
                    << "pair=" << pair << " input_offset=" << input_offset
                    << " output_channel=" << output_channel;
                EXPECT_EQ(static_cast<uint8_t>(packed[index + 1]),
                          static_cast<uint8_t>(value) ^ static_cast<uint8_t>(0xff));
            }
        }
    }
}

TEST(X86Int8KernelTest, RunsBlockedInt8ConvWithCachedMaddubsPairsAcrossBorders) {
    constexpr int64_t kBatch = 1;
    constexpr int64_t kInputChannels = 8;
    constexpr int64_t kInputHeight = 5;
    constexpr int64_t kInputWidth = 11;
    constexpr int64_t kOutputChannels = 8;
    constexpr int64_t kKernel = 3;
    constexpr int64_t kPadding = 1;

    std::vector<int8_t> input_values(static_cast<size_t>(kBatch * kInputChannels * kInputHeight * kInputWidth));
    std::vector<int8_t> weight_values(static_cast<size_t>(kOutputChannels * kInputChannels * kKernel * kKernel));
    std::vector<int32_t> bias_values(static_cast<size_t>(kOutputChannels));
    for (size_t index = 0; index < input_values.size(); ++index) {
        input_values[index] = static_cast<int8_t>((index * 17) % 61 - 30);
    }
    for (size_t index = 0; index < weight_values.size(); ++index) {
        weight_values[index] = static_cast<int8_t>((index * 13) % 29 - 14);
    }
    for (int64_t channel = 0; channel < kOutputChannels; ++channel) {
        bias_values[static_cast<size_t>(channel)] = static_cast<int32_t>(channel * 7 - 19);
    }
    auto input = MakeInt8(std::move(input_values), {kBatch, kInputChannels, kInputHeight, kInputWidth},
                          PerTensor(0.25f, 0));
    auto weight = MakeInt8(std::move(weight_values), {kOutputChannels, kInputChannels, kKernel, kKernel},
                           PerChannel(0, std::vector<float>(kOutputChannels, 0.125f),
                                      std::vector<int32_t>(kOutputChannels, 0)));
    auto bias = MakeInt32(std::move(bias_values), {kOutputChannels});
    auto output = std::make_shared<Tensor>(std::vector<int64_t>{kBatch, kOutputChannels, kInputHeight, kInputWidth});
    output->set_data_type(DataType::INT8);
    output->set_quantization(PerTensor(0.5f, 0));

    feather::operators::Conv2dParam param;
    param.input = input;
    param.w = weight;
    param.bias = bias;
    param.out = output;
    param.pad_h = kPadding;
    param.pad_w = kPadding;
    param.stride_h = 1;
    param.stride_w = 1;
    param.dilation_h = 1;
    param.dilation_w = 1;
    param.group = 1;

    auto kernel = CreateX86Int8Kernel("Conv2D");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    const int8_t* input_data = input->data<int8_t>();
    const int8_t* weight_data = weight->data<int8_t>();
    const int32_t* bias_data = bias->data<int32_t>();
    const int8_t* output_data = output->data<int8_t>();
    for (int64_t output_channel = 0; output_channel < kOutputChannels; ++output_channel) {
        for (int64_t output_y = 0; output_y < kInputHeight; ++output_y) {
            for (int64_t output_x = 0; output_x < kInputWidth; ++output_x) {
                int64_t accumulator = bias_data[output_channel];
                for (int64_t input_channel = 0; input_channel < kInputChannels; ++input_channel) {
                    for (int64_t kernel_y = 0; kernel_y < kKernel; ++kernel_y) {
                        for (int64_t kernel_x = 0; kernel_x < kKernel; ++kernel_x) {
                            const int64_t input_y = output_y + kernel_y - kPadding;
                            const int64_t input_x = output_x + kernel_x - kPadding;
                            if (input_y < 0 || input_y >= kInputHeight || input_x < 0 || input_x >= kInputWidth) {
                                continue;
                            }
                            const size_t input_index = static_cast<size_t>(
                                (input_channel * kInputHeight + input_y) * kInputWidth + input_x);
                            const size_t weight_index = static_cast<size_t>(
                                ((output_channel * kInputChannels + input_channel) * kKernel + kernel_y) *
                                kKernel + kernel_x);
                            accumulator += static_cast<int32_t>(input_data[input_index]) *
                                           static_cast<int32_t>(weight_data[weight_index]);
                        }
                    }
                }
                const int32_t expected = std::max(-128, std::min(127, static_cast<int32_t>(std::round(
                    static_cast<double>(accumulator) * 0.25 * 0.125 / 0.5))));
                const size_t output_index = static_cast<size_t>(
                    (output_channel * kInputHeight + output_y) * kInputWidth + output_x);
                EXPECT_EQ(static_cast<int32_t>(output_data[output_index]), expected)
                    << "oc=" << output_channel << " y=" << output_y << " x=" << output_x;
            }
        }
    }

    const std::vector<int8_t> first_output(
        output->data<int8_t>(), output->data<int8_t>() + output->numel());
    ASSERT_EQ(kernel->compute(), 0);
    for (int64_t index = 0; index < output->numel(); ++index) {
        EXPECT_EQ(output->data<int8_t>()[index], first_output[static_cast<size_t>(index)]) << "index=" << index;
    }
}

TEST(X86Int8KernelTest, AccumulatesOc8Int8DotProductsPairwise) {
    constexpr int64_t kPatchSize = 9;
    std::vector<int8_t> input(kPatchSize);
    std::vector<int8_t> weight(static_cast<size_t>(8 * kPatchSize));
    for (int64_t index = 0; index < kPatchSize; ++index) {
        input[static_cast<size_t>(index)] = static_cast<int8_t>((index * 13) % 29 - 14);
    }
    for (int64_t output_channel = 0; output_channel < 8; ++output_channel) {
        for (int64_t index = 0; index < kPatchSize; ++index) {
            weight[static_cast<size_t>(output_channel * kPatchSize + index)] =
                static_cast<int8_t>((output_channel * 7 + index * 5) % 23 - 11);
        }
    }
    input[0] = 127;
    input[1] = -128;
    weight[0] = -128;
    weight[1] = 127;

    std::vector<int8_t> packed;
    feather::kernel::x86::PackInt8ConvWeightsOc8(weight.data(), 8, kPatchSize, &packed);
    std::array<int32_t, 8> actual{};
    feather::kernel::x86::AccumulateInt8Oc8Pairwise(input.data(), packed.data(), kPatchSize, actual.data());
    for (int64_t output_channel = 0; output_channel < 8; ++output_channel) {
        int32_t expected = 0;
        for (int64_t index = 0; index < kPatchSize; ++index) {
            expected += static_cast<int32_t>(input[static_cast<size_t>(index)]) *
                        weight[static_cast<size_t>(output_channel * kPatchSize + index)];
        }
        EXPECT_EQ(actual[static_cast<size_t>(output_channel)], expected) << "oc=" << output_channel;
    }
}

TEST(X86Int8KernelTest, PacksConvWeightsForVnniFourElementDotProducts) {
    constexpr int64_t kOutputChannels = 8;
    constexpr int64_t kPatchSize = 5;
    std::vector<int8_t> weights(static_cast<size_t>(kOutputChannels * kPatchSize));
    for (int64_t output_channel = 0; output_channel < kOutputChannels; ++output_channel) {
        for (int64_t patch_index = 0; patch_index < kPatchSize; ++patch_index) {
            weights[static_cast<size_t>(output_channel * kPatchSize + patch_index)] =
                static_cast<int8_t>(output_channel * 7 + patch_index - 40);
        }
    }

    std::vector<int8_t> packed;
    std::vector<int32_t> sums;
    feather::kernel::x86::PackInt8ConvWeightsVnni(weights.data(), kOutputChannels, kPatchSize, &packed, &sums);

    ASSERT_EQ(packed.size(), 64U);
    ASSERT_EQ(sums.size(), static_cast<size_t>(kOutputChannels));
    constexpr size_t kBytesPerPatchGroup = 32U;
    for (int64_t output_channel = 0; output_channel < kOutputChannels; ++output_channel) {
        int32_t expected_sum = 0;
        for (int64_t patch_index = 0; patch_index < kPatchSize; ++patch_index) {
            const int8_t value = weights[static_cast<size_t>(output_channel * kPatchSize + patch_index)];
            expected_sum += static_cast<int32_t>(value);
        }
        EXPECT_EQ(sums[static_cast<size_t>(output_channel)], expected_sum);
        for (int64_t patch_index = 0; patch_index < 8; ++patch_index) {
            const size_t group = static_cast<size_t>(patch_index / 4);
            const size_t lane = static_cast<size_t>(patch_index % 4);
            const size_t offset = group * kBytesPerPatchGroup + static_cast<size_t>(output_channel) * 4U + lane;
            const int8_t expected = patch_index < kPatchSize
                                        ? weights[static_cast<size_t>(output_channel * kPatchSize + patch_index)]
                                        : 0;
            EXPECT_EQ(packed[offset], expected)
                << "channel=" << output_channel << " patch=" << patch_index;
        }
    }
}


#ifdef FEATHER_WITH_CUDA
namespace {
bool HasCudaInt8TestDevice() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}
}  // namespace

TEST(CudaInt8KernelTest, RegistersNativeKernelsWithoutFallback) {
    for (const char* op_type : {"FC", "Gemm", "MatMul", "Conv2D"}) {
        auto kernel = KernelDispatcher::instance().create(DeviceType::CUDA, DataType::INT8, op_type);
        ASSERT_NE(kernel, nullptr) << op_type;
        EXPECT_EQ(kernel->device(), DeviceType::CUDA) << op_type;
    }
}

TEST(CudaInt8KernelTest, RunsNativeFcWithInt32Bias) {
    if (!HasCudaInt8TestDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }
    auto input = MakeInt8({5, 1}, {1, 2}, PerTensor(0.5f, 3));
    auto weight = MakeInt8({2, 3, 6, -1}, {2, 2}, PerChannel(1, {0.25f, 0.5f}, {-2, 1}));
    auto bias = MakeInt32({4, -4}, {2});
    auto output = std::make_shared<Tensor>(std::vector<int64_t>{1, 2});
    output->set_data_type(DataType::INT8);
    output->set_quantization(PerTensor(0.25f, 10));
    feather::operators::FcParam param;
    param.input = input;
    param.w = weight;
    param.bias = bias;
    param.out = output;
    auto kernel = KernelDispatcher::instance().create(DeviceType::CUDA, DataType::INT8, "FC");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);
    EXPECT_EQ(output->data<int8_t>()[0], 8);
    EXPECT_EQ(output->data<int8_t>()[1], 14);
}
#endif

}  // namespace
