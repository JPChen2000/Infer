#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/kernel.h"
#include "core/operator_registry.h"
#include "src/operator/conv2d_op.h"
#include "src/operator/relu_op.h"
#include "util/fp16.h"

using feather::Tensor;

namespace {

std::vector<float> ReferenceNchwConv(const std::vector<float>& input, const std::vector<float>& weight,
                                     const std::vector<float>& bias, int64_t batch, int64_t in_c, int64_t in_h,
                                     int64_t in_w, int64_t out_c, int64_t kernel_h, int64_t kernel_w,
                                     int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w) {
    const int64_t out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    const int64_t out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
    std::vector<float> output(static_cast<size_t>(batch * out_c * out_h * out_w), 0.0f);

    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t oc = 0; oc < out_c; ++oc) {
            for (int64_t oh = 0; oh < out_h; ++oh) {
                for (int64_t ow = 0; ow < out_w; ++ow) {
                    float sum = bias.empty() ? 0.0f : bias[static_cast<size_t>(oc)];
                    for (int64_t ic = 0; ic < in_c; ++ic) {
                        for (int64_t kh = 0; kh < kernel_h; ++kh) {
                            for (int64_t kw = 0; kw < kernel_w; ++kw) {
                                const int64_t ih = oh * stride_h - pad_h + kh;
                                const int64_t iw = ow * stride_w - pad_w + kw;
                                if (ih < 0 || ih >= in_h || iw < 0 || iw >= in_w) {
                                    continue;
                                }
                                const int64_t input_offset = ((n * in_c + ic) * in_h + ih) * in_w + iw;
                                const int64_t weight_offset = ((oc * in_c + ic) * kernel_h + kh) * kernel_w + kw;
                                sum += input[static_cast<size_t>(input_offset)] *
                                       weight[static_cast<size_t>(weight_offset)];
                            }
                        }
                    }
                    output[static_cast<size_t>(((n * out_c + oc) * out_h + oh) * out_w + ow)] = sum;
                }
            }
        }
    }
    return output;
}

}  // namespace

TEST(conv_relu_test, Conv2dThenReluRunsOnX86) {
    auto input = std::make_shared<Tensor>();
    input->Assign<float>({
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    }, {3, 3});

    auto weight = std::make_shared<Tensor>();
    weight->Assign<float>({
        1, 0,
        0, -1,
    }, {2, 2});

    auto bias = std::make_shared<Tensor>();
    bias->Assign<float>({
        1, 1,
        1, 1,
    }, {2, 2});

    auto conv_out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});

    feather::operators::Conv2dParam conv_param;
    conv_param.input = input;
    conv_param.w = weight;
    conv_param.bias = bias;
    conv_param.out = conv_out;
    conv_param.stride_h = 1;
    conv_param.stride_w = 1;
    conv_param.pad_h = 0;
    conv_param.pad_w = 0;

    feather::operators::Conv2dOp conv("conv_test", conv_param);
    ASSERT_EQ(conv.CheckShape(), 0);
    ASSERT_EQ(conv.InferOutputShapes(), 0);
    auto conv_kernel = feather::KernelDispatcher::instance().create(
        feather::DeviceType::X86, feather::DataType::FP32, "Conv2D");
    ASSERT_NE(conv_kernel, nullptr);
    conv.AttachKernel(std::move(conv_kernel));
    ASSERT_EQ(conv.Run(), 0);

    std::vector<float> conv_expected = {
        -3, -3,
        -3, -3,
    };
    for (size_t i = 0; i < conv_expected.size(); ++i) {
        EXPECT_FLOAT_EQ(conv.outputs()[0]->data<float>()[i], conv_expected[i]);
    }

    auto relu_out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});
    feather::operators::UnaryParam relu_param;
    relu_param.input = conv.outputs()[0];
    relu_param.out = relu_out;

    feather::operators::ReluOp relu("relu_test", relu_param);
    ASSERT_EQ(relu.CheckShape(), 0);
    ASSERT_EQ(relu.InferOutputShapes(), 0);
    auto relu_kernel = feather::KernelDispatcher::instance().create(
        feather::DeviceType::X86, feather::DataType::FP32, "ReLU");
    ASSERT_NE(relu_kernel, nullptr);
    relu.AttachKernel(std::move(relu_kernel));
    ASSERT_EQ(relu.Run(), 0);

    std::vector<float> relu_expected = {
        0, 0,
        0, 0,
    };
    for (size_t i = 0; i < relu_expected.size(); ++i) {
        EXPECT_FLOAT_EQ(relu.outputs()[0]->data<float>()[i], relu_expected[i]);
    }
}

TEST(conv_relu_test, Conv2dRunsFastPathOnNchw3x3) {
    auto input = std::make_shared<Tensor>();
    input->Assign<float>({
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
    }, {1, 1, 4, 4});

    auto weight = std::make_shared<Tensor>();
    weight->Assign<float>({
        1, 0, -1,
        1, 0, -1,
        1, 0, -1,
    }, {1, 1, 3, 3});

    auto bias = std::make_shared<Tensor>();
    bias->Assign<float>({0}, {1});

    auto conv_out = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 2, 2});

    feather::operators::Conv2dParam conv_param{};
    conv_param.input = input;
    conv_param.w = weight;
    conv_param.bias = bias;
    conv_param.out = conv_out;
    conv_param.stride_h = 1;
    conv_param.stride_w = 1;
    conv_param.pad_h = 0;
    conv_param.pad_w = 0;
    conv_param.dilation_h = 1;
    conv_param.dilation_w = 1;
    conv_param.group = 1;

    feather::operators::Conv2dOp conv("conv_nchw3x3", conv_param);
    ASSERT_EQ(conv.CheckShape(), 0);
    ASSERT_EQ(conv.InferOutputShapes(), 0);
    auto conv_kernel = feather::KernelDispatcher::instance().create(
        feather::DeviceType::X86, feather::DataType::FP32, "Conv2D");
    ASSERT_NE(conv_kernel, nullptr);
    conv.AttachKernel(std::move(conv_kernel));
    ASSERT_EQ(conv.Run(), 0);

    EXPECT_EQ(conv.outputs()[0]->dims().data(), std::vector<int64_t>({1, 1, 2, 2}));
    const std::vector<float> expected = {-6, -6, -6, -6};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(conv.outputs()[0]->data<float>()[i], expected[i]);
    }
}

TEST(conv_relu_test, Conv2dPreservesInt8OutputMetadataWhenGrowingOutput) {
    auto input = std::make_shared<Tensor>();
    input->Assign<int8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 1, 3, 3});
    input->set_layout(feather::DataLayout::NCHW);

    auto weight = std::make_shared<Tensor>();
    weight->Assign<int8_t>({1}, {1, 1, 1, 1});

    auto bias = std::make_shared<Tensor>();
    bias->Assign<int32_t>({0}, {1});

    // Deliberately provide a too-small INT8 buffer so InferOutputShapes must
    // allocate a replacement instead of merely resizing the existing Tensor.
    auto output = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 1, 1});
    output->set_data_type(feather::DataType::INT8);
    output->set_layout(feather::DataLayout::NCHW);
    feather::QuantizationParams quantization;
    quantization.enabled = true;
    quantization.scale = 0.25f;
    quantization.zero_point = -7;
    output->set_quantization(quantization);

    feather::operators::Conv2dParam param{};
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

    feather::operators::Conv2dOp op("conv_int8_metadata", param);
    ASSERT_EQ(op.CheckShape(), 0);
    ASSERT_EQ(op.InferOutputShapes(), 0);

    // Shape inference must preserve the graph-owned Tensor handle and only
    // replace its backing storage when the existing allocation is too small.
    ASSERT_EQ(op.outputs().front(), output);
    EXPECT_EQ(op.outputs().front()->data_type(), feather::DataType::INT8);
    EXPECT_EQ(op.outputs().front()->layout(), feather::DataLayout::NCHW);
    EXPECT_TRUE(op.outputs().front()->quantization().enabled);
    EXPECT_FLOAT_EQ(op.outputs().front()->quantization().scale, 0.25f);
    EXPECT_EQ(op.outputs().front()->quantization().zero_point, -7);
    EXPECT_EQ(op.outputs().front()->dims().data(), std::vector<int64_t>({1, 1, 3, 3}));
    EXPECT_EQ(op.outputs().front()->memory_size(), 9U);
}

TEST(conv_relu_test, Conv2dRunsFastPathOnPointwise1x1) {
    auto input = std::make_shared<Tensor>();
    input->Assign<float>({
        1, 2,
        3, 4,

        10, 20,
        30, 40,
    }, {1, 2, 2, 2});

    auto weight = std::make_shared<Tensor>();
    weight->Assign<float>({
        1, 2,
        3, 4,
    }, {2, 2, 1, 1});

    auto bias = std::make_shared<Tensor>();
    bias->Assign<float>({0.5f, -1.0f}, {2});

    auto conv_out = std::make_shared<Tensor>(std::vector<int64_t>{1, 2, 2, 2});

    feather::operators::Conv2dParam conv_param{};
    conv_param.input = input;
    conv_param.w = weight;
    conv_param.bias = bias;
    conv_param.out = conv_out;
    conv_param.stride_h = 1;
    conv_param.stride_w = 1;
    conv_param.pad_h = 0;
    conv_param.pad_w = 0;
    conv_param.dilation_h = 1;
    conv_param.dilation_w = 1;
    conv_param.group = 1;

    feather::operators::Conv2dOp conv("conv_pointwise", conv_param);
    ASSERT_EQ(conv.CheckShape(), 0);
    ASSERT_EQ(conv.InferOutputShapes(), 0);
    auto conv_kernel = feather::KernelDispatcher::instance().create(
        feather::DeviceType::X86, feather::DataType::FP32, "Conv2D");
    ASSERT_NE(conv_kernel, nullptr);
    conv.AttachKernel(std::move(conv_kernel));
    ASSERT_EQ(conv.Run(), 0);

    EXPECT_EQ(conv.outputs()[0]->dims().data(), std::vector<int64_t>({1, 2, 2, 2}));
    const std::vector<float> expected = {
        21.5f, 42.5f,
        63.5f, 84.5f,
        42.0f, 85.0f,
        128.0f, 171.0f,
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(conv.outputs()[0]->data<float>()[i], expected[i]);
    }
}

TEST(conv_relu_test, Conv2dRunsOnNhwcLayoutOnX86) {
    auto input = std::make_shared<Tensor>();
    input->Assign<float>({
        1, 10,
        2, 20,
        3, 30,
        4, 40,
    }, {1, 2, 2, 2});
    input->set_layout(feather::DataLayout::NHWC);

    auto weight = std::make_shared<Tensor>();
    weight->Assign<float>({
        1, 2,
        3, 4,
    }, {2, 2, 1, 1});

    auto bias = std::make_shared<Tensor>();
    bias->Assign<float>({0.5f, -1.0f}, {2});

    auto conv_out = std::make_shared<Tensor>(std::vector<int64_t>{1, 2, 2, 2});
    conv_out->set_layout(feather::DataLayout::NHWC);

    feather::operators::Conv2dParam conv_param{};
    conv_param.input = input;
    conv_param.w = weight;
    conv_param.bias = bias;
    conv_param.out = conv_out;
    conv_param.stride_h = 1;
    conv_param.stride_w = 1;
    conv_param.pad_h = 0;
    conv_param.pad_w = 0;
    conv_param.dilation_h = 1;
    conv_param.dilation_w = 1;
    conv_param.group = 1;

    feather::operators::Conv2dOp conv("conv_nhwc", conv_param);
    ASSERT_EQ(conv.CheckShape(), 0);
    ASSERT_EQ(conv.InferOutputShapes(), 0);
    auto conv_kernel = feather::CreateKernelForTensor(feather::DeviceType::X86, "Conv2D",
                                                      {input, weight, bias, conv.outputs()[0]},
                                                      feather::DataType::FP32);
    ASSERT_NE(conv_kernel, nullptr);
    conv.AttachKernel(std::move(conv_kernel));
    ASSERT_EQ(conv.Run(), 0);

    EXPECT_EQ(conv.outputs()[0]->dims().data(), std::vector<int64_t>({1, 2, 2, 2}));
    EXPECT_EQ(conv.outputs()[0]->layout(), feather::DataLayout::NHWC);
    const std::vector<float> expected = {
        21.5f, 42.0f,
        42.5f, 85.0f,
        63.5f, 128.0f,
        84.5f, 171.0f,
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(conv.outputs()[0]->data<float>()[i], expected[i]);
    }
}

TEST(conv_relu_test, Conv2dRunsOnNhwcLayoutOnCommon) {
    auto input = std::make_shared<Tensor>();
    input->Assign<float>({
        1, 10,
        2, 20,
        3, 30,
        4, 40,
    }, {1, 2, 2, 2});
    input->set_layout(feather::DataLayout::NHWC);

    auto weight = std::make_shared<Tensor>();
    weight->Assign<float>({
        1, 2,
        3, 4,
    }, {2, 2, 1, 1});

    auto bias = std::make_shared<Tensor>();
    bias->Assign<float>({0.5f, -1.0f}, {2});

    auto conv_out = std::make_shared<Tensor>(std::vector<int64_t>{1, 2, 2, 2});
    conv_out->set_layout(feather::DataLayout::NHWC);

    feather::operators::Conv2dParam conv_param{};
    conv_param.input = input;
    conv_param.w = weight;
    conv_param.bias = bias;
    conv_param.out = conv_out;
    conv_param.stride_h = 1;
    conv_param.stride_w = 1;
    conv_param.pad_h = 0;
    conv_param.pad_w = 0;
    conv_param.dilation_h = 1;
    conv_param.dilation_w = 1;
    conv_param.group = 1;

    feather::operators::Conv2dOp conv("conv_nhwc_common", conv_param);
    ASSERT_EQ(conv.CheckShape(), 0);
    ASSERT_EQ(conv.InferOutputShapes(), 0);
    auto conv_kernel = feather::CreateKernelForTensor(feather::DeviceType::COMMON, "Conv2D",
                                                      {input, weight, bias, conv.outputs()[0]},
                                                      feather::DataType::FP32);
    ASSERT_NE(conv_kernel, nullptr);
    conv.AttachKernel(std::move(conv_kernel));
    ASSERT_EQ(conv.Run(), 0);

    EXPECT_EQ(conv.outputs()[0]->dims().data(), std::vector<int64_t>({1, 2, 2, 2}));
    EXPECT_EQ(conv.outputs()[0]->layout(), feather::DataLayout::NHWC);
    const std::vector<float> expected = {
        21.5f, 42.0f,
        42.5f, 85.0f,
        63.5f, 128.0f,
        84.5f, 171.0f,
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(conv.outputs()[0]->data<float>()[i], expected[i]);
    }
}

TEST(conv_relu_test, Conv2dRunsOnNhwcLayoutOnX86Fp16) {
    auto input = std::make_shared<Tensor>();
    input->Assign<uint16_t>({
        feather::FloatToHalf(1.0f), feather::FloatToHalf(10.0f),
        feather::FloatToHalf(2.0f), feather::FloatToHalf(20.0f),
        feather::FloatToHalf(3.0f), feather::FloatToHalf(30.0f),
        feather::FloatToHalf(4.0f), feather::FloatToHalf(40.0f),
    }, {1, 2, 2, 2});
    input->set_layout(feather::DataLayout::NHWC);

    auto weight = std::make_shared<Tensor>();
    weight->Assign<uint16_t>({
        feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f),
        feather::FloatToHalf(3.0f), feather::FloatToHalf(4.0f),
    }, {2, 2, 1, 1});

    auto bias = std::make_shared<Tensor>();
    bias->Assign<uint16_t>({feather::FloatToHalf(0.5f), feather::FloatToHalf(-1.0f)}, {2});

    auto conv_out = std::make_shared<Tensor>(std::vector<int64_t>{1, 2, 2, 2});
    conv_out->set_layout(feather::DataLayout::NHWC);
    conv_out->mutable_data<uint16_t>();

    feather::operators::Conv2dParam conv_param{};
    conv_param.input = input;
    conv_param.w = weight;
    conv_param.bias = bias;
    conv_param.out = conv_out;
    conv_param.stride_h = 1;
    conv_param.stride_w = 1;
    conv_param.pad_h = 0;
    conv_param.pad_w = 0;
    conv_param.dilation_h = 1;
    conv_param.dilation_w = 1;
    conv_param.group = 1;

    feather::operators::Conv2dOp conv("conv_nhwc_fp16", conv_param);
    ASSERT_EQ(conv.CheckShape(), 0);
    ASSERT_EQ(conv.InferOutputShapes(), 0);
    auto conv_kernel = feather::CreateKernelForTensor(feather::DeviceType::X86, "Conv2D",
                                                      {input, weight, bias, conv.outputs()[0]},
                                                      feather::DataType::FP16);
    ASSERT_NE(conv_kernel, nullptr);
    conv.AttachKernel(std::move(conv_kernel));
    ASSERT_EQ(conv.Run(), 0);

    EXPECT_EQ(conv.outputs()[0]->dims().data(), std::vector<int64_t>({1, 2, 2, 2}));
    EXPECT_EQ(conv.outputs()[0]->layout(), feather::DataLayout::NHWC);
    const std::vector<float> expected = {
        21.5f, 42.0f,
        42.5f, 85.0f,
        63.5f, 128.0f,
        84.5f, 171.0f,
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(conv.outputs()[0]->data<uint16_t>()[i]), expected[i], 1e-2f);
    }
}

TEST(conv_relu_test, Conv2dRuns3x3OnNhwcLayoutOnX86) {
    const std::vector<float> input_data = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };
    auto input = std::make_shared<Tensor>();
    input->Assign<float>(input_data, {1, 3, 3, 1});
    input->set_layout(feather::DataLayout::NHWC);

    const std::vector<float> weight_data = {
        1, 0, -1,
        1, 0, -1,
        1, 0, -1,
        -1, 0, 1,
        -1, 0, 1,
        -1, 0, 1,
    };
    auto weight = std::make_shared<Tensor>();
    weight->Assign<float>(weight_data, {2, 1, 3, 3});

    const std::vector<float> bias_data = {0.5f, -0.5f};
    auto bias = std::make_shared<Tensor>();
    bias->Assign<float>(bias_data, {2});

    auto conv_out = std::make_shared<Tensor>(std::vector<int64_t>{1, 3, 3, 2});
    conv_out->set_layout(feather::DataLayout::NHWC);

    feather::operators::Conv2dParam conv_param{};
    conv_param.input = input;
    conv_param.w = weight;
    conv_param.bias = bias;
    conv_param.out = conv_out;
    conv_param.stride_h = 1;
    conv_param.stride_w = 1;
    conv_param.pad_h = 1;
    conv_param.pad_w = 1;
    conv_param.dilation_h = 1;
    conv_param.dilation_w = 1;
    conv_param.group = 1;

    feather::operators::Conv2dOp conv("conv_nhwc_3x3_x86", conv_param);
    ASSERT_EQ(conv.CheckShape(), 0);
    ASSERT_EQ(conv.InferOutputShapes(), 0);
    auto conv_kernel = feather::CreateKernelForTensor(feather::DeviceType::X86, "Conv2D",
                                                      {input, weight, bias, conv.outputs()[0]},
                                                      feather::DataType::FP32);
    ASSERT_NE(conv_kernel, nullptr);
    conv.AttachKernel(std::move(conv_kernel));
    ASSERT_EQ(conv.Run(), 0);

    const std::vector<float> expected_nchw =
        ReferenceNchwConv(input_data, weight_data, bias_data, 1, 1, 3, 3, 2, 3, 3, 1, 1, 1, 1);
    std::vector<float> expected;
    expected.reserve(expected_nchw.size());
    for (int64_t h = 0; h < 3; ++h) {
        for (int64_t w = 0; w < 3; ++w) {
            expected.push_back(expected_nchw[static_cast<size_t>((0 * 2 + 0) * 3 * 3 + h * 3 + w)]);
            expected.push_back(expected_nchw[static_cast<size_t>((0 * 2 + 1) * 3 * 3 + h * 3 + w)]);
        }
    }
    EXPECT_EQ(conv.outputs()[0]->layout(), feather::DataLayout::NHWC);
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(conv.outputs()[0]->data<float>()[i], expected[i]);
    }
}

TEST(conv_relu_test, Conv2dRuns3x3OnNhwcLayoutOnX86Fp16) {
    const std::vector<float> input_fp32 = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };
    auto input = std::make_shared<Tensor>();
    input->Assign<uint16_t>({
        feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f),
        feather::FloatToHalf(4.0f), feather::FloatToHalf(5.0f), feather::FloatToHalf(6.0f),
        feather::FloatToHalf(7.0f), feather::FloatToHalf(8.0f), feather::FloatToHalf(9.0f),
    }, {1, 3, 3, 1});
    input->set_layout(feather::DataLayout::NHWC);

    const std::vector<float> weight_fp32 = {
        1, 0, -1,
        1, 0, -1,
        1, 0, -1,
        -1, 0, 1,
        -1, 0, 1,
        -1, 0, 1,
    };
    auto weight = std::make_shared<Tensor>();
    weight->Assign<uint16_t>({
        feather::FloatToHalf(1.0f), feather::FloatToHalf(0.0f), feather::FloatToHalf(-1.0f),
        feather::FloatToHalf(1.0f), feather::FloatToHalf(0.0f), feather::FloatToHalf(-1.0f),
        feather::FloatToHalf(1.0f), feather::FloatToHalf(0.0f), feather::FloatToHalf(-1.0f),
        feather::FloatToHalf(-1.0f), feather::FloatToHalf(0.0f), feather::FloatToHalf(1.0f),
        feather::FloatToHalf(-1.0f), feather::FloatToHalf(0.0f), feather::FloatToHalf(1.0f),
        feather::FloatToHalf(-1.0f), feather::FloatToHalf(0.0f), feather::FloatToHalf(1.0f),
    }, {2, 1, 3, 3});

    auto bias = std::make_shared<Tensor>();
    bias->Assign<uint16_t>({feather::FloatToHalf(0.5f), feather::FloatToHalf(-0.5f)}, {2});

    auto conv_out = std::make_shared<Tensor>(std::vector<int64_t>{1, 3, 3, 2});
    conv_out->set_layout(feather::DataLayout::NHWC);
    conv_out->mutable_data<uint16_t>();

    feather::operators::Conv2dParam conv_param{};
    conv_param.input = input;
    conv_param.w = weight;
    conv_param.bias = bias;
    conv_param.out = conv_out;
    conv_param.stride_h = 1;
    conv_param.stride_w = 1;
    conv_param.pad_h = 1;
    conv_param.pad_w = 1;
    conv_param.dilation_h = 1;
    conv_param.dilation_w = 1;
    conv_param.group = 1;

    feather::operators::Conv2dOp conv("conv_nhwc_3x3_x86_fp16", conv_param);
    ASSERT_EQ(conv.CheckShape(), 0);
    ASSERT_EQ(conv.InferOutputShapes(), 0);
    auto conv_kernel = feather::CreateKernelForTensor(feather::DeviceType::X86, "Conv2D",
                                                      {input, weight, bias, conv.outputs()[0]},
                                                      feather::DataType::FP16);
    ASSERT_NE(conv_kernel, nullptr);
    conv.AttachKernel(std::move(conv_kernel));
    ASSERT_EQ(conv.Run(), 0);

    const std::vector<float> expected_nchw =
        ReferenceNchwConv(input_fp32, weight_fp32, {0.5f, -0.5f}, 1, 1, 3, 3, 2, 3, 3, 1, 1, 1, 1);
    std::vector<float> expected;
    expected.reserve(expected_nchw.size());
    for (int64_t h = 0; h < 3; ++h) {
        for (int64_t w = 0; w < 3; ++w) {
            expected.push_back(expected_nchw[static_cast<size_t>((0 * 2 + 0) * 3 * 3 + h * 3 + w)]);
            expected.push_back(expected_nchw[static_cast<size_t>((0 * 2 + 1) * 3 * 3 + h * 3 + w)]);
        }
    }
    EXPECT_EQ(conv.outputs()[0]->layout(), feather::DataLayout::NHWC);
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(conv.outputs()[0]->data<uint16_t>()[i]), expected[i], 1e-2f);
    }
}

TEST(conv_relu_test, Conv2dRunsFastPathOnNchw3x3Oc8FP32) {
    const int64_t batch = 1;
    const int64_t in_c = 2;
    const int64_t in_h = 4;
    const int64_t in_w = 4;
    const int64_t out_c = 8;

    auto input = std::make_shared<Tensor>();
    std::vector<float> input_data(static_cast<size_t>(batch * in_c * in_h * in_w));
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = static_cast<float>((static_cast<int>(i % 11) - 5)) * 0.5f;
    }
    input->Assign<float>(input_data, {batch, in_c, in_h, in_w});

    auto weight = std::make_shared<Tensor>();
    std::vector<float> weight_data(static_cast<size_t>(out_c * in_c * 3 * 3));
    for (size_t i = 0; i < weight_data.size(); ++i) {
        weight_data[i] = static_cast<float>((static_cast<int>((i * 7) % 13) - 6)) * 0.25f;
    }
    weight->Assign<float>(weight_data, {out_c, in_c, 3, 3});

    auto bias = std::make_shared<Tensor>();
    std::vector<float> bias_data(static_cast<size_t>(out_c));
    for (size_t i = 0; i < bias_data.size(); ++i) {
        bias_data[i] = static_cast<float>(static_cast<int>(i) - 3) * 0.25f;
    }
    bias->Assign<float>(bias_data, {out_c});

    auto conv_out = std::make_shared<Tensor>(std::vector<int64_t>{batch, out_c, in_h, in_w});

    feather::operators::Conv2dParam conv_param{};
    conv_param.input = input;
    conv_param.w = weight;
    conv_param.bias = bias;
    conv_param.out = conv_out;
    conv_param.stride_h = 1;
    conv_param.stride_w = 1;
    conv_param.pad_h = 1;
    conv_param.pad_w = 1;
    conv_param.dilation_h = 1;
    conv_param.dilation_w = 1;
    conv_param.group = 1;

    feather::operators::Conv2dOp conv("conv_nchw3x3_oc8_fp32", conv_param);
    ASSERT_EQ(conv.CheckShape(), 0);
    ASSERT_EQ(conv.InferOutputShapes(), 0);
    auto conv_kernel =
        feather::KernelDispatcher::instance().create(feather::DeviceType::X86, feather::DataType::FP32, "Conv2D");
    ASSERT_NE(conv_kernel, nullptr);
    conv.AttachKernel(std::move(conv_kernel));
    ASSERT_EQ(conv.Run(), 0);

    const std::vector<float> expected =
        ReferenceNchwConv(input_data, weight_data, bias_data, batch, in_c, in_h, in_w, out_c, 3, 3, 1, 1, 1, 1);
    ASSERT_EQ(conv.outputs()[0]->dims().data(), std::vector<int64_t>({batch, out_c, in_h, in_w}));
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(conv.outputs()[0]->data<float>()[i], expected[i], 1e-5f);
    }
}

TEST(conv_relu_test, Conv2dRunsFastPathOnNchw3x3Oc8FP16) {
    const int64_t batch = 1;
    const int64_t in_c = 2;
    const int64_t in_h = 4;
    const int64_t in_w = 4;
    const int64_t out_c = 8;

    std::vector<float> input_fp32(static_cast<size_t>(batch * in_c * in_h * in_w));
    for (size_t i = 0; i < input_fp32.size(); ++i) {
        input_fp32[i] = static_cast<float>((static_cast<int>(i % 11) - 5)) * 0.5f;
    }
    auto input = std::make_shared<Tensor>();
    std::vector<uint16_t> input_data(input_fp32.size());
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = feather::FloatToHalf(input_fp32[i]);
    }
    input->Assign<uint16_t>(input_data, {batch, in_c, in_h, in_w});

    std::vector<float> weight_fp32(static_cast<size_t>(out_c * in_c * 3 * 3));
    for (size_t i = 0; i < weight_fp32.size(); ++i) {
        weight_fp32[i] = static_cast<float>((static_cast<int>((i * 7) % 13) - 6)) * 0.25f;
    }
    auto weight = std::make_shared<Tensor>();
    std::vector<uint16_t> weight_data(weight_fp32.size());
    for (size_t i = 0; i < weight_data.size(); ++i) {
        weight_data[i] = feather::FloatToHalf(weight_fp32[i]);
    }
    weight->Assign<uint16_t>(weight_data, {out_c, in_c, 3, 3});

    std::vector<float> bias_fp32(static_cast<size_t>(out_c));
    for (size_t i = 0; i < bias_fp32.size(); ++i) {
        bias_fp32[i] = static_cast<float>(static_cast<int>(i) - 3) * 0.25f;
    }
    auto bias = std::make_shared<Tensor>();
    std::vector<uint16_t> bias_data(bias_fp32.size());
    for (size_t i = 0; i < bias_data.size(); ++i) {
        bias_data[i] = feather::FloatToHalf(bias_fp32[i]);
    }
    bias->Assign<uint16_t>(bias_data, {out_c});

    auto conv_out = std::make_shared<Tensor>(std::vector<int64_t>{batch, out_c, in_h, in_w});
    conv_out->mutable_data<uint16_t>();

    feather::operators::Conv2dParam conv_param{};
    conv_param.input = input;
    conv_param.w = weight;
    conv_param.bias = bias;
    conv_param.out = conv_out;
    conv_param.stride_h = 1;
    conv_param.stride_w = 1;
    conv_param.pad_h = 1;
    conv_param.pad_w = 1;
    conv_param.dilation_h = 1;
    conv_param.dilation_w = 1;
    conv_param.group = 1;

    feather::operators::Conv2dOp conv("conv_nchw3x3_oc8_fp16", conv_param);
    ASSERT_EQ(conv.CheckShape(), 0);
    ASSERT_EQ(conv.InferOutputShapes(), 0);
    auto conv_kernel =
        feather::KernelDispatcher::instance().create(feather::DeviceType::X86, feather::DataType::FP16, "Conv2D");
    ASSERT_NE(conv_kernel, nullptr);
    conv.AttachKernel(std::move(conv_kernel));
    ASSERT_EQ(conv.Run(), 0);

    const std::vector<float> expected =
        ReferenceNchwConv(input_fp32, weight_fp32, bias_fp32, batch, in_c, in_h, in_w, out_c, 3, 3, 1, 1, 1, 1);
    ASSERT_EQ(conv.outputs()[0]->dims().data(), std::vector<int64_t>({batch, out_c, in_h, in_w}));
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(conv.outputs()[0]->data<uint16_t>()[i]), expected[i], 5e-2f);
    }
}

TEST(conv_relu_test, Conv2dRunsFastPathOnNchw3x3FP16) {
    auto input = std::make_shared<Tensor>();
    input->Assign<uint16_t>({
        feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f), feather::FloatToHalf(4.0f),
        feather::FloatToHalf(5.0f), feather::FloatToHalf(6.0f), feather::FloatToHalf(7.0f), feather::FloatToHalf(8.0f),
        feather::FloatToHalf(9.0f), feather::FloatToHalf(10.0f), feather::FloatToHalf(11.0f), feather::FloatToHalf(12.0f),
        feather::FloatToHalf(13.0f), feather::FloatToHalf(14.0f), feather::FloatToHalf(15.0f), feather::FloatToHalf(16.0f),
    }, {1, 1, 4, 4});

    auto weight = std::make_shared<Tensor>();
    weight->Assign<uint16_t>({
        feather::FloatToHalf(1.0f), feather::FloatToHalf(0.0f), feather::FloatToHalf(-1.0f),
        feather::FloatToHalf(1.0f), feather::FloatToHalf(0.0f), feather::FloatToHalf(-1.0f),
        feather::FloatToHalf(1.0f), feather::FloatToHalf(0.0f), feather::FloatToHalf(-1.0f),
    }, {1, 1, 3, 3});

    auto bias = std::make_shared<Tensor>();
    bias->Assign<uint16_t>({feather::FloatToHalf(0.0f)}, {1});

    auto conv_out = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 2, 2});
    conv_out->mutable_data<uint16_t>();

    feather::operators::Conv2dParam conv_param{};
    conv_param.input = input;
    conv_param.w = weight;
    conv_param.bias = bias;
    conv_param.out = conv_out;
    conv_param.stride_h = 1;
    conv_param.stride_w = 1;
    conv_param.pad_h = 0;
    conv_param.pad_w = 0;
    conv_param.dilation_h = 1;
    conv_param.dilation_w = 1;
    conv_param.group = 1;

    feather::operators::Conv2dOp conv("conv_nchw3x3_fp16", conv_param);
    ASSERT_EQ(conv.CheckShape(), 0);
    ASSERT_EQ(conv.InferOutputShapes(), 0);
    auto conv_kernel = feather::KernelDispatcher::instance().create(
        feather::DeviceType::X86, feather::DataType::FP16, "Conv2D");
    ASSERT_NE(conv_kernel, nullptr);
    conv.AttachKernel(std::move(conv_kernel));
    ASSERT_EQ(conv.Run(), 0);

    EXPECT_EQ(conv.outputs()[0]->dims().data(), std::vector<int64_t>({1, 1, 2, 2}));
    const std::vector<float> expected = {-6, -6, -6, -6};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(conv.outputs()[0]->data<uint16_t>()[i]), expected[i], 1e-3f);
    }
}

TEST(conv_relu_test, Conv2dRunsFastPathOnPointwise1x1FP16) {
    auto input = std::make_shared<Tensor>();
    input->Assign<uint16_t>({
        feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f),
        feather::FloatToHalf(3.0f), feather::FloatToHalf(4.0f),

        feather::FloatToHalf(10.0f), feather::FloatToHalf(20.0f),
        feather::FloatToHalf(30.0f), feather::FloatToHalf(40.0f),
    }, {1, 2, 2, 2});

    auto weight = std::make_shared<Tensor>();
    weight->Assign<uint16_t>({
        feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f),
        feather::FloatToHalf(3.0f), feather::FloatToHalf(4.0f),
    }, {2, 2, 1, 1});

    auto bias = std::make_shared<Tensor>();
    bias->Assign<uint16_t>({
        feather::FloatToHalf(0.5f), feather::FloatToHalf(-1.0f),
    }, {2});

    auto conv_out = std::make_shared<Tensor>(std::vector<int64_t>{1, 2, 2, 2});
    conv_out->mutable_data<uint16_t>();

    feather::operators::Conv2dParam conv_param{};
    conv_param.input = input;
    conv_param.w = weight;
    conv_param.bias = bias;
    conv_param.out = conv_out;
    conv_param.stride_h = 1;
    conv_param.stride_w = 1;
    conv_param.pad_h = 0;
    conv_param.pad_w = 0;
    conv_param.dilation_h = 1;
    conv_param.dilation_w = 1;
    conv_param.group = 1;

    feather::operators::Conv2dOp conv("conv_pointwise_fp16", conv_param);
    ASSERT_EQ(conv.CheckShape(), 0);
    ASSERT_EQ(conv.InferOutputShapes(), 0);
    auto conv_kernel = feather::KernelDispatcher::instance().create(
        feather::DeviceType::X86, feather::DataType::FP16, "Conv2D");
    ASSERT_NE(conv_kernel, nullptr);
    conv.AttachKernel(std::move(conv_kernel));
    ASSERT_EQ(conv.Run(), 0);

    EXPECT_EQ(conv.outputs()[0]->dims().data(), std::vector<int64_t>({1, 2, 2, 2}));
    const std::vector<float> expected = {
        21.5f, 42.5f,
        63.5f, 84.5f,
        42.0f, 85.0f,
        128.0f, 171.0f,
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(conv.outputs()[0]->data<uint16_t>()[i]), expected[i], 2e-2f);
    }
}

TEST(conv_relu_test, Conv2dFp16CanRunRepeatedlyWithCachedWeights) {
    auto input = std::make_shared<Tensor>();
    input->Assign<uint16_t>({
        feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f),
        feather::FloatToHalf(3.0f), feather::FloatToHalf(4.0f),

        feather::FloatToHalf(10.0f), feather::FloatToHalf(20.0f),
        feather::FloatToHalf(30.0f), feather::FloatToHalf(40.0f),
    }, {1, 2, 2, 2});

    auto weight = std::make_shared<Tensor>();
    weight->Assign<uint16_t>({
        feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f),
        feather::FloatToHalf(3.0f), feather::FloatToHalf(4.0f),
    }, {2, 2, 1, 1});

    auto bias = std::make_shared<Tensor>();
    bias->Assign<uint16_t>({
        feather::FloatToHalf(0.5f), feather::FloatToHalf(-1.0f),
    }, {2});

    auto conv_out = std::make_shared<Tensor>(std::vector<int64_t>{1, 2, 2, 2});
    conv_out->mutable_data<uint16_t>();

    feather::operators::Conv2dParam conv_param{};
    conv_param.input = input;
    conv_param.w = weight;
    conv_param.bias = bias;
    conv_param.out = conv_out;
    conv_param.stride_h = 1;
    conv_param.stride_w = 1;
    conv_param.pad_h = 0;
    conv_param.pad_w = 0;
    conv_param.dilation_h = 1;
    conv_param.dilation_w = 1;
    conv_param.group = 1;

    feather::operators::Conv2dOp conv("conv_pointwise_fp16_reuse", conv_param);
    ASSERT_EQ(conv.CheckShape(), 0);
    ASSERT_EQ(conv.InferOutputShapes(), 0);
    auto conv_kernel = feather::KernelDispatcher::instance().create(
        feather::DeviceType::X86, feather::DataType::FP16, "Conv2D");
    ASSERT_NE(conv_kernel, nullptr);
    conv.AttachKernel(std::move(conv_kernel));

    ASSERT_EQ(conv.Run(), 0);
    const std::vector<float> first_expected = {
        21.5f, 42.5f,
        63.5f, 84.5f,
        42.0f, 85.0f,
        128.0f, 171.0f,
    };
    for (size_t i = 0; i < first_expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(conv.outputs()[0]->data<uint16_t>()[i]), first_expected[i], 2e-2f);
    }

    input->Assign<uint16_t>({
        feather::FloatToHalf(2.0f), feather::FloatToHalf(1.0f),
        feather::FloatToHalf(0.0f), feather::FloatToHalf(-1.0f),

        feather::FloatToHalf(5.0f), feather::FloatToHalf(6.0f),
        feather::FloatToHalf(7.0f), feather::FloatToHalf(8.0f),
    }, {1, 2, 2, 2});
    ASSERT_EQ(conv.Run(), 0);

    const std::vector<float> second_expected = {
        12.5f, 13.5f,
        14.5f, 15.5f,
        25.0f, 26.0f,
        27.0f, 28.0f,
    };
    for (size_t i = 0; i < second_expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(conv.outputs()[0]->data<uint16_t>()[i]), second_expected[i], 2e-2f);
    }
}

TEST(conv_relu_test, Conv2dRunsPointwiseFp16WithChannelTail) {
    auto input = std::make_shared<Tensor>();
    input->Assign<uint16_t>({
        feather::FloatToHalf(1.0f),  feather::FloatToHalf(11.0f),
        feather::FloatToHalf(2.0f),  feather::FloatToHalf(12.0f),
        feather::FloatToHalf(3.0f),  feather::FloatToHalf(13.0f),
        feather::FloatToHalf(4.0f),  feather::FloatToHalf(14.0f),
        feather::FloatToHalf(5.0f),  feather::FloatToHalf(15.0f),
        feather::FloatToHalf(6.0f),  feather::FloatToHalf(16.0f),
        feather::FloatToHalf(7.0f),  feather::FloatToHalf(17.0f),
        feather::FloatToHalf(8.0f),  feather::FloatToHalf(18.0f),
        feather::FloatToHalf(9.0f),  feather::FloatToHalf(19.0f),
    }, {1, 9, 2, 1});

    auto weight = std::make_shared<Tensor>();
    weight->Assign<uint16_t>({
        feather::FloatToHalf(1.0f),  feather::FloatToHalf(1.0f),  feather::FloatToHalf(1.0f),
        feather::FloatToHalf(1.0f),  feather::FloatToHalf(1.0f),  feather::FloatToHalf(1.0f),
        feather::FloatToHalf(1.0f),  feather::FloatToHalf(1.0f),  feather::FloatToHalf(1.0f),

        feather::FloatToHalf(1.0f),  feather::FloatToHalf(-1.0f), feather::FloatToHalf(1.0f),
        feather::FloatToHalf(-1.0f), feather::FloatToHalf(1.0f),  feather::FloatToHalf(-1.0f),
        feather::FloatToHalf(1.0f),  feather::FloatToHalf(-1.0f), feather::FloatToHalf(1.0f),

        feather::FloatToHalf(0.5f),  feather::FloatToHalf(0.5f),  feather::FloatToHalf(0.5f),
        feather::FloatToHalf(0.5f),  feather::FloatToHalf(0.5f),  feather::FloatToHalf(0.5f),
        feather::FloatToHalf(0.5f),  feather::FloatToHalf(0.5f),  feather::FloatToHalf(0.5f),
    }, {3, 9, 1, 1});

    auto bias = std::make_shared<Tensor>();
    bias->Assign<uint16_t>({
        feather::FloatToHalf(0.5f),
        feather::FloatToHalf(-1.0f),
        feather::FloatToHalf(-1.0f),
    }, {3});

    auto conv_out = std::make_shared<Tensor>(std::vector<int64_t>{1, 3, 2, 1});
    conv_out->mutable_data<uint16_t>();

    feather::operators::Conv2dParam conv_param{};
    conv_param.input = input;
    conv_param.w = weight;
    conv_param.bias = bias;
    conv_param.out = conv_out;
    conv_param.stride_h = 1;
    conv_param.stride_w = 1;
    conv_param.pad_h = 0;
    conv_param.pad_w = 0;
    conv_param.dilation_h = 1;
    conv_param.dilation_w = 1;
    conv_param.group = 1;

    feather::operators::Conv2dOp conv("conv_pointwise_fp16_tail", conv_param);
    ASSERT_EQ(conv.CheckShape(), 0);
    ASSERT_EQ(conv.InferOutputShapes(), 0);
    auto conv_kernel = feather::KernelDispatcher::instance().create(
        feather::DeviceType::X86, feather::DataType::FP16, "Conv2D");
    ASSERT_NE(conv_kernel, nullptr);
    conv.AttachKernel(std::move(conv_kernel));
    ASSERT_EQ(conv.Run(), 0);

    const std::vector<float> expected = {
        45.5f, 135.5f,
        4.0f,  14.0f,
        21.5f, 66.5f,
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(conv.outputs()[0]->data<uint16_t>()[i]), expected[i], 2e-2f);
    }
}

TEST(conv_relu_test, Conv2dRunsDirectStride2Fp16) {
    auto input = std::make_shared<Tensor>();
    input->Assign<uint16_t>({
        feather::FloatToHalf(1.0f),  feather::FloatToHalf(2.0f),  feather::FloatToHalf(3.0f),  feather::FloatToHalf(4.0f),
        feather::FloatToHalf(5.0f),  feather::FloatToHalf(6.0f),  feather::FloatToHalf(7.0f),  feather::FloatToHalf(8.0f),
        feather::FloatToHalf(9.0f),  feather::FloatToHalf(10.0f), feather::FloatToHalf(11.0f), feather::FloatToHalf(12.0f),
        feather::FloatToHalf(13.0f), feather::FloatToHalf(14.0f), feather::FloatToHalf(15.0f), feather::FloatToHalf(16.0f),

        feather::FloatToHalf(1.0f),  feather::FloatToHalf(1.0f),  feather::FloatToHalf(1.0f),  feather::FloatToHalf(1.0f),
        feather::FloatToHalf(2.0f),  feather::FloatToHalf(2.0f),  feather::FloatToHalf(2.0f),  feather::FloatToHalf(2.0f),
        feather::FloatToHalf(3.0f),  feather::FloatToHalf(3.0f),  feather::FloatToHalf(3.0f),  feather::FloatToHalf(3.0f),
        feather::FloatToHalf(4.0f),  feather::FloatToHalf(4.0f),  feather::FloatToHalf(4.0f),  feather::FloatToHalf(4.0f),
    }, {1, 2, 4, 4});

    auto weight = std::make_shared<Tensor>();
    weight->Assign<uint16_t>({
        feather::FloatToHalf(1.0f),  feather::FloatToHalf(0.0f),  feather::FloatToHalf(0.0f),
        feather::FloatToHalf(0.0f),  feather::FloatToHalf(1.0f),  feather::FloatToHalf(0.0f),
        feather::FloatToHalf(0.0f),  feather::FloatToHalf(0.0f),  feather::FloatToHalf(1.0f),

        feather::FloatToHalf(0.0f),  feather::FloatToHalf(1.0f),  feather::FloatToHalf(0.0f),
        feather::FloatToHalf(1.0f),  feather::FloatToHalf(0.0f),  feather::FloatToHalf(1.0f),
        feather::FloatToHalf(0.0f),  feather::FloatToHalf(1.0f),  feather::FloatToHalf(0.0f),
    }, {1, 2, 3, 3});

    auto bias = std::make_shared<Tensor>();
    bias->Assign<uint16_t>({feather::FloatToHalf(0.5f)}, {1});

    auto conv_out = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 2, 2});
    conv_out->mutable_data<uint16_t>();

    feather::operators::Conv2dParam conv_param{};
    conv_param.input = input;
    conv_param.w = weight;
    conv_param.bias = bias;
    conv_param.out = conv_out;
    conv_param.stride_h = 2;
    conv_param.stride_w = 2;
    conv_param.pad_h = 1;
    conv_param.pad_w = 1;
    conv_param.dilation_h = 1;
    conv_param.dilation_w = 1;
    conv_param.group = 1;

    feather::operators::Conv2dOp conv("conv_direct_stride2_fp16", conv_param);
    ASSERT_EQ(conv.CheckShape(), 0);
    ASSERT_EQ(conv.InferOutputShapes(), 0);
    auto conv_kernel = feather::KernelDispatcher::instance().create(
        feather::DeviceType::X86, feather::DataType::FP16, "Conv2D");
    ASSERT_NE(conv_kernel, nullptr);
    conv.AttachKernel(std::move(conv_kernel));
    ASSERT_EQ(conv.Run(), 0);

    const std::vector<float> expected = {10.5f, 15.5f, 32.5f, 45.5f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(conv.outputs()[0]->data<uint16_t>()[i]), expected[i], 2e-2f);
    }
}

TEST(conv_relu_test, Conv2dRunsDirectMultiOutputFp16) {
    auto input = std::make_shared<Tensor>();
    input->Assign<uint16_t>({
        feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f),
        feather::FloatToHalf(4.0f), feather::FloatToHalf(5.0f), feather::FloatToHalf(6.0f),
        feather::FloatToHalf(7.0f), feather::FloatToHalf(8.0f), feather::FloatToHalf(9.0f),
    }, {1, 1, 3, 3});

    auto weight = std::make_shared<Tensor>();
    weight->Assign<uint16_t>({
        feather::FloatToHalf(1.0f), feather::FloatToHalf(0.0f),
        feather::FloatToHalf(0.0f), feather::FloatToHalf(1.0f),

        feather::FloatToHalf(0.0f), feather::FloatToHalf(1.0f),
        feather::FloatToHalf(1.0f), feather::FloatToHalf(0.0f),

        feather::FloatToHalf(1.0f),  feather::FloatToHalf(1.0f),
        feather::FloatToHalf(-1.0f), feather::FloatToHalf(-1.0f),

        feather::FloatToHalf(0.5f), feather::FloatToHalf(0.5f),
        feather::FloatToHalf(0.5f), feather::FloatToHalf(0.5f),
    }, {4, 1, 2, 2});

    auto bias = std::make_shared<Tensor>();
    bias->Assign<uint16_t>({
        feather::FloatToHalf(0.0f),
        feather::FloatToHalf(1.0f),
        feather::FloatToHalf(-1.0f),
        feather::FloatToHalf(2.0f),
    }, {4});

    auto conv_out = std::make_shared<Tensor>(std::vector<int64_t>{1, 4, 2, 2});
    conv_out->mutable_data<uint16_t>();

    feather::operators::Conv2dParam conv_param{};
    conv_param.input = input;
    conv_param.w = weight;
    conv_param.bias = bias;
    conv_param.out = conv_out;
    conv_param.stride_h = 1;
    conv_param.stride_w = 1;
    conv_param.pad_h = 0;
    conv_param.pad_w = 0;
    conv_param.dilation_h = 1;
    conv_param.dilation_w = 1;
    conv_param.group = 1;

    feather::operators::Conv2dOp conv("conv_direct_multi_output_fp16", conv_param);
    ASSERT_EQ(conv.CheckShape(), 0);
    ASSERT_EQ(conv.InferOutputShapes(), 0);
    auto conv_kernel = feather::KernelDispatcher::instance().create(
        feather::DeviceType::X86, feather::DataType::FP16, "Conv2D");
    ASSERT_NE(conv_kernel, nullptr);
    conv.AttachKernel(std::move(conv_kernel));
    ASSERT_EQ(conv.Run(), 0);

    const std::vector<float> expected = {
        6.0f, 8.0f, 12.0f, 14.0f,
        7.0f, 9.0f, 13.0f, 15.0f,
        -7.0f, -7.0f, -7.0f, -7.0f,
        8.0f, 10.0f, 14.0f, 16.0f,
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(conv.outputs()[0]->data<uint16_t>()[i]), expected[i], 2e-2f);
    }
}

TEST(conv_relu_test, Conv2dRunsDirectOutputChannelTailFp16) {
    auto input = std::make_shared<Tensor>();
    input->Assign<uint16_t>({
        feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f),
        feather::FloatToHalf(4.0f), feather::FloatToHalf(5.0f), feather::FloatToHalf(6.0f),
        feather::FloatToHalf(7.0f), feather::FloatToHalf(8.0f), feather::FloatToHalf(9.0f),
    }, {1, 1, 3, 3});

    auto weight = std::make_shared<Tensor>();
    weight->Assign<uint16_t>({
        feather::FloatToHalf(1.0f),  feather::FloatToHalf(0.0f),  feather::FloatToHalf(0.0f),  feather::FloatToHalf(1.0f),
        feather::FloatToHalf(0.0f),  feather::FloatToHalf(1.0f),  feather::FloatToHalf(1.0f),  feather::FloatToHalf(0.0f),
        feather::FloatToHalf(1.0f),  feather::FloatToHalf(1.0f),  feather::FloatToHalf(1.0f),  feather::FloatToHalf(1.0f),
        feather::FloatToHalf(0.5f),  feather::FloatToHalf(0.5f),  feather::FloatToHalf(0.5f),  feather::FloatToHalf(0.5f),
        feather::FloatToHalf(-1.0f), feather::FloatToHalf(0.0f),  feather::FloatToHalf(0.0f),  feather::FloatToHalf(1.0f),
        feather::FloatToHalf(1.0f),  feather::FloatToHalf(-1.0f), feather::FloatToHalf(-1.0f), feather::FloatToHalf(1.0f),
        feather::FloatToHalf(1.0f),  feather::FloatToHalf(2.0f),  feather::FloatToHalf(3.0f),  feather::FloatToHalf(4.0f),
        feather::FloatToHalf(0.0f),  feather::FloatToHalf(0.0f),  feather::FloatToHalf(0.0f),  feather::FloatToHalf(0.0f),
        feather::FloatToHalf(1.0f),  feather::FloatToHalf(0.0f),  feather::FloatToHalf(1.0f),  feather::FloatToHalf(0.0f),
    }, {9, 1, 2, 2});

    auto bias = std::make_shared<Tensor>();
    bias->Assign<uint16_t>({
        feather::FloatToHalf(0.0f),
        feather::FloatToHalf(1.0f),
        feather::FloatToHalf(0.0f),
        feather::FloatToHalf(2.0f),
        feather::FloatToHalf(0.0f),
        feather::FloatToHalf(0.0f),
        feather::FloatToHalf(0.0f),
        feather::FloatToHalf(3.0f),
        feather::FloatToHalf(-1.0f),
    }, {9});

    auto conv_out = std::make_shared<Tensor>(std::vector<int64_t>{1, 9, 2, 2});
    conv_out->mutable_data<uint16_t>();

    feather::operators::Conv2dParam conv_param{};
    conv_param.input = input;
    conv_param.w = weight;
    conv_param.bias = bias;
    conv_param.out = conv_out;
    conv_param.stride_h = 1;
    conv_param.stride_w = 1;
    conv_param.pad_h = 0;
    conv_param.pad_w = 0;
    conv_param.dilation_h = 1;
    conv_param.dilation_w = 1;
    conv_param.group = 1;

    feather::operators::Conv2dOp conv("conv_direct_output_tail_fp16", conv_param);
    ASSERT_EQ(conv.CheckShape(), 0);
    ASSERT_EQ(conv.InferOutputShapes(), 0);
    auto conv_kernel = feather::KernelDispatcher::instance().create(
        feather::DeviceType::X86, feather::DataType::FP16, "Conv2D");
    ASSERT_NE(conv_kernel, nullptr);
    conv.AttachKernel(std::move(conv_kernel));
    ASSERT_EQ(conv.Run(), 0);

    const std::vector<float> expected = {
        6.0f,  8.0f,  12.0f, 14.0f,
        7.0f,  9.0f,  13.0f, 15.0f,
        12.0f, 16.0f, 24.0f, 28.0f,
        8.0f,  10.0f, 14.0f, 16.0f,
        4.0f,  4.0f,  4.0f,  4.0f,
        0.0f,  0.0f,  0.0f,  0.0f,
        37.0f, 47.0f, 67.0f, 77.0f,
        3.0f,  3.0f,  3.0f,  3.0f,
        4.0f,  6.0f,  10.0f, 12.0f,
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(conv.outputs()[0]->data<uint16_t>()[i]), expected[i], 2e-2f);
    }
}

TEST(conv_relu_test, Conv2dRunsDirectOc8MultiInputFp16) {
    auto input = std::make_shared<Tensor>();
    input->Assign<uint16_t>({
        feather::FloatToHalf(1.0f),  feather::FloatToHalf(2.0f),  feather::FloatToHalf(3.0f),
        feather::FloatToHalf(4.0f),  feather::FloatToHalf(5.0f),  feather::FloatToHalf(6.0f),
        feather::FloatToHalf(7.0f),  feather::FloatToHalf(8.0f),  feather::FloatToHalf(9.0f),

        feather::FloatToHalf(10.0f), feather::FloatToHalf(11.0f), feather::FloatToHalf(12.0f),
        feather::FloatToHalf(13.0f), feather::FloatToHalf(14.0f), feather::FloatToHalf(15.0f),
        feather::FloatToHalf(16.0f), feather::FloatToHalf(17.0f), feather::FloatToHalf(18.0f),
    }, {1, 2, 3, 3});

    auto weight = std::make_shared<Tensor>();
    weight->Assign<uint16_t>({
        feather::FloatToHalf(1.0f),  feather::FloatToHalf(0.0f),  feather::FloatToHalf(0.0f),  feather::FloatToHalf(1.0f),
        feather::FloatToHalf(0.0f),  feather::FloatToHalf(1.0f),  feather::FloatToHalf(1.0f),  feather::FloatToHalf(0.0f),

        feather::FloatToHalf(0.0f),  feather::FloatToHalf(1.0f),  feather::FloatToHalf(1.0f),  feather::FloatToHalf(0.0f),
        feather::FloatToHalf(1.0f),  feather::FloatToHalf(0.0f),  feather::FloatToHalf(0.0f),  feather::FloatToHalf(1.0f),

        feather::FloatToHalf(1.0f),  feather::FloatToHalf(1.0f),  feather::FloatToHalf(1.0f),  feather::FloatToHalf(1.0f),
        feather::FloatToHalf(-1.0f), feather::FloatToHalf(-1.0f), feather::FloatToHalf(-1.0f), feather::FloatToHalf(-1.0f),

        feather::FloatToHalf(0.5f),  feather::FloatToHalf(0.5f),  feather::FloatToHalf(0.5f),  feather::FloatToHalf(0.5f),
        feather::FloatToHalf(0.25f), feather::FloatToHalf(0.25f), feather::FloatToHalf(0.25f), feather::FloatToHalf(0.25f),

        feather::FloatToHalf(-1.0f), feather::FloatToHalf(0.0f),  feather::FloatToHalf(0.0f),  feather::FloatToHalf(1.0f),
        feather::FloatToHalf(1.0f),  feather::FloatToHalf(0.0f),  feather::FloatToHalf(0.0f),  feather::FloatToHalf(-1.0f),

        feather::FloatToHalf(1.0f),  feather::FloatToHalf(-1.0f), feather::FloatToHalf(-1.0f), feather::FloatToHalf(1.0f),
        feather::FloatToHalf(-1.0f), feather::FloatToHalf(1.0f),  feather::FloatToHalf(1.0f),  feather::FloatToHalf(-1.0f),

        feather::FloatToHalf(1.0f),  feather::FloatToHalf(2.0f),  feather::FloatToHalf(3.0f),  feather::FloatToHalf(4.0f),
        feather::FloatToHalf(-4.0f), feather::FloatToHalf(-3.0f), feather::FloatToHalf(-2.0f), feather::FloatToHalf(-1.0f),

        feather::FloatToHalf(0.0f),  feather::FloatToHalf(0.0f),  feather::FloatToHalf(0.0f),  feather::FloatToHalf(0.0f),
        feather::FloatToHalf(1.0f),  feather::FloatToHalf(1.0f),  feather::FloatToHalf(1.0f),  feather::FloatToHalf(1.0f),
    }, {8, 2, 2, 2});

    auto bias = std::make_shared<Tensor>();
    bias->Assign<uint16_t>({
        feather::FloatToHalf(0.0f),
        feather::FloatToHalf(1.0f),
        feather::FloatToHalf(-1.0f),
        feather::FloatToHalf(2.0f),
        feather::FloatToHalf(0.5f),
        feather::FloatToHalf(-0.5f),
        feather::FloatToHalf(3.0f),
        feather::FloatToHalf(4.0f),
    }, {8});

    auto conv_out = std::make_shared<Tensor>(std::vector<int64_t>{1, 8, 2, 2});
    conv_out->mutable_data<uint16_t>();

    feather::operators::Conv2dParam conv_param{};
    conv_param.input = input;
    conv_param.w = weight;
    conv_param.bias = bias;
    conv_param.out = conv_out;
    conv_param.stride_h = 1;
    conv_param.stride_w = 1;
    conv_param.pad_h = 0;
    conv_param.pad_w = 0;
    conv_param.dilation_h = 1;
    conv_param.dilation_w = 1;
    conv_param.group = 1;

    feather::operators::Conv2dOp conv("conv_direct_oc8_multi_input_fp16", conv_param);
    ASSERT_EQ(conv.CheckShape(), 0);
    ASSERT_EQ(conv.InferOutputShapes(), 0);
    auto conv_kernel = feather::KernelDispatcher::instance().create(
        feather::DeviceType::X86, feather::DataType::FP16, "Conv2D");
    ASSERT_NE(conv_kernel, nullptr);
    conv.AttachKernel(std::move(conv_kernel));
    ASSERT_EQ(conv.Run(), 0);

    const std::vector<float> expected = {
        30.0f, 34.0f, 42.0f, 46.0f,
        31.0f, 35.0f, 43.0f, 47.0f,
        -37.0f, -37.0f, -37.0f, -37.0f,
        20.0f, 23.0f, 29.0f, 32.0f,
        0.5f, 0.5f, 0.5f, 0.5f,
        -0.5f, -0.5f, -0.5f, -0.5f,
        -73.0f, -73.0f, -73.0f, -73.0f,
        52.0f, 56.0f, 64.0f, 68.0f,
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(conv.outputs()[0]->data<uint16_t>()[i]), expected[i], 2e-2f);
    }
}

TEST(conv_relu_test, Conv2dRunsPointwiseOc8Fp16) {
    auto input = std::make_shared<Tensor>();
    input->Assign<uint16_t>({
        feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f),
        feather::FloatToHalf(3.0f), feather::FloatToHalf(4.0f),

        feather::FloatToHalf(10.0f), feather::FloatToHalf(20.0f),
        feather::FloatToHalf(30.0f), feather::FloatToHalf(40.0f),
    }, {1, 2, 2, 2});

    auto weight = std::make_shared<Tensor>();
    weight->Assign<uint16_t>({
        feather::FloatToHalf(1.0f),  feather::FloatToHalf(2.0f),
        feather::FloatToHalf(3.0f),  feather::FloatToHalf(4.0f),
        feather::FloatToHalf(0.5f),  feather::FloatToHalf(-1.0f),
        feather::FloatToHalf(-2.0f), feather::FloatToHalf(0.25f),
        feather::FloatToHalf(1.0f),  feather::FloatToHalf(0.0f),
        feather::FloatToHalf(0.0f),  feather::FloatToHalf(1.0f),
        feather::FloatToHalf(1.5f),  feather::FloatToHalf(1.5f),
        feather::FloatToHalf(-1.0f), feather::FloatToHalf(-1.0f),
    }, {8, 2, 1, 1});

    auto bias = std::make_shared<Tensor>();
    bias->Assign<uint16_t>({
        feather::FloatToHalf(0.5f),
        feather::FloatToHalf(-1.0f),
        feather::FloatToHalf(2.0f),
        feather::FloatToHalf(-3.0f),
        feather::FloatToHalf(4.0f),
        feather::FloatToHalf(5.0f),
        feather::FloatToHalf(0.0f),
        feather::FloatToHalf(-2.0f),
    }, {8});

    auto conv_out = std::make_shared<Tensor>(std::vector<int64_t>{1, 8, 2, 2});
    conv_out->mutable_data<uint16_t>();

    feather::operators::Conv2dParam conv_param{};
    conv_param.input = input;
    conv_param.w = weight;
    conv_param.bias = bias;
    conv_param.out = conv_out;
    conv_param.stride_h = 1;
    conv_param.stride_w = 1;
    conv_param.pad_h = 0;
    conv_param.pad_w = 0;
    conv_param.dilation_h = 1;
    conv_param.dilation_w = 1;
    conv_param.group = 1;

    feather::operators::Conv2dOp conv("conv_pointwise_oc8_fp16", conv_param);
    ASSERT_EQ(conv.CheckShape(), 0);
    ASSERT_EQ(conv.InferOutputShapes(), 0);
    auto conv_kernel = feather::KernelDispatcher::instance().create(
        feather::DeviceType::X86, feather::DataType::FP16, "Conv2D");
    ASSERT_NE(conv_kernel, nullptr);
    conv.AttachKernel(std::move(conv_kernel));
    ASSERT_EQ(conv.Run(), 0);

    const std::vector<float> expected = {
        21.5f, 42.5f, 63.5f, 84.5f,
        42.0f, 85.0f, 128.0f, 171.0f,
        -7.5f, -17.0f, -26.5f, -36.0f,
        -2.5f, -2.0f, -1.5f, -1.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        15.0f, 25.0f, 35.0f, 45.0f,
        16.5f, 33.0f, 49.5f, 66.0f,
        -13.0f, -24.0f, -35.0f, -46.0f,
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(conv.outputs()[0]->data<uint16_t>()[i]), expected[i], 2e-2f);
    }
}
