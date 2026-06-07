#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/kernel.h"
#include "src/operator/conv2d_op.h"
#include "src/operator/relu_op.h"
#include "util/fp16.h"

using feather::Tensor;

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
