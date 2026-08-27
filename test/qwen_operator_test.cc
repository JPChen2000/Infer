#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#ifdef FEATHER_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include "core/kernel.h"
#include "core/operator.h"
#include "core/tensor.h"
#include "src/kernel/conv2d.h"
#include "src/kernel/qwen_rms_norm.h"
#include "src/operator/conv2d_op.h"
#include "src/operator/cos_op.h"
#include "src/operator/exp_op.h"
#include "src/operator/neg_op.h"
#include "src/operator/params.h"
#include "src/operator/qwen_depthwise_conv_op.h"
#include "src/operator/reduce_sum_op.h"
#include "src/operator/sin_op.h"
#include "src/operator/softplus_op.h"
#include "util/bf16.h"
#include "util/fp8.h"
#include "util/fp16.h"

namespace {

using feather::BFloat16;
using feather::DataType;
using feather::DeviceType;
using feather::KernelDispatcher;
using feather::OpBase;
using feather::Tensor;
using feather::operators::ReduceSumParam;
using feather::operators::UnaryParam;

#ifdef FEATHER_WITH_CUDA
bool HasCudaDevice() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}
#endif

template <DataType dtype>
struct Storage;

template <>
struct Storage<DataType::FP32> {
    using type = float;
    static type Encode(float value) { return value; }
    static float Decode(type value) { return value; }
};

template <>
struct Storage<DataType::FP16> {
    using type = uint16_t;
    static type Encode(float value) { return feather::FloatToHalf(value); }
    static float Decode(type value) { return feather::HalfToFloat(value); }
};

template <>
struct Storage<DataType::BF16> {
    using type = BFloat16;
    static type Encode(float value) { return BFloat16{feather::FloatToBFloat16(value)}; }
    static float Decode(type value) { return feather::BFloat16ToFloat(value.bits); }
};

template <DataType dtype, typename MakeOp>
void ExpectUnaryMath(MakeOp make_op, const std::vector<float>& input,
                     const std::vector<float>& expected, float tolerance) {
    using T = typename Storage<dtype>::type;
    std::vector<T> encoded;
    encoded.reserve(input.size());
    for (float value : input) {
        encoded.push_back(Storage<dtype>::Encode(value));
    }

    auto input_tensor = std::make_shared<Tensor>();
    input_tensor->Assign<T>(encoded, {static_cast<int64_t>(input.size())});
    auto output_tensor = std::make_shared<Tensor>(std::vector<int64_t>{static_cast<int64_t>(input.size())});
    output_tensor->set_data_type(dtype);

    UnaryParam param;
    param.input = input_tensor;
    param.out = output_tensor;
    std::shared_ptr<OpBase> op = make_op(param);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);
    auto kernel = KernelDispatcher::instance().create(DeviceType::COMMON, dtype, op->type());
    ASSERT_NE(kernel, nullptr) << "missing Common kernel for " << op->type();
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(Storage<dtype>::Decode(output_tensor->data<T>()[i]), expected[i], tolerance)
            << "op=" << op->type() << " index=" << i;
    }
}

template <DataType dtype>
void ExpectUnaryRegistration(const char* op_type) {
    auto kernel = KernelDispatcher::instance().create(DeviceType::COMMON, dtype, op_type);
    ASSERT_NE(kernel, nullptr) << "missing Common " << op_type << " kernel";
    EXPECT_EQ(kernel->data_type(), dtype);
}

}  // namespace

TEST(qwen_operator_test, RegistersMathAtomicsForAllCommonFloatTypes) {
    for (const auto dtype : {DataType::FP32, DataType::FP16, DataType::BF16}) {
        auto exp = KernelDispatcher::instance().create(DeviceType::COMMON, dtype, "Exp");
        auto sin = KernelDispatcher::instance().create(DeviceType::COMMON, dtype, "Sin");
        auto cos = KernelDispatcher::instance().create(DeviceType::COMMON, dtype, "Cos");
        auto neg = KernelDispatcher::instance().create(DeviceType::COMMON, dtype, "Neg");
        auto softplus = KernelDispatcher::instance().create(DeviceType::COMMON, dtype, "Softplus");
        EXPECT_NE(exp, nullptr);
        EXPECT_NE(sin, nullptr);
        EXPECT_NE(cos, nullptr);
        EXPECT_NE(neg, nullptr);
        EXPECT_NE(softplus, nullptr);
    }
    ExpectUnaryRegistration<DataType::FP32>("Exp");
    ExpectUnaryRegistration<DataType::FP16>("Exp");
    ExpectUnaryRegistration<DataType::BF16>("Exp");
}

TEST(qwen_operator_test, RunsExpSinCosNegAndSoftplusOnBF16) {
    const std::vector<float> input = {-1.0f, 0.0f, 0.5f, 1.0f};
    ExpectUnaryMath<DataType::BF16>(
        [](const UnaryParam& param) { return std::make_shared<feather::operators::ExpOp>("exp", param); }, input,
        {std::exp(-1.0f), 1.0f, std::exp(0.5f), std::exp(1.0f)}, 0.02f);
    ExpectUnaryMath<DataType::BF16>(
        [](const UnaryParam& param) { return std::make_shared<feather::operators::SinOp>("sin", param); }, input,
        {std::sin(-1.0f), 0.0f, std::sin(0.5f), std::sin(1.0f)}, 0.01f);
    ExpectUnaryMath<DataType::BF16>(
        [](const UnaryParam& param) { return std::make_shared<feather::operators::CosOp>("cos", param); }, input,
        {std::cos(-1.0f), 1.0f, std::cos(0.5f), std::cos(1.0f)}, 0.01f);
    ExpectUnaryMath<DataType::BF16>(
        [](const UnaryParam& param) { return std::make_shared<feather::operators::NegOp>("neg", param); }, input,
        {1.0f, 0.0f, -0.5f, -1.0f}, 0.01f);
    ExpectUnaryMath<DataType::BF16>(
        [](const UnaryParam& param) { return std::make_shared<feather::operators::SoftplusOp>("softplus", param); }, input,
        {std::log1p(std::exp(-1.0f)), std::log(2.0f), std::log1p(std::exp(0.5f)), std::log1p(std::exp(1.0f))},
        0.02f);
}

TEST(qwen_operator_test, RunsMathAtomicsOnFP16AndFP32) {
    const std::vector<float> input = {-0.75f, 0.25f, 1.5f};
    const std::vector<float> expected = {std::exp(-0.75f), std::exp(0.25f), std::exp(1.5f)};
    ExpectUnaryMath<DataType::FP16>(
        [](const UnaryParam& param) { return std::make_shared<feather::operators::ExpOp>("exp_fp16", param); }, input,
        expected, 0.01f);
    ExpectUnaryMath<DataType::FP32>(
        [](const UnaryParam& param) { return std::make_shared<feather::operators::ExpOp>("exp_fp32", param); }, input,
        expected, 1e-6f);
}

TEST(qwen_operator_test, ReduceSumInfersAxesAndRunsOnBF16) {
    std::vector<BFloat16> values;
    for (float value : {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}) {
        values.push_back(BFloat16{feather::FloatToBFloat16(value)});
    }
    auto input = std::make_shared<Tensor>();
    input->Assign<BFloat16>(values, {2, 3});
    auto output = std::make_shared<Tensor>(std::vector<int64_t>{2, 1});
    output->set_data_type(DataType::BF16);

    ReduceSumParam param;
    param.input = input;
    param.out = output;
    param.axes = {1};
    param.keepdims = true;
    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::ReduceSumOp>("reduce_sum", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);
    EXPECT_EQ(output->dims().data(), std::vector<int64_t>({2, 1}));
    auto kernel = KernelDispatcher::instance().create(DeviceType::COMMON, DataType::BF16, "ReduceSum");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);
    EXPECT_NEAR(feather::BFloat16ToFloat(output->data<BFloat16>()[0].bits), 6.0f, 0.05f);
    EXPECT_NEAR(feather::BFloat16ToFloat(output->data<BFloat16>()[1].bits), 15.0f, 0.1f);
}

TEST(qwen_operator_test, X86ReduceSumUsesNativeFp32KernelForLastAxis) {
    auto input = std::make_shared<Tensor>();
    std::vector<float> values;
    for (int64_t i = 0; i < 2 * 3 * 17; ++i) values.push_back(static_cast<float>((i % 11) - 5));
    input->Assign<float>(values, {2, 3, 17});
    auto output = std::make_shared<Tensor>(std::vector<int64_t>{2, 3, 1});

    ReduceSumParam param{};
    param.input = input;
    param.out = output;
    param.axes = {-1};
    param.keepdims = true;
    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "ReduceSum");
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(kernel->device(), DeviceType::X86);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);
    EXPECT_EQ(output->data_type(), DataType::FP32);
    for (int64_t row = 0; row < 6; ++row) {
        float expected = 0.0f;
        for (int64_t col = 0; col < 17; ++col) expected += values[static_cast<size_t>(row * 17 + col)];
        EXPECT_FLOAT_EQ(output->data<float>()[row], expected);
    }
}

TEST(qwen_operator_test, RunsDepthwiseConv2DOnBF16) {
    const auto encode = [](std::initializer_list<float> values) {
        std::vector<BFloat16> result;
        result.reserve(values.size());
        for (const float value : values) {
            result.push_back(BFloat16{feather::FloatToBFloat16(value)});
        }
        return result;
    };

    auto input = std::make_shared<Tensor>();
    input->Assign<BFloat16>(encode({1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 4.0f, 6.0f, 8.0f}), {1, 2, 1, 4});
    auto weight = std::make_shared<Tensor>();
    weight->Assign<BFloat16>(encode({1.0f, 0.0f, -1.0f, 0.5f, 0.25f, -0.5f}), {2, 1, 1, 3});
    auto bias = std::make_shared<Tensor>();
    bias->Assign<BFloat16>(encode({0.5f, -1.0f}), {2});
    auto output = std::make_shared<Tensor>(std::vector<int64_t>{1, 2, 1, 4});
    output->set_data_type(DataType::BF16);

    feather::operators::Conv2dParam param{};
    param.input = input;
    param.w = weight;
    param.bias = bias;
    param.out = output;
    param.stride_h = 1;
    param.stride_w = 1;
    param.pad_h = 0;
    param.pad_w = 1;
    param.dilation_h = 1;
    param.dilation_w = 1;
    param.group = 2;
    feather::operators::Conv2dOp op("qwen_depthwise_conv", param);
    ASSERT_EQ(op.CheckShape(), 0);
    ASSERT_EQ(op.InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::COMMON, DataType::BF16, "Conv2D");
    ASSERT_NE(kernel, nullptr);
    op.AttachKernel(std::move(kernel));
    ASSERT_EQ(op.Run(), 0);

    const std::vector<float> expected = {-1.5f, -1.5f, -1.5f, 3.5f, -2.5f, -2.0f, -1.5f, 4.0f};
    const auto* values = op.outputs()[0]->data<BFloat16>();
    ASSERT_NE(values, nullptr);
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::BFloat16ToFloat(values[i].bits), expected[i], 0.04f) << "index=" << i;
    }
}

TEST(qwen_operator_test, X86QwenBf16DepthwiseConv2DMatchesCommon) {
    constexpr int64_t kChannels = 6144;
    constexpr int64_t kKernelWidth = 4;
    std::vector<BFloat16> input_values(static_cast<size_t>(kChannels * kKernelWidth));
    std::vector<BFloat16> weight_values(static_cast<size_t>(kChannels * kKernelWidth));
    std::vector<BFloat16> bias_values(static_cast<size_t>(kChannels));
    for (int64_t channel = 0; channel < kChannels; ++channel) {
        for (int64_t offset = 0; offset < kKernelWidth; ++offset) {
            const size_t index = static_cast<size_t>(channel * kKernelWidth + offset);
            input_values[index] = BFloat16{feather::FloatToBFloat16(
                0.0625f * static_cast<float>((channel * 5 + offset * 3) % 29 - 14))};
            weight_values[index] = BFloat16{feather::FloatToBFloat16(
                0.03125f * static_cast<float>((channel * 7 + offset * 11) % 31 - 15))};
        }
        bias_values[static_cast<size_t>(channel)] =
            BFloat16{feather::FloatToBFloat16(0.125f * static_cast<float>(channel % 9 - 4))};
    }

    auto input = std::make_shared<Tensor>();
    input->Assign<BFloat16>(input_values, {1, kChannels, 1, kKernelWidth});
    auto weight = std::make_shared<Tensor>();
    weight->Assign<BFloat16>(weight_values, {kChannels, 1, 1, kKernelWidth});
    auto bias = std::make_shared<Tensor>();
    bias->Assign<BFloat16>(bias_values, {kChannels});
    auto common_output = std::make_shared<Tensor>(std::vector<int64_t>{1, kChannels, 1, 1});
    auto x86_output = std::make_shared<Tensor>(std::vector<int64_t>{1, kChannels, 1, 1});

    feather::operators::Conv2dParam common_param{};
    common_param.input = input;
    common_param.w = weight;
    common_param.bias = bias;
    common_param.out = common_output;
    common_param.stride_h = 1;
    common_param.stride_w = 1;
    common_param.pad_h = 0;
    common_param.pad_w = 0;
    common_param.dilation_h = 1;
    common_param.dilation_w = 1;
    common_param.group = kChannels;
    auto x86_param = common_param;
    x86_param.out = x86_output;

    auto common_kernel = KernelDispatcher::instance().create(DeviceType::COMMON, DataType::BF16, "Conv2D");
    ASSERT_NE(common_kernel, nullptr);
    common_kernel->SetParam(&common_param);
    ASSERT_EQ(common_kernel->compute(), 0);

    auto x86_kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::BF16, "Conv2D");
    ASSERT_NE(x86_kernel, nullptr);
    EXPECT_EQ(x86_kernel->device(), DeviceType::X86);
    EXPECT_NE((dynamic_cast<feather::kernel::Conv2DKernel<DeviceType::X86, DataType::BF16>*>(x86_kernel.get())),
              nullptr);
    x86_kernel->SetParam(&x86_param);
    ASSERT_EQ(x86_kernel->compute(), 0);

    ASSERT_EQ(common_output->data_type(), DataType::BF16);
    ASSERT_EQ(x86_output->data_type(), DataType::BF16);
    const auto* common_data = common_output->data<BFloat16>();
    const auto* x86_data = x86_output->data<BFloat16>();
    ASSERT_NE(common_data, nullptr);
    ASSERT_NE(x86_data, nullptr);
    for (int64_t channel = 0; channel < kChannels; ++channel) {
        EXPECT_NEAR(feather::BFloat16ToFloat(x86_data[channel].bits),
                    feather::BFloat16ToFloat(common_data[channel].bits), 0.02f)
            << "channel=" << channel;
    }
}

TEST(qwen_operator_test, X86QwenDepthwiseConvStateMatchesConvAndStateShift) {
    constexpr int64_t kChannels = 19;
    std::vector<BFloat16> state_values(static_cast<size_t>(kChannels * 3));
    std::vector<BFloat16> mixed_values(static_cast<size_t>(kChannels));
    std::vector<BFloat16> weight_values(static_cast<size_t>(kChannels * 4));
    std::vector<BFloat16> joined_values(static_cast<size_t>(kChannels * 4));
    for (int64_t channel = 0; channel < kChannels; ++channel) {
        for (int64_t offset = 0; offset < 3; ++offset) {
            const size_t state_index = static_cast<size_t>(channel * 3 + offset);
            state_values[state_index] = BFloat16{feather::FloatToBFloat16(
                0.0625f * static_cast<float>((channel * 7 + offset * 5) % 31 - 15))};
            joined_values[static_cast<size_t>(channel * 4 + offset)] = state_values[state_index];
        }
        mixed_values[static_cast<size_t>(channel)] = BFloat16{feather::FloatToBFloat16(
            0.03125f * static_cast<float>((channel * 11) % 29 - 14))};
        joined_values[static_cast<size_t>(channel * 4 + 3)] = mixed_values[static_cast<size_t>(channel)];
        for (int64_t offset = 0; offset < 4; ++offset) {
            weight_values[static_cast<size_t>(channel * 4 + offset)] = BFloat16{feather::FloatToBFloat16(
                0.046875f * static_cast<float>((channel * 13 + offset * 3) % 37 - 18))};
        }
    }

    auto state = std::make_shared<Tensor>();
    state->Assign<BFloat16>(state_values, {1, kChannels, 3});
    auto mixed = std::make_shared<Tensor>();
    mixed->Assign<BFloat16>(mixed_values, {1, kChannels, 1});
    auto weight = std::make_shared<Tensor>();
    weight->Assign<BFloat16>(weight_values, {kChannels, 1, 1, 4});
    auto joined = std::make_shared<Tensor>();
    joined->Assign<BFloat16>(joined_values, {1, kChannels, 1, 4});
    auto reference_conv_out = std::make_shared<Tensor>(std::vector<int64_t>{1, kChannels, 1, 1});

    feather::operators::Conv2dParam reference_param{};
    reference_param.input = joined;
    reference_param.w = weight;
    reference_param.out = reference_conv_out;
    reference_param.stride_h = 1;
    reference_param.stride_w = 1;
    reference_param.pad_h = 0;
    reference_param.pad_w = 0;
    reference_param.dilation_h = 1;
    reference_param.dilation_w = 1;
    reference_param.group = kChannels;
    auto reference_kernel = KernelDispatcher::instance().create(DeviceType::COMMON, DataType::BF16, "Conv2D");
    ASSERT_NE(reference_kernel, nullptr);
    reference_kernel->SetParam(&reference_param);
    ASSERT_EQ(reference_kernel->compute(), 0);

    auto conv_out = std::make_shared<Tensor>(std::vector<int64_t>{1, kChannels, 1, 1});
    auto discarded_prefix = std::make_shared<Tensor>(std::vector<int64_t>{1, kChannels, 1});
    auto next_state = std::make_shared<Tensor>(std::vector<int64_t>{1, kChannels, 3});
    feather::operators::QwenDepthwiseConvStateParam param{};
    param.state = state;
    param.mixed = mixed;
    param.weight = weight;
    param.conv_out = conv_out;
    param.discarded_prefix = discarded_prefix;
    param.next_state = next_state;
    feather::operators::QwenDepthwiseConvStateOp op("qwen_depthwise_conv_state", param);
    ASSERT_EQ(op.InferOutputShapes(), 0);
    ASSERT_EQ(op.CheckShape(), 0);

    feather::kernel::EnsureQwenDepthwiseConvStateKernelsRegistered();
    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::BF16, "QwenDepthwiseConvState");
    ASSERT_NE(kernel, nullptr);
    op.AttachKernel(std::move(kernel));
    ASSERT_EQ(op.Run(), 0);

    const auto* reference_data = reference_conv_out->data<BFloat16>();
    const auto* conv_data = conv_out->data<BFloat16>();
    const auto* discarded_data = discarded_prefix->data<BFloat16>();
    const auto* next_state_data = next_state->data<BFloat16>();
    ASSERT_NE(reference_data, nullptr);
    ASSERT_NE(conv_data, nullptr);
    ASSERT_NE(discarded_data, nullptr);
    ASSERT_NE(next_state_data, nullptr);
    for (int64_t channel = 0; channel < kChannels; ++channel) {
        EXPECT_EQ(conv_data[channel].bits, reference_data[channel].bits) << "conv channel=" << channel;
        EXPECT_EQ(discarded_data[channel].bits, state_values[static_cast<size_t>(channel * 3)].bits)
            << "prefix channel=" << channel;
        EXPECT_EQ(next_state_data[channel * 3].bits, state_values[static_cast<size_t>(channel * 3 + 1)].bits)
            << "state channel=" << channel << " offset=0";
        EXPECT_EQ(next_state_data[channel * 3 + 1].bits, state_values[static_cast<size_t>(channel * 3 + 2)].bits)
            << "state channel=" << channel << " offset=1";
        EXPECT_EQ(next_state_data[channel * 3 + 2].bits, mixed_values[static_cast<size_t>(channel)].bits)
            << "state channel=" << channel << " offset=2";
    }
}

TEST(qwen_operator_test, X86QwenFp8DepthwiseConvStateMatchesQuantizedConvAndStateShift) {
    constexpr int64_t channels = 19;
    constexpr float input_scale = 0.25f;
    constexpr float weight_scale = 0.125f;
    constexpr float output_scale = 0.25f;
    std::vector<BFloat16> state_values(static_cast<size_t>(channels * 3));
    std::vector<BFloat16> mixed_values(static_cast<size_t>(channels));
    std::vector<feather::Fp8E4M3> weight_values(static_cast<size_t>(channels * 4));
    for (int64_t channel = 0; channel < channels; ++channel) {
        for (int64_t offset = 0; offset < 3; ++offset) {
            state_values[static_cast<size_t>(channel * 3 + offset)] = BFloat16{feather::FloatToBFloat16(
                0.0625f * static_cast<float>((channel * 7 + offset * 5) % 31 - 15))};
        }
        mixed_values[static_cast<size_t>(channel)] = BFloat16{feather::FloatToBFloat16(
            0.03125f * static_cast<float>((channel * 11) % 29 - 14))};
        for (int64_t offset = 0; offset < 4; ++offset) {
            const float value = 0.046875f * static_cast<float>((channel * 13 + offset * 3) % 37 - 18);
            weight_values[static_cast<size_t>(channel * 4 + offset)] =
                feather::Fp8E4M3{feather::FloatToFp8E4M3(value / weight_scale)};
        }
    }

    auto state = std::make_shared<Tensor>();
    state->Assign<BFloat16>(state_values, {1, channels, 3});
    auto mixed = std::make_shared<Tensor>();
    mixed->Assign<BFloat16>(mixed_values, {1, channels, 1});
    auto weight = std::make_shared<Tensor>();
    weight->Assign<feather::Fp8E4M3>(weight_values, {channels, 1, 1, 4});
    weight->set_quantization({true, weight_scale});
    auto conv_out = std::make_shared<Tensor>(std::vector<int64_t>{1, channels, 1, 1});
    conv_out->set_data_type(DataType::BF16);
    auto discarded_prefix = std::make_shared<Tensor>(std::vector<int64_t>{1, channels, 1});
    discarded_prefix->set_data_type(DataType::BF16);
    auto next_state = std::make_shared<Tensor>(std::vector<int64_t>{1, channels, 3});
    next_state->set_data_type(DataType::BF16);

    feather::operators::QwenDepthwiseConvStateParam param{};
    param.state = state;
    param.mixed = mixed;
    param.weight = weight;
    param.conv_out = conv_out;
    param.discarded_prefix = discarded_prefix;
    param.next_state = next_state;
    param.fp8_dtype = DataType::FP8E4M3;
    param.fp8_input_scale = input_scale;
    param.fp8_output_scale = output_scale;

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP8E4M3,
                                                       "QwenDepthwiseConvState");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    const auto* conv_data = conv_out->data<BFloat16>();
    const auto* prefix = discarded_prefix->data<BFloat16>();
    const auto* next = next_state->data<BFloat16>();
    for (int64_t channel = 0; channel < channels; ++channel) {
        float expected_value = 0.0f;
        for (int64_t tap = 0; tap < 4; ++tap) {
            const int64_t source_index = channel * 3 + tap;
            const float source = tap < 3 ? feather::BFloat16ToFloat(state_values[static_cast<size_t>(source_index)].bits)
                                         : feather::BFloat16ToFloat(mixed_values[static_cast<size_t>(channel)].bits);
            const float quantized_source = feather::Fp8E4M3ToFloat(feather::FloatToFp8E4M3(source / input_scale)) * input_scale;
            const float decoded_weight = feather::Fp8E4M3ToFloat(weight_values[static_cast<size_t>(channel * 4 + tap)].bits) *
                                         weight_scale;
            expected_value += quantized_source * decoded_weight;
        }
        const uint8_t expected_code = feather::FloatToFp8E4M3(expected_value / output_scale);
        const float expected_bf16 = feather::Fp8E4M3ToFloat(expected_code) * output_scale;
        EXPECT_EQ(conv_data[channel].bits, feather::FloatToBFloat16(expected_bf16)) << "channel=" << channel;
        EXPECT_EQ(prefix[channel].bits, state_values[static_cast<size_t>(channel * 3)].bits)
            << "prefix channel=" << channel;
        EXPECT_EQ(next[channel * 3].bits, state_values[static_cast<size_t>(channel * 3 + 1)].bits)
            << "state channel=" << channel << " offset=0";
        EXPECT_EQ(next[channel * 3 + 1].bits, state_values[static_cast<size_t>(channel * 3 + 2)].bits)
            << "state channel=" << channel << " offset=1";
        EXPECT_EQ(next[channel * 3 + 2].bits, mixed_values[static_cast<size_t>(channel)].bits)
            << "state channel=" << channel << " offset=2";
    }
}

#ifdef FEATHER_WITH_CUDA
TEST(qwen_operator_test, RegistersQwenFloatAtomicsNativelyOnCuda) {
    const std::vector<const char*> operators = {
        "Add",       "Cast",      "Concat",    "Conv2D",    "Cos",       "Div",
        "Exp",       "Expand",    "Gather",    "Gemm",      "Identity",  "MatMul",
        "Mul",       "Neg",       "ReduceMean", "ReduceSum", "Reshape",   "Sigmoid",
        "Sin",       "SiLU",      "Softmax",   "Softplus",  "Split",     "Sqrt",      "Sub",
        "Transpose", "Unsqueeze",
    };
    for (const auto dtype : {DataType::FP32, DataType::FP16, DataType::BF16}) {
        for (const auto* op_type : operators) {
            auto kernel = KernelDispatcher::instance().create(DeviceType::CUDA, dtype, op_type);
            ASSERT_NE(kernel, nullptr) << "missing CUDA kernel for " << op_type;
            EXPECT_EQ(kernel->device(), DeviceType::CUDA) << "CUDA fell back for " << op_type;
        }
    }
}

TEST(qwen_operator_test, RegistersQwenFusedKernelsNativelyOnCuda) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }
    struct KernelSpec {
        DataType dtype;
        const char* op_type;
    };
    const std::vector<KernelSpec> specs = {
        {DataType::BF16, "QwenRmsNorm"},
        {DataType::FP32, "QwenRmsNorm"},
        {DataType::BF16, "QwenDepthwiseConvState"},
        {DataType::FP32, "QwenGatedDeltaState"},
        {DataType::FP32, "QwenGatedDeltaOutput"},
        {DataType::FP32, "QwenGatedDelta"},
        {DataType::BF16, "QwenGemmArgmax"},
    };
    for (const auto& spec : specs) {
        auto kernel = KernelDispatcher::instance().create(DeviceType::CUDA, spec.dtype, spec.op_type);
        ASSERT_NE(kernel, nullptr) << "missing CUDA kernel for " << spec.op_type;
        EXPECT_EQ(kernel->device(), DeviceType::CUDA) << spec.op_type;
        EXPECT_EQ(kernel->data_type(), spec.dtype) << spec.op_type;
    }
}

TEST(qwen_operator_test, CudaQwenRmsNormMatchesCommonBf16WithWeightOffset) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }
    auto input = std::make_shared<Tensor>();
    input->Assign<BFloat16>({BFloat16{feather::FloatToBFloat16(1.0f)},
                             BFloat16{feather::FloatToBFloat16(-2.0f)},
                             BFloat16{feather::FloatToBFloat16(0.5f)},
                             BFloat16{feather::FloatToBFloat16(3.0f)},
                             BFloat16{feather::FloatToBFloat16(-1.5f)},
                             BFloat16{feather::FloatToBFloat16(2.0f)},
                             BFloat16{feather::FloatToBFloat16(0.25f)},
                             BFloat16{feather::FloatToBFloat16(-0.75f)}},
                            {2, 4});
    auto weight = std::make_shared<Tensor>();
    weight->Assign<BFloat16>({BFloat16{feather::FloatToBFloat16(0.5f)},
                              BFloat16{feather::FloatToBFloat16(1.0f)},
                              BFloat16{feather::FloatToBFloat16(-0.75f)},
                              BFloat16{feather::FloatToBFloat16(2.0f)}},
                             {4});
    auto epsilon = std::make_shared<Tensor>();
    epsilon->Assign<float>({1.0e-6f}, {1});
    auto common_out = std::make_shared<Tensor>(std::vector<int64_t>{2, 4});
    common_out->set_data_type(DataType::BF16);
    auto cuda_out = std::make_shared<Tensor>(std::vector<int64_t>{2, 4});
    cuda_out->set_data_type(DataType::BF16);

    feather::operators::QwenRmsNormParam common_param{};
    common_param.input = input;
    common_param.weight = weight;
    common_param.epsilon = epsilon;
    common_param.out = common_out;
    common_param.weight_offset = 0.5f;
    auto common = KernelDispatcher::instance().create(DeviceType::COMMON, DataType::BF16, "QwenRmsNorm");
    ASSERT_NE(common, nullptr);
    common->SetParam(&common_param);
    ASSERT_EQ(common->compute(), 0);

    auto cuda_param = common_param;
    cuda_param.out = cuda_out;
    auto cuda = KernelDispatcher::instance().create(DeviceType::CUDA, DataType::BF16, "QwenRmsNorm");
    ASSERT_NE(cuda, nullptr);
    cuda->SetParam(&cuda_param);
    ASSERT_EQ(cuda->compute(), 0);

    for (int64_t index = 0; index < common_out->numel(); ++index) {
        EXPECT_NEAR(feather::BFloat16ToFloat(cuda_out->data<BFloat16>()[index].bits),
                    feather::BFloat16ToFloat(common_out->data<BFloat16>()[index].bits), 0.02f)
            << "index=" << index;
    }
}

TEST(qwen_operator_test, CudaQwenDepthwiseConvStateMatchesX86) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }
    constexpr int64_t channels = 7;
    std::vector<BFloat16> state_values(static_cast<size_t>(channels * 3));
    std::vector<BFloat16> mixed_values(static_cast<size_t>(channels));
    std::vector<BFloat16> weight_values(static_cast<size_t>(channels * 4));
    for (int64_t channel = 0; channel < channels; ++channel) {
        for (int64_t offset = 0; offset < 3; ++offset) {
            state_values[static_cast<size_t>(channel * 3 + offset)] =
                BFloat16{feather::FloatToBFloat16(0.0625f * static_cast<float>((channel + offset * 3) % 11 - 5))};
        }
        mixed_values[static_cast<size_t>(channel)] =
            BFloat16{feather::FloatToBFloat16(0.03125f * static_cast<float>((channel * 5) % 13 - 6))};
        for (int64_t offset = 0; offset < 4; ++offset) {
            weight_values[static_cast<size_t>(channel * 4 + offset)] =
                BFloat16{feather::FloatToBFloat16(0.046875f * static_cast<float>((channel * 7 + offset) % 17 - 8))};
        }
    }
    auto state = std::make_shared<Tensor>();
    state->Assign<BFloat16>(state_values, {1, channels, 3});
    auto mixed = std::make_shared<Tensor>();
    mixed->Assign<BFloat16>(mixed_values, {1, channels, 1});
    auto weight = std::make_shared<Tensor>();
    weight->Assign<BFloat16>(weight_values, {channels, 1, 1, 4});

    auto make_output = [](const std::vector<int64_t>& dims) {
        auto tensor = std::make_shared<Tensor>(dims);
        tensor->set_data_type(DataType::BF16);
        return tensor;
    };
    auto x86_conv = make_output({1, channels, 1, 1});
    auto x86_prefix = make_output({1, channels, 1});
    auto x86_next = make_output({1, channels, 3});
    auto cuda_conv = make_output({1, channels, 1, 1});
    auto cuda_prefix = make_output({1, channels, 1});
    auto cuda_next = make_output({1, channels, 3});

    feather::operators::QwenDepthwiseConvStateParam x86_param{};
    x86_param.state = state;
    x86_param.mixed = mixed;
    x86_param.weight = weight;
    x86_param.conv_out = x86_conv;
    x86_param.discarded_prefix = x86_prefix;
    x86_param.next_state = x86_next;
    auto x86 = KernelDispatcher::instance().create(DeviceType::X86, DataType::BF16, "QwenDepthwiseConvState");
    ASSERT_NE(x86, nullptr);
    x86->SetParam(&x86_param);
    ASSERT_EQ(x86->compute(), 0);

    auto cuda_param = x86_param;
    cuda_param.conv_out = cuda_conv;
    cuda_param.discarded_prefix = cuda_prefix;
    cuda_param.next_state = cuda_next;
    auto cuda = KernelDispatcher::instance().create(DeviceType::CUDA, DataType::BF16, "QwenDepthwiseConvState");
    ASSERT_NE(cuda, nullptr);
    cuda->SetParam(&cuda_param);
    ASSERT_EQ(cuda->compute(), 0);

    for (int64_t channel = 0; channel < channels; ++channel) {
        EXPECT_NEAR(feather::BFloat16ToFloat(cuda_conv->data<BFloat16>()[channel].bits),
                    feather::BFloat16ToFloat(x86_conv->data<BFloat16>()[channel].bits), 0.02f)
            << "conv channel=" << channel;
        EXPECT_EQ(cuda_prefix->data<BFloat16>()[channel].bits, x86_prefix->data<BFloat16>()[channel].bits)
            << "prefix channel=" << channel;
    }
    for (int64_t index = 0; index < cuda_next->numel(); ++index) {
        EXPECT_EQ(cuda_next->data<BFloat16>()[index].bits, x86_next->data<BFloat16>()[index].bits)
            << "state index=" << index;
    }
}
#endif
