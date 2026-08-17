#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/kernel.h"
#include "core/operator.h"
#include "core/tensor.h"
#include "src/operator/conv2d_op.h"
#include "src/operator/cos_op.h"
#include "src/operator/exp_op.h"
#include "src/operator/neg_op.h"
#include "src/operator/params.h"
#include "src/operator/reduce_sum_op.h"
#include "src/operator/sin_op.h"
#include "src/operator/softplus_op.h"
#include "util/bf16.h"
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

#ifdef FEATHER_WITH_CUDA
TEST(qwen_operator_test, RegistersQwenFloatAtomicsNativelyOnCuda) {
    const std::vector<const char*> operators = {
        "Add",       "Cast",      "Concat",    "Conv2D",    "Cos",       "Div",
        "Exp",       "Expand",    "Gather",    "Gemm",      "Identity",  "MatMul",
        "Mul",       "Neg",       "ReduceMean", "ReduceSum", "Reshape",   "Sigmoid",
        "Sin",       "Softmax",   "Softplus",  "Split",     "Sqrt",      "Sub",
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
#endif
