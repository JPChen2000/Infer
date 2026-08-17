#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "core/kernel.h"
#include "core/operator_registry.h"
#include "core/tensor.h"
#include "model/model_format.h"
#include "util/bf16.h"
#include "util/types.h"

namespace {

TEST(bfloat16_test, ConvertsFloatWithRoundToNearestEven) {
    EXPECT_EQ(feather::FloatToBFloat16(1.0f), 0x3f80u);
    EXPECT_EQ(feather::FloatToBFloat16(1.00390625f), 0x3f80u);
    EXPECT_EQ(feather::FloatToBFloat16(1.01171875f), 0x3f82u);
}

TEST(bfloat16_test, PreservesFiniteAndSpecialValues) {
    EXPECT_FLOAT_EQ(feather::BFloat16ToFloat(0x3f80u), 1.0f);
    EXPECT_FLOAT_EQ(feather::BFloat16ToFloat(0xc020u), -2.5f);
    EXPECT_TRUE(std::isinf(feather::BFloat16ToFloat(feather::FloatToBFloat16(
        std::numeric_limits<float>::infinity()))));
    EXPECT_TRUE(std::isnan(feather::BFloat16ToFloat(feather::FloatToBFloat16(
        std::numeric_limits<float>::quiet_NaN()))));
}

TEST(bfloat16_test, PreservesNaNWhenRoundingWouldOverflowExponent) {
    const uint32_t float_bits = 0x7fffffffu;
    float value = 0.0f;
    std::memcpy(&value, &float_bits, sizeof(value));

    const uint16_t converted = feather::FloatToBFloat16(value);
    EXPECT_EQ(converted & 0x7f80u, 0x7f80u);
    EXPECT_NE(converted & 0x007fu, 0u);
    EXPECT_TRUE(std::isnan(feather::BFloat16ToFloat(converted)));
}

TEST(bfloat16_test, IsDistinctTwoByteDataType) {
    EXPECT_EQ(feather::DataTypeTrait<feather::BFloat16>::type(), feather::DataType::BF16);
    EXPECT_EQ(feather::DataTypeBytes(feather::DataType::BF16), 2U);
    EXPECT_NE(feather::DataType::BF16, feather::DataType::FP16);
}

TEST(bfloat16_test, RegistersCoreTransformerKernelsOnCommon) {
    const std::vector<std::string> operators = {
        "Add",       "Sub",       "Mul",       "Div",      "Pow",        "Sqrt",     "Erf",
        "Sigmoid",   "SiLU",      "Softmax",   "MatMul",   "Gemm",       "Gather",   "ReduceMean",
        "Reshape",   "Transpose", "Concat",    "Split",    "Slice",      "Squeeze",  "Unsqueeze",
        "Expand",    "Where",     "Equal",     "Cast",     "Identity",   "ReLU",     "Tanh",
    };
    for (const auto& op : operators) {
        auto kernel = feather::KernelDispatcher::instance().create(feather::DeviceType::COMMON,
                                                                     feather::DataType::BF16, op);
        EXPECT_NE(kernel, nullptr) << "missing Common BF16 kernel for " << op;
        if (kernel != nullptr) {
            EXPECT_EQ(kernel->device(), feather::DeviceType::COMMON);
            EXPECT_EQ(kernel->data_type(), feather::DataType::BF16);
        }
    }
}

TEST(bfloat16_test, RunsMatMulAndReduceMeanWithBFloat16Storage) {
    auto make_tensor = [](const std::vector<float>& values, const std::vector<int64_t>& dims) {
        std::vector<feather::BFloat16> storage;
        storage.reserve(values.size());
        for (const float value : values) {
            storage.push_back({feather::FloatToBFloat16(value)});
        }
        auto tensor = std::make_shared<feather::Tensor>();
        tensor->Assign<feather::BFloat16>(storage, dims);
        return tensor;
    };

    feather::OperatorRegistry::TensorMap tensors;
    tensors["lhs"] = make_tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    tensors["rhs"] = make_tensor({2.0f, 0.0f, 1.0f, 2.0f}, {2, 2});
    tensors["matmul_out"] = std::make_shared<feather::Tensor>(std::vector<int64_t>{2, 2});
    tensors["mean_out"] = std::make_shared<feather::Tensor>(std::vector<int64_t>{2, 1});

    feather::model::NodeDesc matmul;
    matmul.name = "matmul";
    matmul.op_type = "MatMul";
    matmul.inputs = {"lhs", "rhs"};
    matmul.outputs = {"matmul_out"};

    feather::model::NodeDesc reduce_mean;
    reduce_mean.name = "reduce_mean";
    reduce_mean.op_type = "ReduceMean";
    reduce_mean.inputs = {"matmul_out"};
    reduce_mean.outputs = {"mean_out"};
    reduce_mean.attributes["axes"] = std::vector<int64_t>{1};
    reduce_mean.attributes["keepdims"] = int64_t{1};

    feather::KernelDeviceScope scope(feather::DeviceType::COMMON);
    auto matmul_op = feather::OperatorRegistry::instance().Create(matmul, tensors);
    ASSERT_NE(matmul_op, nullptr);
    ASSERT_EQ(matmul_op->Run(), 0);
    tensors["matmul_out"] = matmul_op->outputs().front();

    auto reduce_mean_op = feather::OperatorRegistry::instance().Create(reduce_mean, tensors);
    ASSERT_NE(reduce_mean_op, nullptr);
    ASSERT_EQ(reduce_mean_op->Run(), 0);

    const auto matmul_out = matmul_op->outputs().front();
    ASSERT_NE(matmul_out, nullptr);
    EXPECT_EQ(matmul_out->data_type(), feather::DataType::BF16);
    EXPECT_NEAR(feather::BFloat16ToFloat(matmul_out->data<feather::BFloat16>()[0].bits), 4.0f, 0.02f);
    EXPECT_NEAR(feather::BFloat16ToFloat(matmul_out->data<feather::BFloat16>()[3].bits), 8.0f, 0.02f);

    const auto mean_out = reduce_mean_op->outputs().front();
    ASSERT_NE(mean_out, nullptr);
    EXPECT_EQ(mean_out->data_type(), feather::DataType::BF16);
    EXPECT_NEAR(feather::BFloat16ToFloat(mean_out->data<feather::BFloat16>()[0].bits), 4.0f, 0.03f);
    EXPECT_NEAR(feather::BFloat16ToFloat(mean_out->data<feather::BFloat16>()[1].bits), 9.0f, 0.05f);
}

}  // namespace
