#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "core/tensor.h"
#include "quant/quantization.h"

namespace {

using feather::QuantizationGranularity;
using feather::QuantizationParams;

TEST(quantization_math_test, QuantizesWithZeroPointAndSaturates) {
    QuantizationParams params;
    params.enabled = true;
    params.scale = 0.5f;
    params.zero_point = -3;

    const std::vector<float> source = {-100.0f, -1.0f, -0.25f, 0.25f, 1.0f, 100.0f};
    std::vector<int8_t> quantized(source.size(), 0);
    ASSERT_EQ(feather::QuantizeInt8(source.data(), quantized.data(), source.size(), params), 0);

    EXPECT_EQ(quantized, (std::vector<int8_t>{-128, -5, -4, -3, -1, 127}));

    std::vector<float> restored(source.size(), 0.0f);
    ASSERT_EQ(feather::DequantizeInt8(quantized.data(), restored.data(), restored.size(), params), 0);
    EXPECT_FLOAT_EQ(restored[1], -1.0f);
    EXPECT_FLOAT_EQ(restored[4], 1.0f);
}

TEST(quantization_math_test, AppliesPerChannelParametersAlongAxis) {
    QuantizationParams params;
    params.enabled = true;
    params.granularity = QuantizationGranularity::kPerChannel;
    params.axis = 1;
    params.scales = {0.5f, 1.0f, 2.0f};
    params.zero_points = {0, 1, -2};

    const std::vector<int64_t> dims = {2, 3};
    const std::vector<float> source = {1.0f, 2.0f, 4.0f, -1.0f, -2.0f, -4.0f};
    std::vector<int8_t> quantized(source.size(), 0);
    ASSERT_EQ(feather::QuantizeInt8(source.data(), quantized.data(), dims, params), 0);
    EXPECT_EQ(quantized, (std::vector<int8_t>{2, 3, 0, -2, -1, -4}));

    std::vector<float> restored(source.size(), 0.0f);
    ASSERT_EQ(feather::DequantizeInt8(quantized.data(), restored.data(), dims, params), 0);
    EXPECT_FLOAT_EQ(restored[0], 1.0f);
    EXPECT_FLOAT_EQ(restored[1], 2.0f);
    EXPECT_FLOAT_EQ(restored[2], 4.0f);
    EXPECT_FLOAT_EQ(restored[3], -1.0f);
}

TEST(quantization_math_test, RequantizesInt32IntoOutputDomain) {
    QuantizationParams accumulator;
    accumulator.enabled = true;
    accumulator.scale = 0.25f;
    QuantizationParams output;
    output.enabled = true;
    output.scale = 0.5f;
    output.zero_point = 2;

    const int32_t source[] = {-4, 0, 4, 1000};
    int8_t result[4] = {};
    ASSERT_EQ(feather::RequantizeInt32(source, result, 4, accumulator, output), 0);
    EXPECT_EQ(result[0], 0);
    EXPECT_EQ(result[1], 2);
    EXPECT_EQ(result[2], 4);
    EXPECT_EQ(result[3], 127);
}

TEST(quantization_math_test, RejectsInvalidAndUnsupportedMetadata) {
    QuantizationParams invalid;
    invalid.enabled = true;
    invalid.scale = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(feather::ValidateQuantizationParams(invalid, {4}));

    QuantizationParams mismatch;
    mismatch.enabled = true;
    mismatch.granularity = QuantizationGranularity::kPerChannel;
    mismatch.axis = 1;
    mismatch.scales = {1.0f, 2.0f};
    EXPECT_FALSE(feather::ValidateQuantizationParams(mismatch, {2, 3}));

    QuantizationParams blocked;
    blocked.enabled = true;
    blocked.granularity = QuantizationGranularity::kPerBlock;
    blocked.block_size = 32;
    blocked.scales = {1.0f};
    EXPECT_FALSE(feather::ValidateQuantizationParams(blocked, {1, 32}));
}

TEST(quantization_math_test, TensorCanonicalizesScalarAndVectorMetadata) {
    auto tensor = std::make_shared<feather::Tensor>(std::vector<int64_t>{2, 3});
    QuantizationParams params;
    params.enabled = true;
    params.scale = 0.25f;
    params.zero_point = -2;
    tensor->set_quantization(params);
    ASSERT_EQ(tensor->quantization().scales.size(), 1U);
    ASSERT_EQ(tensor->quantization().zero_points.size(), 1U);
    EXPECT_FLOAT_EQ(tensor->quantization().scales.front(), 0.25f);
    EXPECT_EQ(tensor->quantization().zero_points.front(), -2);

    params.granularity = QuantizationGranularity::kPerChannel;
    params.axis = 1;
    params.scales = {0.25f, 0.5f, 1.0f};
    params.zero_points = {-2, 0, 3};
    tensor->set_quantization(params);
    EXPECT_FLOAT_EQ(tensor->quantization().scale, 0.25f);
    EXPECT_EQ(tensor->quantization().zero_point, -2);
    EXPECT_FLOAT_EQ(tensor->quantization().scales[2], 1.0f);
    EXPECT_EQ(tensor->quantization().zero_points[2], 3);
}

}  // namespace
