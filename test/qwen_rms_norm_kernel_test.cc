#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "core/kernel.h"
#include "core/tensor.h"
#include "src/kernel/qwen_rms_norm.h"
#include "src/operator/params.h"
#include "util/bf16.h"

namespace {

using feather::BFloat16;
using feather::BFloat16ToFloat;
using feather::FloatToBFloat16;

std::shared_ptr<feather::Tensor> MakeBf16(const std::vector<float>& values, const std::vector<int64_t>& dims) {
    std::vector<BFloat16> converted;
    converted.reserve(values.size());
    for (const float value : values) {
        converted.push_back(BFloat16{FloatToBFloat16(value)});
    }
    auto tensor = std::make_shared<feather::Tensor>();
    tensor->Assign<BFloat16>(converted, dims);
    return tensor;
}

std::shared_ptr<feather::Tensor> MakeFp32(const std::vector<float>& values, const std::vector<int64_t>& dims) {
    auto tensor = std::make_shared<feather::Tensor>();
    tensor->Assign<float>(values, dims);
    return tensor;
}

float Reference(const std::vector<float>& input, const std::vector<float>& weight, int64_t row, int64_t hidden,
                float epsilon, int64_t column) {
    float sum_square = 0.0f;
    for (int64_t index = 0; index < hidden; ++index) {
        const float value = input[static_cast<size_t>(row * hidden + index)];
        sum_square += value * value;
    }
    const float inverse = 1.0f / std::sqrt(sum_square / static_cast<float>(hidden) + epsilon);
    return input[static_cast<size_t>(row * hidden + column)] * inverse * weight[static_cast<size_t>(column)];
}

TEST(qwen_rms_norm_kernel_test, NormalizesBf16RowsAndWritesBf16) {
    const std::vector<float> input_values = {1.0f, -2.0f, 0.5f, 4.0f, -3.0f, 2.0f, 1.0f, -0.25f};
    const std::vector<float> weight_values = {0.5f, 1.25f, -0.75f, 2.0f};
    auto input = MakeBf16(input_values, {2, 4});
    auto weight = MakeFp32(weight_values, {4});
    auto epsilon = MakeFp32({1.0e-5f}, {1});
    auto output = std::make_shared<feather::Tensor>(std::vector<int64_t>{2, 4});
    output->set_data_type(feather::DataType::BF16);

    feather::operators::QwenRmsNormParam param{};
    param.input = input;
    param.weight = weight;
    param.epsilon = epsilon;
    param.out = output;
    auto kernel = feather::KernelDispatcher::instance().create(feather::DeviceType::X86, feather::DataType::BF16,
                                                                "QwenRmsNorm");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    for (int64_t row = 0; row < 2; ++row) {
        for (int64_t column = 0; column < 4; ++column) {
            const size_t offset = static_cast<size_t>(row * 4 + column);
            EXPECT_NEAR(BFloat16ToFloat(output->data<BFloat16>()[offset].bits),
                        Reference(input_values, weight_values, row, 4, 1.0e-5f, column), 0.03f);
        }
    }
}

TEST(qwen_rms_norm_kernel_test, KeepsFp32OutputForGatedPath) {
    const std::vector<float> input_values = {0.25f, -1.5f, 2.0f, 0.75f};
    const std::vector<float> weight_values = {1.0f, 0.5f, 1.5f, -2.0f};
    auto input = MakeBf16(input_values, {1, 4});
    auto weight = MakeFp32(weight_values, {4});
    auto epsilon = MakeFp32({1.0e-6f}, {1});
    auto output = std::make_shared<feather::Tensor>(std::vector<int64_t>{1, 4});
    output->set_data_type(feather::DataType::FP32);

    feather::operators::QwenRmsNormParam param{};
    param.input = input;
    param.weight = weight;
    param.epsilon = epsilon;
    param.out = output;
    auto kernel = feather::KernelDispatcher::instance().create(feather::DeviceType::X86, feather::DataType::BF16,
                                                                "QwenRmsNorm");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    for (int64_t column = 0; column < 4; ++column) {
        EXPECT_NEAR(output->data<float>()[column], Reference(input_values, weight_values, 0, 4, 1.0e-6f, column),
                    1.0e-5f);
    }
}

}  // namespace
