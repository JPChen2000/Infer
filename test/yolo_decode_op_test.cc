#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "core/kernel.h"
#include "core/operator.h"
#include "core/tensor.h"
#include "src/kernel/yolo_decode.h"
#include "src/operator/params.h"
#include "src/operator/yolo_decode_op.h"
#include "util/fp16.h"

using feather::DataType;
using feather::DeviceType;
using feather::KernelDispatcher;
using feather::OpBase;
using feather::Tensor;
using feather::operators::YoloDecodeParam;

namespace {

float Sigmoid(float value) { return 1.0f / (1.0f + std::exp(-value)); }

std::shared_ptr<Tensor> MakeTensor(const std::vector<float>& values, const std::vector<int64_t>& shape) {
    auto tensor = std::make_shared<Tensor>();
    tensor->Assign<float>(values, shape);
    return tensor;
}

std::shared_ptr<Tensor> MakeHalfTensor(const std::vector<float>& values, const std::vector<int64_t>& shape) {
    std::vector<uint16_t> half(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        half[i] = feather::FloatToHalf(values[i]);
    }
    auto tensor = std::make_shared<Tensor>();
    tensor->Assign<uint16_t>(half, shape);
    return tensor;
}

YoloDecodeParam MakeYoloDecodeParam(bool fp16) {
    YoloDecodeParam param{};
    const std::vector<float> raw = {
        0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f,
        2.0f, -2.0f, 0.5f, -0.5f, 0.0f, 4.0f,
    };
    param.input = fp16 ? MakeHalfTensor(raw, {1, 12, 1, 1}) : MakeTensor(raw, {1, 12, 1, 1});
    param.xy_scale = fp16 ? MakeHalfTensor({2.0f}, {1}) : MakeTensor({2.0f}, {1});
    param.grid = fp16 ? MakeHalfTensor({10.0f, 20.0f, 30.0f, 40.0f}, {1, 2, 1, 1, 2})
                      : MakeTensor({10.0f, 20.0f, 30.0f, 40.0f}, {1, 2, 1, 1, 2});
    param.stride = fp16 ? MakeHalfTensor({8.0f}, {1}) : MakeTensor({8.0f}, {1});
    param.wh_scale = fp16 ? MakeHalfTensor({2.0f}, {1}) : MakeTensor({2.0f}, {1});
    param.anchor_grid = fp16 ? MakeHalfTensor({4.0f, 6.0f, 8.0f, 10.0f}, {1, 2, 1, 1, 2})
                             : MakeTensor({4.0f, 6.0f, 8.0f, 10.0f}, {1, 2, 1, 1, 2});
    param.out = std::make_shared<Tensor>(std::vector<int64_t>{1, 2, 6});
    if (fp16) {
        param.out->mutable_data<uint16_t>();
    }
    return param;
}

std::vector<float> ExpectedDecoded() {
    const std::vector<float> raw = {
        0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f,
        2.0f, -2.0f, 0.5f, -0.5f, 0.0f, 4.0f,
    };
    const std::vector<float> grid = {10.0f, 20.0f, 30.0f, 40.0f};
    const std::vector<float> anchor = {4.0f, 6.0f, 8.0f, 10.0f};
    std::vector<float> expected(12);
    for (int anchor_idx = 0; anchor_idx < 2; ++anchor_idx) {
        const int in_base = anchor_idx * 6;
        const int out_base = anchor_idx * 6;
        expected[out_base + 0] = (Sigmoid(raw[in_base + 0]) * 2.0f + grid[anchor_idx * 2 + 0]) * 8.0f;
        expected[out_base + 1] = (Sigmoid(raw[in_base + 1]) * 2.0f + grid[anchor_idx * 2 + 1]) * 8.0f;
        const float w = Sigmoid(raw[in_base + 2]) * 2.0f;
        const float h = Sigmoid(raw[in_base + 3]) * 2.0f;
        expected[out_base + 2] = w * w * anchor[anchor_idx * 2 + 0];
        expected[out_base + 3] = h * h * anchor[anchor_idx * 2 + 1];
        expected[out_base + 4] = Sigmoid(raw[in_base + 4]);
        expected[out_base + 5] = Sigmoid(raw[in_base + 5]);
    }
    return expected;
}

}  // namespace

TEST(yolo_decode_op_test, YoloDecodeRunsOnCommonFP32) {
    auto param = MakeYoloDecodeParam(false);
    auto op = std::make_shared<feather::operators::YoloDecodeOp>("decode0", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::COMMON, DataType::FP32, "YoloDecode");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    const auto expected = ExpectedDecoded();
    ASSERT_EQ(param.out->dims().data(), std::vector<int64_t>({1, 2, 6}));
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(param.out->data<float>()[i], expected[i], 1e-5f);
    }
}

TEST(yolo_decode_op_test, YoloDecodeRunsOnCommonFP16) {
    auto param = MakeYoloDecodeParam(true);
    auto op = std::make_shared<feather::operators::YoloDecodeOp>("decode_fp16", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::COMMON, DataType::FP16, "YoloDecode");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    const auto expected = ExpectedDecoded();
    ASSERT_EQ(param.out->dims().data(), std::vector<int64_t>({1, 2, 6}));
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(param.out->data<uint16_t>()[i]), expected[i], 0.2f);
    }
}
