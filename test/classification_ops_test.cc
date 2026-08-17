#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/operator_registry.h"
#include "core/tensor.h"
#include "model/model_format.h"
#include "src/operator/div_op.h"
#include "src/operator/erf_op.h"
#include "src/operator/sqrt_op.h"
#include "src/operator/sub_op.h"
#include "src/operator/tanh_op.h"
#include "util/fp16.h"

using feather::DataType;
using feather::DeviceType;
using feather::KernelDeviceScope;
using feather::OpBase;
using feather::OperatorRegistry;
using feather::Tensor;

namespace {

std::shared_ptr<Tensor> MakeTensor(const std::vector<float>& data, const std::vector<int64_t>& dims) {
    auto tensor = std::make_shared<Tensor>();
    tensor->Assign<float>(data, dims);
    return tensor;
}

std::shared_ptr<Tensor> MakeHalfTensor(const std::vector<float>& data, const std::vector<int64_t>& dims) {
    std::vector<uint16_t> storage;
    storage.reserve(data.size());
    for (const float value : data) {
        storage.push_back(feather::FloatToHalf(value));
    }
    auto tensor = std::make_shared<Tensor>();
    tensor->Assign<uint16_t>(storage, dims);
    return tensor;
}

}  // namespace

TEST(classification_ops_test, BatchNormalizationRunsThroughRegistryOnCpu) {
    OperatorRegistry::TensorMap tensors;
    tensors["input"] = MakeTensor({1.0f, 2.0f, 5.0f, 9.0f, 2.0f, 4.0f, 6.0f, 8.0f}, {1, 2, 2, 2});
    tensors["scale"] = MakeTensor({2.0f, 3.0f}, {2});
    tensors["bias"] = MakeTensor({0.5f, -1.0f}, {2});
    tensors["mean"] = MakeTensor({1.0f, 2.0f}, {2});
    tensors["var"] = MakeTensor({4.0f, 9.0f}, {2});
    tensors["output"] = std::make_shared<Tensor>(std::vector<int64_t>{1, 2, 2, 2});

    feather::model::NodeDesc node;
    node.name = "bn0";
    node.op_type = "BatchNormalization";
    node.inputs = {"input", "scale", "bias", "mean", "var"};
    node.outputs = {"output"};
    node.attributes["epsilon"] = 0.0f;

    KernelDeviceScope scope(DeviceType::X86);
    auto op = OperatorRegistry::instance().Create(node, tensors);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);
    ASSERT_EQ(op->Run(), 0);

    auto out = op->outputs().front();
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->dims().data(), std::vector<int64_t>({1, 2, 2, 2}));
    ASSERT_EQ(out->data_type(), DataType::FP32);
    const std::vector<float> expected = {
        0.5f, 1.5f, 4.5f, 8.5f,
        -1.0f, 1.0f, 3.0f, 5.0f,
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(out->data<float>()[i], expected[i], 1e-5f);
    }
}

TEST(classification_ops_test, GlobalAveragePoolRunsThroughRegistryOnCpu) {
    OperatorRegistry::TensorMap tensors;
    tensors["input"] = MakeTensor({
                                      1.0f, 2.0f,
                                      3.0f, 4.0f,
                                      10.0f, 20.0f,
                                      30.0f, 40.0f,
                                  },
                                  {1, 2, 2, 2});
    tensors["output"] = std::make_shared<Tensor>(std::vector<int64_t>{1, 2, 1, 1});

    feather::model::NodeDesc node;
    node.name = "gap0";
    node.op_type = "GlobalAveragePool";
    node.inputs = {"input"};
    node.outputs = {"output"};

    KernelDeviceScope scope(DeviceType::X86);
    auto op = OperatorRegistry::instance().Create(node, tensors);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);
    ASSERT_EQ(op->Run(), 0);

    auto out = tensors["output"];
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->dims().data(), std::vector<int64_t>({1, 2, 1, 1}));
    ASSERT_EQ(out->data_type(), DataType::FP32);
    EXPECT_FLOAT_EQ(out->data<float>()[0], 2.5f);
    EXPECT_FLOAT_EQ(out->data<float>()[1], 25.0f);
}

TEST(classification_ops_test, SubAndDivRunThroughCommonRegistryInFp16) {
    OperatorRegistry::TensorMap tensors;
    tensors["lhs"] = MakeHalfTensor({4.0f, 9.0f, 8.0f, 12.0f}, {2, 2});
    tensors["rhs"] = MakeHalfTensor({2.0f, 3.0f}, {1, 2});
    tensors["output"] = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});

    feather::model::NodeDesc node;
    node.name = "binary_fp16";
    node.inputs = {"lhs", "rhs"};
    node.outputs = {"output"};

    KernelDeviceScope scope(DeviceType::COMMON);
    for (const auto& item : std::vector<std::pair<std::string, std::vector<float>>>{
             {"Sub", {2.0f, 6.0f, 6.0f, 9.0f}},
             {"Div", {2.0f, 3.0f, 4.0f, 4.0f}},
         }) {
        node.op_type = item.first;
        auto op = OperatorRegistry::instance().Create(node, tensors);
        ASSERT_NE(op, nullptr) << item.first;
        if (item.first == "Sub") {
            EXPECT_NE(dynamic_cast<feather::operators::SubOp*>(op.get()), nullptr);
        } else {
            EXPECT_NE(dynamic_cast<feather::operators::DivOp*>(op.get()), nullptr);
        }
        ASSERT_EQ(op->Run(), 0) << item.first;

        const auto output = op->outputs().front();
        ASSERT_NE(output, nullptr);
        ASSERT_EQ(output->data_type(), DataType::FP16);
        for (size_t i = 0; i < item.second.size(); ++i) {
            EXPECT_NEAR(feather::HalfToFloat(output->data<uint16_t>()[i]), item.second[i], 3e-3f) << item.first;
        }
    }
}

TEST(classification_ops_test, UnaryElementwiseOperatorsRunThroughCommonRegistryInFp16) {
    OperatorRegistry::TensorMap tensors;
    tensors["input"] = MakeHalfTensor({0.0f, 1.0f, 4.0f}, {3});
    tensors["output"] = std::make_shared<Tensor>(std::vector<int64_t>{3});

    feather::model::NodeDesc node;
    node.name = "unary_fp16";
    node.inputs = {"input"};
    node.outputs = {"output"};

    KernelDeviceScope scope(DeviceType::COMMON);
    for (const auto& item : std::vector<std::pair<std::string, std::vector<float>>>{
             {"Sqrt", {0.0f, 1.0f, 2.0f}},
             {"Tanh", {0.0f, std::tanh(1.0f), std::tanh(4.0f)}},
             {"Erf", {0.0f, std::erf(1.0f), std::erf(4.0f)}},
         }) {
        node.op_type = item.first;
        auto op = OperatorRegistry::instance().Create(node, tensors);
        ASSERT_NE(op, nullptr) << item.first;
        if (item.first == "Sqrt") {
            EXPECT_NE(dynamic_cast<feather::operators::SqrtOp*>(op.get()), nullptr);
        } else if (item.first == "Tanh") {
            EXPECT_NE(dynamic_cast<feather::operators::TanhOp*>(op.get()), nullptr);
        } else {
            EXPECT_NE(dynamic_cast<feather::operators::ErfOp*>(op.get()), nullptr);
        }
        ASSERT_EQ(op->Run(), 0) << item.first;

        const auto output = op->outputs().front();
        ASSERT_NE(output, nullptr);
        ASSERT_EQ(output->data_type(), DataType::FP16);
        for (size_t i = 0; i < item.second.size(); ++i) {
            EXPECT_NEAR(feather::HalfToFloat(output->data<uint16_t>()[i]), item.second[i], 3e-3f) << item.first;
        }
    }
}
