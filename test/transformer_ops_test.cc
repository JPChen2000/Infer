#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "core/graph.h"
#include "core/graph_lowering.h"
#include "core/operator_registry.h"
#include "core/static_graph.h"
#include "core/tensor.h"
#include "model/model_format.h"
#include "util/fp16.h"

namespace {

constexpr feather::DataType kBoolDataType = feather::DataType::BOOL;

feather::model::ValueDesc MakeValue(const std::string& name, const std::vector<int64_t>& dims,
                                    feather::DataType data_type, bool constant = false) {
    feather::model::ValueDesc value;
    value.tensor.name = name;
    value.tensor.dims = dims;
    value.tensor.data_type = data_type;
    value.constant = constant;
    return value;
}

feather::model::NodeDesc MakeNode(const std::string& name, const std::string& op_type,
                                  const std::vector<std::string>& inputs, const std::string& output) {
    feather::model::NodeDesc node;
    node.name = name;
    node.op_type = op_type;
    node.inputs = inputs;
    node.outputs = {output};
    return node;
}

std::shared_ptr<feather::Tensor> MakeHalfTensor(const std::vector<float>& values,
                                                 const std::vector<int64_t>& dims) {
    std::vector<uint16_t> storage;
    storage.reserve(values.size());
    for (const float value : values) {
        storage.push_back(feather::FloatToHalf(value));
    }
    auto tensor = std::make_shared<feather::Tensor>();
    tensor->Assign<uint16_t>(storage, dims);
    return tensor;
}

}  // namespace

TEST(transformer_ops_test, RunsAttentionMaskSubgraphOnCpuWithOriginalOperators) {
    feather::model::ModelDesc model;
    model.name = "attention_mask_subgraph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"attention_mask"};
    model.graph.outputs = {"where_out"};
    model.graph.values = {
        MakeValue("attention_mask", {1, 4}, feather::DataType::INT64),
        MakeValue("zero", {}, feather::DataType::INT64, true),
        MakeValue("equal_out", {1, 4}, kBoolDataType),
        MakeValue("reshape_out", {1, 1, 1, 4}, kBoolDataType),
        MakeValue("scores", {1, 2, 4, 4}, feather::DataType::FP32),
        MakeValue("shape_out", {4}, feather::DataType::INT64),
        MakeValue("expand_out", {1, 2, 4, 4}, kBoolDataType),
        MakeValue("cast_out", {1, 2, 4, 4}, kBoolDataType),
        MakeValue("negative_infinity", {}, feather::DataType::FP32, true),
        MakeValue("where_out", {1, 2, 4, 4}, feather::DataType::FP32),
    };

    auto equal = MakeNode("equal", "Equal", {"attention_mask", "zero"}, "equal_out");
    auto reshape = MakeNode("reshape", "Reshape", {"equal_out"}, "reshape_out");
    reshape.attributes["shape"] = std::vector<int64_t>{1, 1, 1, 4};
    auto shape = MakeNode("shape", "Shape", {"scores"}, "shape_out");
    auto expand = MakeNode("expand", "Expand", {"reshape_out", "shape_out"}, "expand_out");
    auto cast = MakeNode("cast", "Cast", {"expand_out"}, "cast_out");
    cast.attributes["to"] = int64_t{9};
    auto where = MakeNode("where", "Where", {"cast_out", "negative_infinity", "scores"}, "where_out");
    model.graph.nodes = {equal, reshape, shape, expand, cast, where};

    auto attention_mask = std::make_shared<feather::Tensor>();
    attention_mask->Assign<int64_t>({1, 0, 1, 0}, {1, 4});
    auto zero = std::make_shared<feather::Tensor>();
    zero->Assign<int64_t>({0}, {});
    auto scores = std::make_shared<feather::Tensor>();
    std::vector<float> score_data(32, 0.0f);
    for (size_t i = 0; i < score_data.size(); ++i) {
        score_data[i] = static_cast<float>(i + 1);
    }
    scores->Assign<float>(score_data, {1, 2, 4, 4});
    auto negative_infinity = std::make_shared<feather::Tensor>();
    negative_infinity->Assign<float>({-INFINITY}, {});

    feather::StaticGraph static_graph;
    ASSERT_EQ(static_graph.SetModel(model), 0);
    static_graph.SetKernelDevice(feather::DeviceType::COMMON);
    ASSERT_EQ(static_graph.SetTensor("attention_mask", attention_mask), 0);
    ASSERT_EQ(static_graph.SetTensor("zero", zero), 0);
    ASSERT_EQ(static_graph.SetTensor("scores", scores), 0);
    ASSERT_EQ(static_graph.SetTensor("negative_infinity", negative_infinity), 0);
    ASSERT_EQ(static_graph.Build(), 0);

    ASSERT_EQ(static_graph.nodes().size(), 6U);
    EXPECT_EQ(static_graph.nodes()[0].op_type, "Equal");
    EXPECT_EQ(static_graph.nodes()[2].op_type, "Shape");
    EXPECT_EQ(static_graph.nodes()[3].op_type, "Expand");
    EXPECT_EQ(static_graph.nodes()[5].op_type, "Where");

    feather::RuntimeGraph runtime_graph;
    runtime_graph.SetThreadMode(feather::RuntimeThreadMode::kSerialGraph);
    feather::GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(static_graph, &runtime_graph), 0);
    ASSERT_EQ(runtime_graph.Run(), 0);

    auto shape_out = runtime_graph.GetTensor("shape_out");
    ASSERT_NE(shape_out, nullptr);
    ASSERT_EQ(shape_out->data_type(), feather::DataType::INT64);
    EXPECT_EQ(shape_out->dims().data(), std::vector<int64_t>({4}));
    EXPECT_EQ(std::vector<int64_t>(shape_out->data<int64_t>(), shape_out->data<int64_t>() + 4),
              (std::vector<int64_t>{1, 2, 4, 4}));

    auto expanded_mask = runtime_graph.GetTensor("expand_out");
    ASSERT_NE(expanded_mask, nullptr);
    ASSERT_EQ(expanded_mask->data_type(), kBoolDataType);
    for (int64_t i = 0; i < expanded_mask->numel(); ++i) {
        const int64_t key_position = i % 4;
        const uint8_t expected = key_position == 1 || key_position == 3 ? 1 : 0;
        EXPECT_EQ(expanded_mask->data<uint8_t>()[i], expected);
    }

    auto cast_mask = runtime_graph.GetTensor("cast_out");
    ASSERT_NE(cast_mask, nullptr);
    ASSERT_EQ(cast_mask->data_type(), kBoolDataType);
    ASSERT_EQ(cast_mask->dims().data(), std::vector<int64_t>({1, 2, 4, 4}));
    for (int64_t i = 0; i < cast_mask->numel(); ++i) {
        const int64_t key_position = i % 4;
        const uint8_t expected = key_position == 1 || key_position == 3 ? 1 : 0;
        EXPECT_EQ(cast_mask->data<uint8_t>()[i], expected);
    }

    auto where_out = runtime_graph.GetTensor("where_out");
    ASSERT_NE(where_out, nullptr);
    ASSERT_EQ(where_out->data_type(), feather::DataType::FP32);
    for (int64_t i = 0; i < where_out->numel(); ++i) {
        const int64_t key_position = i % 4;
        if (key_position == 1 || key_position == 3) {
            EXPECT_TRUE(std::isinf(where_out->data<float>()[i]));
            EXPECT_LT(where_out->data<float>()[i], 0.0f);
        } else {
            EXPECT_FLOAT_EQ(where_out->data<float>()[i], score_data[static_cast<size_t>(i)]);
        }
    }
}

TEST(transformer_ops_test, PreservesTensorControlInputsForUnsqueezeConcatAndReshape) {
    feather::model::ModelDesc model;
    model.name = "tensor_control_reshape";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"data"};
    model.graph.outputs = {"reshaped"};
    model.graph.values = {
        MakeValue("data", {2, 3}, feather::DataType::FP32),
        MakeValue("dimension_zero", {}, feather::DataType::INT64, true),
        MakeValue("dimension_one", {}, feather::DataType::INT64, true),
        MakeValue("axes", {1}, feather::DataType::INT64, true),
        MakeValue("shape_head", {1}, feather::DataType::INT64),
        MakeValue("shape_tail", {1}, feather::DataType::INT64),
        MakeValue("target_shape", {2}, feather::DataType::INT64),
        MakeValue("reshaped", {3, 2}, feather::DataType::FP32),
    };

    const auto unsqueeze_head = MakeNode("unsqueeze_head", "Unsqueeze", {"dimension_zero", "axes"}, "shape_head");
    const auto unsqueeze_tail = MakeNode("unsqueeze_tail", "Unsqueeze", {"dimension_one", "axes"}, "shape_tail");
    auto concat = MakeNode("concat_shape", "Concat", {"shape_head", "shape_tail"}, "target_shape");
    concat.attributes["axis"] = int64_t{0};
    const auto reshape = MakeNode("reshape", "Reshape", {"data", "target_shape"}, "reshaped");
    model.graph.nodes = {unsqueeze_head, unsqueeze_tail, concat, reshape};

    auto data = std::make_shared<feather::Tensor>();
    data->Assign<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    auto dimension_zero = std::make_shared<feather::Tensor>();
    dimension_zero->Assign<int64_t>({3}, {});
    auto dimension_one = std::make_shared<feather::Tensor>();
    dimension_one->Assign<int64_t>({2}, {});
    auto axes = std::make_shared<feather::Tensor>();
    axes->Assign<int64_t>({0}, {1});

    feather::StaticGraph static_graph;
    ASSERT_EQ(static_graph.SetModel(model), 0);
    static_graph.SetKernelDevice(feather::DeviceType::COMMON);
    ASSERT_EQ(static_graph.SetTensor("data", data), 0);
    ASSERT_EQ(static_graph.SetTensor("dimension_zero", dimension_zero), 0);
    ASSERT_EQ(static_graph.SetTensor("dimension_one", dimension_one), 0);
    ASSERT_EQ(static_graph.SetTensor("axes", axes), 0);
    ASSERT_EQ(static_graph.Build(), 0);

    ASSERT_EQ(static_graph.nodes().size(), 4U);
    EXPECT_EQ(static_graph.nodes()[0].inputs, (std::vector<std::string>{"dimension_zero", "axes"}));
    EXPECT_EQ(static_graph.nodes()[1].inputs, (std::vector<std::string>{"dimension_one", "axes"}));
    EXPECT_EQ(static_graph.nodes()[3].inputs, (std::vector<std::string>{"data", "target_shape"}));

    feather::RuntimeGraph runtime_graph;
    runtime_graph.SetThreadMode(feather::RuntimeThreadMode::kSerialGraph);
    feather::GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(static_graph, &runtime_graph), 0);
    ASSERT_EQ(runtime_graph.Run(), 0);

    const auto target_shape = runtime_graph.GetTensor("target_shape");
    ASSERT_NE(target_shape, nullptr);
    ASSERT_EQ(target_shape->data_type(), feather::DataType::INT64);
    EXPECT_EQ(std::vector<int64_t>(target_shape->data<int64_t>(), target_shape->data<int64_t>() + 2),
              (std::vector<int64_t>{3, 2}));

    const auto reshaped = runtime_graph.GetTensor("reshaped");
    ASSERT_NE(reshaped, nullptr);
    EXPECT_EQ(reshaped->dims().data(), (std::vector<int64_t>{3, 2}));
    EXPECT_EQ(std::vector<float>(reshaped->data<float>(), reshaped->data<float>() + 6),
              (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
}

TEST(transformer_ops_test, ResolvesZeroAndInferredDimensionsFromShapeTensor) {
    feather::model::ModelDesc model;
    model.name = "reshape_shape_semantics";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"data"};
    model.graph.outputs = {"reshaped"};
    model.graph.values = {
        MakeValue("data", {2, 3, 4}, feather::DataType::FP32),
        MakeValue("shape", {3}, feather::DataType::INT64, true),
        MakeValue("reshaped", {2, 12, 1}, feather::DataType::FP32),
    };
    model.graph.nodes = {MakeNode("reshape", "Reshape", {"data", "shape"}, "reshaped")};

    auto data = std::make_shared<feather::Tensor>();
    std::vector<float> values(24, 0.0f);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i);
    }
    data->Assign<float>(values, {2, 3, 4});
    auto shape = std::make_shared<feather::Tensor>();
    shape->Assign<int64_t>({0, -1, 1}, {3});

    feather::StaticGraph static_graph;
    ASSERT_EQ(static_graph.SetModel(model), 0);
    static_graph.SetKernelDevice(feather::DeviceType::COMMON);
    ASSERT_EQ(static_graph.SetTensor("data", data), 0);
    ASSERT_EQ(static_graph.SetTensor("shape", shape), 0);
    ASSERT_EQ(static_graph.Build(), 0);

    feather::RuntimeGraph runtime_graph;
    runtime_graph.SetThreadMode(feather::RuntimeThreadMode::kSerialGraph);
    feather::GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(static_graph, &runtime_graph), 0);
    ASSERT_EQ(runtime_graph.Run(), 0);

    const auto reshaped = runtime_graph.GetTensor("reshaped");
    ASSERT_NE(reshaped, nullptr);
    EXPECT_EQ(reshaped->dims().data(), (std::vector<int64_t>{2, 12, 1}));
    EXPECT_EQ(std::vector<float>(reshaped->data<float>(), reshaped->data<float>() + 24), values);
}

TEST(transformer_ops_test, ReduceMeanPreservesFp16OnCpu) {
    feather::OperatorRegistry::TensorMap tensors;
    auto input = std::make_shared<feather::Tensor>();
    input->Assign<uint16_t>({feather::FloatToHalf(1.0f), feather::FloatToHalf(3.0f), feather::FloatToHalf(5.0f),
                             feather::FloatToHalf(7.0f)},
                            {2, 2});
    tensors["input"] = input;
    tensors["output"] = std::make_shared<feather::Tensor>(std::vector<int64_t>{2, 1});

    feather::model::NodeDesc node;
    node.name = "reduce_mean_fp16";
    node.op_type = "ReduceMean";
    node.inputs = {"input"};
    node.outputs = {"output"};
    node.attributes["axes"] = std::vector<int64_t>{1};
    node.attributes["keepdims"] = int64_t{1};

    feather::KernelDeviceScope scope(feather::DeviceType::COMMON);
    auto op = feather::OperatorRegistry::instance().Create(node, tensors);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->Run(), 0);

    auto output = op->outputs().front();
    ASSERT_NE(output, nullptr);
    EXPECT_EQ(output->data_type(), feather::DataType::FP16);
    EXPECT_EQ(output->dims().data(), (std::vector<int64_t>{2, 1}));
    EXPECT_NEAR(feather::HalfToFloat(output->data<uint16_t>()[0]), 2.0f, 3e-3f);
    EXPECT_NEAR(feather::HalfToFloat(output->data<uint16_t>()[1]), 6.0f, 3e-3f);
}

TEST(transformer_ops_test, UnsqueezeAndSqueezePreserveFp16OnCpu) {
    feather::OperatorRegistry::TensorMap tensors;
    tensors["input"] = MakeHalfTensor({1.0f, 2.0f}, {2});
    tensors["unsqueezed"] = std::make_shared<feather::Tensor>(std::vector<int64_t>{1, 2, 1});

    auto unsqueeze = MakeNode("unsqueeze_fp16", "Unsqueeze", {"input"}, "unsqueezed");
    unsqueeze.attributes["axes"] = std::vector<int64_t>{0, 2};

    feather::KernelDeviceScope scope(feather::DeviceType::COMMON);
    auto unsqueeze_op = feather::OperatorRegistry::instance().Create(unsqueeze, tensors);
    ASSERT_NE(unsqueeze_op, nullptr);
    ASSERT_EQ(unsqueeze_op->Run(), 0);

    auto unsqueezed = unsqueeze_op->outputs().front();
    ASSERT_NE(unsqueezed, nullptr);
    EXPECT_EQ(unsqueezed->data_type(), feather::DataType::FP16);
    EXPECT_EQ(unsqueezed->dims().data(), (std::vector<int64_t>{1, 2, 1}));
    EXPECT_NEAR(feather::HalfToFloat(unsqueezed->data<uint16_t>()[0]), 1.0f, 3e-3f);
    EXPECT_NEAR(feather::HalfToFloat(unsqueezed->data<uint16_t>()[1]), 2.0f, 3e-3f);

    tensors["unsqueezed"] = unsqueezed;
    tensors["squeezed"] = std::make_shared<feather::Tensor>(std::vector<int64_t>{2});
    auto squeeze = MakeNode("squeeze_fp16", "Squeeze", {"unsqueezed"}, "squeezed");
    squeeze.attributes["axes"] = std::vector<int64_t>{0, 2};
    auto squeeze_op = feather::OperatorRegistry::instance().Create(squeeze, tensors);
    ASSERT_NE(squeeze_op, nullptr);
    ASSERT_EQ(squeeze_op->Run(), 0);

    auto squeezed = squeeze_op->outputs().front();
    ASSERT_NE(squeezed, nullptr);
    EXPECT_EQ(squeezed->data_type(), feather::DataType::FP16);
    EXPECT_EQ(squeezed->dims().data(), (std::vector<int64_t>{2}));
    EXPECT_NEAR(feather::HalfToFloat(squeezed->data<uint16_t>()[0]), 1.0f, 3e-3f);
    EXPECT_NEAR(feather::HalfToFloat(squeezed->data<uint16_t>()[1]), 2.0f, 3e-3f);
}

TEST(transformer_ops_test, GatherPreservesFp16OnCpu) {
    feather::OperatorRegistry::TensorMap tensors;
    tensors["data"] = MakeHalfTensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2});
    auto indices = std::make_shared<feather::Tensor>();
    indices->Assign<int64_t>({2, 0}, {2});
    tensors["indices"] = indices;
    tensors["output"] = std::make_shared<feather::Tensor>(std::vector<int64_t>{2, 2});

    auto node = MakeNode("gather_fp16", "Gather", {"data", "indices"}, "output");
    node.attributes["axis"] = int64_t{0};

    feather::KernelDeviceScope scope(feather::DeviceType::COMMON);
    auto op = feather::OperatorRegistry::instance().Create(node, tensors);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->Run(), 0);

    auto output = op->outputs().front();
    ASSERT_NE(output, nullptr);
    EXPECT_EQ(output->data_type(), feather::DataType::FP16);
    const std::vector<float> expected = {5.0f, 6.0f, 1.0f, 2.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(output->data<uint16_t>()[i]), expected[i], 3e-3f);
    }
}

TEST(transformer_ops_test, CastFp16ToFp32UsesCommonFp16Kernel) {
    feather::OperatorRegistry::TensorMap tensors;
    tensors["input"] = MakeHalfTensor({1.5f, -2.25f}, {2});
    tensors["output"] = std::make_shared<feather::Tensor>(std::vector<int64_t>{2});

    auto node = MakeNode("cast_fp16_to_fp32", "Cast", {"input"}, "output");
    node.attributes["to"] = int64_t{1};

    feather::KernelDeviceScope scope(feather::DeviceType::COMMON);
    auto op = feather::OperatorRegistry::instance().Create(node, tensors);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->Run(), 0);

    const auto output = op->outputs().front();
    ASSERT_NE(output, nullptr);
    EXPECT_EQ(output->data_type(), feather::DataType::FP32);
    EXPECT_FLOAT_EQ(output->data<float>()[0], 1.5f);
    EXPECT_FLOAT_EQ(output->data<float>()[1], -2.25f);

    auto kernel = op->DetachKernel();
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(kernel->device(), feather::DeviceType::COMMON);
    EXPECT_EQ(kernel->data_type(), feather::DataType::FP16);
}
