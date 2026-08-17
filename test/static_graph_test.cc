#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/graph.h"
#include "core/graph_lowering.h"
#include "core/static_graph.h"
#include "core/tensor.h"
#include "model/model_format.h"

using feather::DataType;
using feather::StaticGraph;
using feather::Tensor;
using feather::model::ModelDesc;
using feather::model::NodeDesc;
using feather::model::ValueDesc;

TEST(static_graph_test, BuildStaticGraphFromModelDesc) {
    ModelDesc model;
    model.name = "fc_static_graph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = {"output"};

    ValueDesc input;
    input.tensor.name = "input";
    input.tensor.dims = {3, 2};
    input.tensor.data_type = DataType::FP32;

    ValueDesc weight;
    weight.tensor.name = "weight";
    weight.tensor.dims = {2, 4};
    weight.tensor.data_type = DataType::FP32;
    weight.constant = true;

    ValueDesc bias;
    bias.tensor.name = "bias";
    bias.tensor.dims = {3, 4};
    bias.tensor.data_type = DataType::FP32;
    bias.constant = true;

    ValueDesc output;
    output.tensor.name = "output";
    output.tensor.dims = {3, 4};
    output.tensor.data_type = DataType::FP32;

    NodeDesc node;
    node.name = "fc0";
    node.op_type = "FC";
    node.inputs = {"input", "weight", "bias"};
    node.outputs = {"output"};

    model.graph.values = {input, weight, bias, output};
    model.graph.nodes = {node};

    auto input_tensor = std::make_shared<Tensor>();
    input_tensor->Assign<float>({1, 2, 3, 4, 5, 6}, {3, 2});

    auto weight_tensor = std::make_shared<Tensor>();
    weight_tensor->Assign<float>({1, 2, 3, 4, 5, 6, 7, 8}, {2, 4});

    auto bias_tensor = std::make_shared<Tensor>();
    bias_tensor->Assign<float>({1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}, {3, 4});

    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(model), 0);
    ASSERT_EQ(graph.SetTensor("input", input_tensor), 0);
    ASSERT_EQ(graph.SetTensor("weight", weight_tensor), 0);
    ASSERT_EQ(graph.SetTensor("bias", bias_tensor), 0);

    ASSERT_EQ(graph.Build(), 0);
    ASSERT_EQ(graph.OperatorSize(), 1U);

    auto output_tensor = graph.GetTensor("output");
    ASSERT_NE(output_tensor, nullptr);
    EXPECT_EQ(output_tensor->dims().data(), std::vector<int64_t>({3, 4}));
}

TEST(static_graph_test, RejectsUnknownOperatorType) {
    ModelDesc model;
    model.name = "bad_graph";
    model.version = 1;
    model.graph.name = "main";

    ValueDesc input;
    input.tensor.name = "input";
    input.tensor.dims = {1, 2};
    input.tensor.data_type = DataType::FP32;

    ValueDesc output;
    output.tensor.name = "output";
    output.tensor.dims = {1, 2};
    output.tensor.data_type = DataType::FP32;

    NodeDesc node;
    node.name = "mystery";
    node.op_type = "Unknown";
    node.inputs = {"input"};
    node.outputs = {"output"};

    model.graph.values = {input, output};
    model.graph.nodes = {node};

    auto input_tensor = std::make_shared<Tensor>();
    input_tensor->Assign<float>({1, 2}, {1, 2});

    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(model), 0);
    ASSERT_EQ(graph.SetTensor("input", input_tensor), 0);
    EXPECT_EQ(graph.Build(), -1);
}

TEST(static_graph_test, BuildStaticGraphFromGemmModelDesc) {
    ModelDesc model;
    model.name = "gemm_static_graph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = {"output"};

    ValueDesc input;
    input.tensor.name = "input";
    input.tensor.dims = {2, 3};
    input.tensor.data_type = DataType::FP32;

    ValueDesc weight;
    weight.tensor.name = "weight";
    weight.tensor.dims = {3, 2};
    weight.tensor.data_type = DataType::FP32;
    weight.constant = true;

    ValueDesc output;
    output.tensor.name = "output";
    output.tensor.dims = {2, 2};
    output.tensor.data_type = DataType::FP32;

    NodeDesc node;
    node.name = "gemm0";
    node.op_type = "Gemm";
    node.inputs = {"input", "weight"};
    node.outputs = {"output"};

    model.graph.values = {input, weight, output};
    model.graph.nodes = {node};

    auto input_tensor = std::make_shared<Tensor>();
    input_tensor->Assign<float>({1, 2, 3, 4, 5, 6}, {2, 3});

    auto weight_tensor = std::make_shared<Tensor>();
    weight_tensor->Assign<float>({1, 2, 3, 4, 5, 6}, {3, 2});

    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(model), 0);
    ASSERT_EQ(graph.SetTensor("input", input_tensor), 0);
    ASSERT_EQ(graph.SetTensor("weight", weight_tensor), 0);

    ASSERT_EQ(graph.Build(), 0);
    ASSERT_EQ(graph.OperatorSize(), 1U);

    auto output_tensor = graph.GetTensor("output");
    ASSERT_NE(output_tensor, nullptr);
    EXPECT_EQ(output_tensor->dims().data(), std::vector<int64_t>({2, 2}));
}

TEST(static_graph_test, PreservesDeclaredTypeAfterDynamicShapeAllocation) {
    ModelDesc model;
    model.name = "dynamic_shape_type_propagation";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = {"output"};

    ValueDesc input;
    input.tensor.name = "input";
    input.tensor.dims = {1, 2, 2};
    input.tensor.data_type = DataType::FP32;

    ValueDesc indices;
    indices.tensor.name = "indices";
    indices.tensor.dims = {1};
    indices.tensor.data_type = DataType::INT64;
    indices.constant = true;

    ValueDesc identity_output;
    identity_output.tensor.name = "identity_output";
    identity_output.tensor.dims = {0, 0, 0};
    identity_output.tensor.data_type = DataType::FP32;

    ValueDesc output;
    output.tensor.name = "output";
    output.tensor.dims = {0, 0, 0};
    output.tensor.data_type = DataType::FP32;

    NodeDesc identity;
    identity.name = "identity";
    identity.op_type = "Identity";
    identity.inputs = {"input"};
    identity.outputs = {"identity_output"};

    NodeDesc gather;
    gather.name = "gather";
    gather.op_type = "Gather";
    gather.inputs = {"identity_output", "indices"};
    gather.outputs = {"output"};
    gather.attributes["axis"] = int64_t{1};

    model.graph.values = {input, indices, identity_output, output};
    model.graph.nodes = {identity, gather};

    auto input_tensor = std::make_shared<Tensor>();
    input_tensor->Assign<float>({1.0f, 2.0f, 3.0f, 4.0f}, {1, 2, 2});
    auto indices_tensor = std::make_shared<Tensor>();
    indices_tensor->Assign<int64_t>({0}, {1});

    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(model), 0);
    ASSERT_EQ(graph.SetTensor("input", input_tensor), 0);
    ASSERT_EQ(graph.SetTensor("indices", indices_tensor), 0);
    ASSERT_EQ(graph.Build(), 0);

    auto identity_tensor = graph.GetTensor("identity_output");
    ASSERT_NE(identity_tensor, nullptr);
    EXPECT_EQ(identity_tensor->data_type(), DataType::FP32);
    EXPECT_EQ(identity_tensor->dims().data(), std::vector<int64_t>({1, 2, 2}));

    auto output_tensor = graph.GetTensor("output");
    ASSERT_NE(output_tensor, nullptr);
    EXPECT_EQ(output_tensor->data_type(), DataType::FP32);
    EXPECT_EQ(output_tensor->dims().data(), std::vector<int64_t>({1, 1, 2}));
}

TEST(static_graph_test, RetainsStaticControlNodeAndInputsAfterShapeEvaluation) {
    ModelDesc model;
    model.name = "static_control_node";
    model.version = 1;
    model.graph.name = "main";
    model.graph.outputs = {"output"};

    ValueDesc data;
    data.tensor.name = "data";
    data.tensor.dims = {2};
    data.tensor.data_type = DataType::INT64;
    data.constant = true;

    ValueDesc axes;
    axes.tensor.name = "axes";
    axes.tensor.dims = {1};
    axes.tensor.data_type = DataType::INT64;
    axes.constant = true;

    ValueDesc output;
    output.tensor.name = "output";
    output.tensor.dims = {2, 1};
    output.tensor.data_type = DataType::INT64;

    NodeDesc unsqueeze;
    unsqueeze.name = "unsqueeze";
    unsqueeze.op_type = "Unsqueeze";
    unsqueeze.inputs = {"data", "axes"};
    unsqueeze.outputs = {"output"};

    model.graph.values = {data, axes, output};
    model.graph.nodes = {unsqueeze};

    auto data_tensor = std::make_shared<Tensor>();
    data_tensor->Assign<int64_t>({7, 9}, {2});
    auto axes_tensor = std::make_shared<Tensor>();
    axes_tensor->Assign<int64_t>({1}, {1});

    StaticGraph static_graph;
    ASSERT_EQ(static_graph.SetModel(model), 0);
    ASSERT_EQ(static_graph.SetTensor("data", data_tensor), 0);
    ASSERT_EQ(static_graph.SetTensor("axes", axes_tensor), 0);
    ASSERT_EQ(static_graph.Build(), 0);
    ASSERT_EQ(static_graph.NodeSize(), 1U);
    ASSERT_EQ(static_graph.OperatorSize(), 1U);

    const auto* static_node = static_graph.GetNode("unsqueeze");
    ASSERT_NE(static_node, nullptr);
    EXPECT_EQ(static_node->op_type, "Unsqueeze");
    EXPECT_EQ(static_node->inputs, std::vector<std::string>({"data", "axes"}));
    EXPECT_EQ(static_node->outputs, std::vector<std::string>({"output"}));

    feather::RuntimeGraph runtime_graph;
    feather::GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(static_graph, &runtime_graph), 0);
    ASSERT_EQ(runtime_graph.NodeSize(), 1U);
    const auto* runtime_node = runtime_graph.GetNode("unsqueeze");
    ASSERT_NE(runtime_node, nullptr);
    EXPECT_EQ(runtime_node->inputs, std::vector<std::string>({"data", "axes"}));
    ASSERT_EQ(runtime_graph.Run(), 0);

    const auto output_tensor = runtime_graph.GetTensor("output");
    ASSERT_NE(output_tensor, nullptr);
    EXPECT_EQ(output_tensor->dims().data(), std::vector<int64_t>({2, 1}));
    EXPECT_EQ(output_tensor->data<int64_t>()[0], 7);
    EXPECT_EQ(output_tensor->data<int64_t>()[1], 9);
}
