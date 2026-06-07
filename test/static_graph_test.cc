#include <gtest/gtest.h>

#include <memory>
#include <vector>

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
