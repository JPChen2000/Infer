#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <typeinfo>
#include <unordered_map>

#include "core/graph_lowering.h"
#include "core/graph.h"
#include "core/static_graph.h"
#include "core/tensor.h"
#include "model/model_format.h"
#include "src/kernel/add.h"
#include "util/threading.h"

using feather::DataType;
using feather::DeviceType;
using feather::GraphLowering;
using feather::RuntimeGraph;
using feather::StaticGraph;
using feather::Tensor;
using feather::model::ModelDesc;
using feather::model::NodeDesc;
using feather::model::ValueDesc;

namespace {

void BuildAddRuntimeWithBackend(DeviceType device, RuntimeGraph* runtime_graph) {
    ModelDesc model;
    model.name = "add_graph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"lhs", "rhs"};
    model.graph.outputs = {"out"};

    ValueDesc lhs;
    lhs.tensor.name = "lhs";
    lhs.tensor.dims = {2};
    lhs.tensor.data_type = DataType::FP32;

    ValueDesc rhs;
    rhs.tensor.name = "rhs";
    rhs.tensor.dims = {2};
    rhs.tensor.data_type = DataType::FP32;

    ValueDesc out;
    out.tensor.name = "out";
    out.tensor.dims = {2};
    out.tensor.data_type = DataType::FP32;

    NodeDesc node;
    node.name = "add0";
    node.op_type = "Add";
    node.inputs = {"lhs", "rhs"};
    node.outputs = {"out"};

    model.graph.values = {lhs, rhs, out};
    model.graph.nodes = {node};

    auto lhs_tensor = std::make_shared<Tensor>();
    lhs_tensor->Assign<float>({1.0f, 2.0f}, {2});

    auto rhs_tensor = std::make_shared<Tensor>();
    rhs_tensor->Assign<float>({3.0f, 4.0f}, {2});

    StaticGraph static_graph;
    static_graph.SetKernelDevice(device);
    ASSERT_EQ(static_graph.SetModel(model), 0);
    ASSERT_EQ(static_graph.SetTensor("lhs", lhs_tensor), 0);
    ASSERT_EQ(static_graph.SetTensor("rhs", rhs_tensor), 0);
    ASSERT_EQ(static_graph.Build(), 0);

    GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(static_graph, runtime_graph), 0);
}

}  // namespace

TEST(runtime_graph_test, BuildRuntimeGraphFromStaticGraph) {
    ModelDesc model;
    model.name = "fc_graph";
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

    StaticGraph static_graph;
    ASSERT_EQ(static_graph.SetModel(model), 0);
    ASSERT_EQ(static_graph.SetTensor("input", input_tensor), 0);
    ASSERT_EQ(static_graph.SetTensor("weight", weight_tensor), 0);
    ASSERT_EQ(static_graph.SetTensor("bias", bias_tensor), 0);
    ASSERT_EQ(static_graph.Build(), 0);
    ASSERT_EQ(static_graph.OperatorSize(), 1U);

    RuntimeGraph runtime_graph;
    GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(static_graph, &runtime_graph), 0);
    ASSERT_EQ(runtime_graph.NodeSize(), 1U);
    ASSERT_EQ(runtime_graph.Run(), 0);

    auto output_tensor = runtime_graph.GetTensor("output");
    ASSERT_NE(output_tensor, nullptr);
    EXPECT_EQ(output_tensor->dims().data(), std::vector<int64_t>({3, 4}));

    std::vector<float> expected = {
        12, 15, 18, 21,
        25, 32, 39, 46,
        38, 49, 60, 71,
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(output_tensor->data<float>()[i], expected[i]);
    }
}

TEST(static_graph_test, KernelDeviceSelectsCommonBackend) {
    RuntimeGraph runtime_graph;
    BuildAddRuntimeWithBackend(DeviceType::COMMON, &runtime_graph);

    const auto* node = runtime_graph.GetNode("add0");
    ASSERT_NE(node, nullptr);
    ASSERT_NE(node->kernel, nullptr);
    EXPECT_EQ(typeid(*node->kernel),
              typeid(feather::kernel::AddKernel<DeviceType::COMMON, DataType::FP32>));
}

TEST(static_graph_test, KernelDeviceSelectsX86Backend) {
    RuntimeGraph runtime_graph;
    BuildAddRuntimeWithBackend(DeviceType::X86, &runtime_graph);

    const auto* node = runtime_graph.GetNode("add0");
    ASSERT_NE(node, nullptr);
    ASSERT_NE(node->kernel, nullptr);
    EXPECT_EQ(typeid(*node->kernel),
              typeid(feather::kernel::AddKernel<DeviceType::X86, DataType::FP32>));
}

TEST(static_graph_test, BuildFailsWhenRequiredTensorMissing) {
    ModelDesc model;
    model.name = "missing_weight";
    model.version = 1;
    model.graph.name = "main";

    ValueDesc input;
    input.tensor.name = "input";
    input.tensor.dims = {1, 2};
    input.tensor.data_type = DataType::FP32;

    ValueDesc weight;
    weight.tensor.name = "weight";
    weight.tensor.dims = {2, 2};
    weight.tensor.data_type = DataType::FP32;
    weight.constant = true;

    ValueDesc output;
    output.tensor.name = "output";
    output.tensor.dims = {1, 2};
    output.tensor.data_type = DataType::FP32;

    NodeDesc node;
    node.name = "fc0";
    node.op_type = "FC";
    node.inputs = {"input", "weight"};
    node.outputs = {"output"};

    model.graph.values = {input, weight, output};
    model.graph.nodes = {node};

    auto input_tensor = std::make_shared<Tensor>();
    input_tensor->Assign<float>({1, 2}, {1, 2});

    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(model), 0);
    ASSERT_EQ(graph.SetTensor("input", input_tensor), 0);
    EXPECT_EQ(graph.Build(), -1);
}

TEST(static_graph_test, BuildFailsWhenFcShapeIsInvalid) {
    ModelDesc model;
    model.name = "bad_fc_shape";
    model.version = 1;
    model.graph.name = "main";

    ValueDesc input;
    input.tensor.name = "input";
    input.tensor.dims = {1, 3};
    input.tensor.data_type = DataType::FP32;

    ValueDesc weight;
    weight.tensor.name = "weight";
    weight.tensor.dims = {2, 2};
    weight.tensor.data_type = DataType::FP32;
    weight.constant = true;

    ValueDesc output;
    output.tensor.name = "output";
    output.tensor.dims = {1, 2};
    output.tensor.data_type = DataType::FP32;

    NodeDesc node;
    node.name = "fc0";
    node.op_type = "FC";
    node.inputs = {"input", "weight"};
    node.outputs = {"output"};

    model.graph.values = {input, weight, output};
    model.graph.nodes = {node};

    auto input_tensor = std::make_shared<Tensor>();
    input_tensor->Assign<float>({1, 2, 3}, {1, 3});

    auto weight_tensor = std::make_shared<Tensor>();
    weight_tensor->Assign<float>({1, 2, 3, 4}, {2, 2});

    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(model), 0);
    ASSERT_EQ(graph.SetTensor("input", input_tensor), 0);
    ASSERT_EQ(graph.SetTensor("weight", weight_tensor), 0);
    EXPECT_EQ(graph.Build(), -1);
}

TEST(runtime_graph_test, LoweringProducesExecutableRuntimeNode) {
    ModelDesc model;
    model.name = "fc_graph_bound";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = {"output"};

    ValueDesc input;
    input.tensor.name = "input";
    input.tensor.dims = {1, 2};
    input.tensor.data_type = DataType::FP32;

    ValueDesc weight;
    weight.tensor.name = "weight";
    weight.tensor.dims = {2, 2};
    weight.tensor.data_type = DataType::FP32;
    weight.constant = true;

    ValueDesc output;
    output.tensor.name = "output";
    output.tensor.dims = {1, 2};
    output.tensor.data_type = DataType::FP32;

    NodeDesc node;
    node.name = "fc0";
    node.op_type = "FC";
    node.inputs = {"input", "weight"};
    node.outputs = {"output"};

    model.graph.values = {input, weight, output};
    model.graph.nodes = {node};

    auto input_tensor = std::make_shared<Tensor>();
    input_tensor->Assign<float>({1, 2}, {1, 2});

    auto weight_tensor = std::make_shared<Tensor>();
    weight_tensor->Assign<float>({1, 2, 3, 4}, {2, 2});

    StaticGraph static_graph;
    ASSERT_EQ(static_graph.SetModel(model), 0);
    ASSERT_EQ(static_graph.SetTensor("input", input_tensor), 0);
    ASSERT_EQ(static_graph.SetTensor("weight", weight_tensor), 0);
    ASSERT_EQ(static_graph.Build(), 0);

    RuntimeGraph runtime_graph;
    GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(static_graph, &runtime_graph), 0);
    ASSERT_EQ(runtime_graph.NodeSize(), 1U);
    ASSERT_EQ(runtime_graph.Run(), 0);

    auto output_tensor = runtime_graph.GetTensor("output");
    ASSERT_NE(output_tensor, nullptr);
    EXPECT_FLOAT_EQ(output_tensor->data<float>()[0], 7.0f);
    EXPECT_FLOAT_EQ(output_tensor->data<float>()[1], 10.0f);
}

TEST(runtime_graph_test, LoweringPreservesDagDependencies) {
    ModelDesc model;
    model.name = "branch_graph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = {"left_out", "right_out"};

    ValueDesc input;
    input.tensor.name = "input";
    input.tensor.dims = {2, 2};
    input.tensor.data_type = DataType::FP32;

    ValueDesc add_bias;
    add_bias.tensor.name = "add_bias";
    add_bias.tensor.dims = {2, 2};
    add_bias.tensor.data_type = DataType::FP32;
    add_bias.constant = true;

    ValueDesc mid;
    mid.tensor.name = "mid";
    mid.tensor.dims = {2, 2};
    mid.tensor.data_type = DataType::FP32;

    ValueDesc left_bias;
    left_bias.tensor.name = "left_bias";
    left_bias.tensor.dims = {2, 2};
    left_bias.tensor.data_type = DataType::FP32;
    left_bias.constant = true;

    ValueDesc right_scale;
    right_scale.tensor.name = "right_scale";
    right_scale.tensor.dims = {2, 2};
    right_scale.tensor.data_type = DataType::FP32;
    right_scale.constant = true;

    ValueDesc left_out;
    left_out.tensor.name = "left_out";
    left_out.tensor.dims = {2, 2};
    left_out.tensor.data_type = DataType::FP32;

    ValueDesc right_out;
    right_out.tensor.name = "right_out";
    right_out.tensor.dims = {2, 2};
    right_out.tensor.data_type = DataType::FP32;

    NodeDesc add0;
    add0.name = "add0";
    add0.op_type = "Add";
    add0.inputs = {"input", "add_bias"};
    add0.outputs = {"mid"};

    NodeDesc add1;
    add1.name = "add1";
    add1.op_type = "Add";
    add1.inputs = {"mid", "left_bias"};
    add1.outputs = {"left_out"};

    NodeDesc mul0;
    mul0.name = "mul0";
    mul0.op_type = "Mul";
    mul0.inputs = {"mid", "right_scale"};
    mul0.outputs = {"right_out"};

    model.graph.values = {input, add_bias, mid, left_bias, right_scale, left_out, right_out};
    model.graph.nodes = {add0, add1, mul0};

    auto input_tensor = std::make_shared<Tensor>();
    input_tensor->Assign<float>({1, 2, 3, 4}, {2, 2});

    auto add_bias_tensor = std::make_shared<Tensor>();
    add_bias_tensor->Assign<float>({1, 1, 1, 1}, {2, 2});

    auto left_bias_tensor = std::make_shared<Tensor>();
    left_bias_tensor->Assign<float>({2, 2, 2, 2}, {2, 2});

    auto right_scale_tensor = std::make_shared<Tensor>();
    right_scale_tensor->Assign<float>({3, 3, 3, 3}, {2, 2});

    StaticGraph static_graph;
    ASSERT_EQ(static_graph.SetModel(model), 0);
    ASSERT_EQ(static_graph.SetTensor("input", input_tensor), 0);
    ASSERT_EQ(static_graph.SetTensor("add_bias", add_bias_tensor), 0);
    ASSERT_EQ(static_graph.SetTensor("left_bias", left_bias_tensor), 0);
    ASSERT_EQ(static_graph.SetTensor("right_scale", right_scale_tensor), 0);
    ASSERT_EQ(static_graph.Build(), 0);

    RuntimeGraph runtime_graph;
    GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(static_graph, &runtime_graph), 0);
    ASSERT_EQ(runtime_graph.NodeSize(), 3U);

    const auto* add_node = runtime_graph.GetNode("add0");
    const auto* left_node = runtime_graph.GetNode("add1");
    const auto* right_node = runtime_graph.GetNode("mul0");
    ASSERT_NE(add_node, nullptr);
    ASSERT_NE(left_node, nullptr);
    ASSERT_NE(right_node, nullptr);

    EXPECT_EQ(add_node->pending_dependencies, 0U);
    EXPECT_EQ(add_node->successors.size(), 2U);
    EXPECT_EQ(left_node->pending_dependencies, 1U);
    EXPECT_EQ(right_node->pending_dependencies, 1U);
}

TEST(runtime_graph_test, LoweringInitializesReusableExecutionWorkers) {
    ModelDesc model;
    model.name = "branch_graph_workers";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = {"left_out", "right_out"};

    ValueDesc input;
    input.tensor.name = "input";
    input.tensor.dims = {2, 2};
    input.tensor.data_type = DataType::FP32;

    ValueDesc add_bias;
    add_bias.tensor.name = "add_bias";
    add_bias.tensor.dims = {2, 2};
    add_bias.tensor.data_type = DataType::FP32;
    add_bias.constant = true;

    ValueDesc mid;
    mid.tensor.name = "mid";
    mid.tensor.dims = {2, 2};
    mid.tensor.data_type = DataType::FP32;

    ValueDesc left_bias;
    left_bias.tensor.name = "left_bias";
    left_bias.tensor.dims = {2, 2};
    left_bias.tensor.data_type = DataType::FP32;
    left_bias.constant = true;

    ValueDesc right_scale;
    right_scale.tensor.name = "right_scale";
    right_scale.tensor.dims = {2, 2};
    right_scale.tensor.data_type = DataType::FP32;
    right_scale.constant = true;

    ValueDesc left_out;
    left_out.tensor.name = "left_out";
    left_out.tensor.dims = {2, 2};
    left_out.tensor.data_type = DataType::FP32;

    ValueDesc right_out;
    right_out.tensor.name = "right_out";
    right_out.tensor.dims = {2, 2};
    right_out.tensor.data_type = DataType::FP32;

    NodeDesc add0;
    add0.name = "add0";
    add0.op_type = "Add";
    add0.inputs = {"input", "add_bias"};
    add0.outputs = {"mid"};

    NodeDesc add1;
    add1.name = "add1";
    add1.op_type = "Add";
    add1.inputs = {"mid", "left_bias"};
    add1.outputs = {"left_out"};

    NodeDesc mul0;
    mul0.name = "mul0";
    mul0.op_type = "Mul";
    mul0.inputs = {"mid", "right_scale"};
    mul0.outputs = {"right_out"};

    model.graph.values = {input, add_bias, mid, left_bias, right_scale, left_out, right_out};
    model.graph.nodes = {add0, add1, mul0};

    auto input_tensor = std::make_shared<Tensor>();
    input_tensor->Assign<float>({1, 2, 3, 4}, {2, 2});

    auto add_bias_tensor = std::make_shared<Tensor>();
    add_bias_tensor->Assign<float>({1, 1, 1, 1}, {2, 2});

    auto left_bias_tensor = std::make_shared<Tensor>();
    left_bias_tensor->Assign<float>({2, 2, 2, 2}, {2, 2});

    auto right_scale_tensor = std::make_shared<Tensor>();
    right_scale_tensor->Assign<float>({3, 3, 3, 3}, {2, 2});

    StaticGraph static_graph;
    ASSERT_EQ(static_graph.SetModel(model), 0);
    ASSERT_EQ(static_graph.SetTensor("input", input_tensor), 0);
    ASSERT_EQ(static_graph.SetTensor("add_bias", add_bias_tensor), 0);
    ASSERT_EQ(static_graph.SetTensor("left_bias", left_bias_tensor), 0);
    ASSERT_EQ(static_graph.SetTensor("right_scale", right_scale_tensor), 0);
    ASSERT_EQ(static_graph.Build(), 0);

    RuntimeGraph runtime_graph;
    GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(static_graph, &runtime_graph), 0);

    const size_t expected_workers = std::min(feather::DefaultThreadCount(), runtime_graph.NodeSize());
    EXPECT_EQ(runtime_graph.WorkerCount(), expected_workers);

    ASSERT_EQ(runtime_graph.Run(), 0);
    ASSERT_EQ(runtime_graph.Run(), 0);
    EXPECT_EQ(runtime_graph.WorkerCount(), expected_workers);
}

TEST(runtime_graph_test, RuntimeNodeProfileLabelUsesNodeNameAndOpType) {
    feather::RuntimeNode node;
    node.name = "conv_17";
    node.op_type = "Conv2D";
    EXPECT_EQ(node.ProfileLabel(), "RuntimeNode::conv_17[Conv2D]");
}
