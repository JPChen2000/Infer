#include <gtest/gtest.h>

#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "pass/graph_pass.h"
#include "pass/dead_node_elimination_pass.h"
#include "pass/identity_elimination_pass.h"
#include "pass/matmul_add_fusion_pass.h"
#include "pass/no_op_elimination_pass.h"
#include "pass/reshape_chain_elimination_pass.h"
#include "pass/sigmoid_mul_fusion_pass.h"
#include "core/static_graph.h"
#include "core/tensor.h"
#include "pass/yolo_decode_fusion_pass.h"
#include "model/model_format.h"

using feather::DataType;
using feather::DeadNodeEliminationPass;
using feather::GraphPass;
using feather::IdentityEliminationPass;
using feather::MatMulAddFusionPass;
using feather::NoOpEliminationPass;
using feather::PassManager;
using feather::ReshapeChainEliminationPass;
using feather::SigmoidMulFusionPass;
using feather::StaticGraph;
using feather::Tensor;
using feather::YoloDecodeFusionPass;
using feather::model::ModelDesc;
using feather::model::NodeDesc;
using feather::model::ValueDesc;

namespace {

ModelDesc BuildConvReluModelDesc() {
    ModelDesc model;
    model.name = "conv_relu_graph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = {"relu_out"};

    ValueDesc input;
    input.tensor.name = "input";
    input.tensor.dims = {3, 3};
    input.tensor.data_type = DataType::FP32;

    ValueDesc weight;
    weight.tensor.name = "weight";
    weight.tensor.dims = {2, 2};
    weight.tensor.data_type = DataType::FP32;
    weight.constant = true;

    ValueDesc bias;
    bias.tensor.name = "bias";
    bias.tensor.dims = {2, 2};
    bias.tensor.data_type = DataType::FP32;
    bias.constant = true;

    ValueDesc conv_out;
    conv_out.tensor.name = "conv_out";
    conv_out.tensor.dims = {2, 2};
    conv_out.tensor.data_type = DataType::FP32;

    ValueDesc relu_out;
    relu_out.tensor.name = "relu_out";
    relu_out.tensor.dims = {2, 2};
    relu_out.tensor.data_type = DataType::FP32;

    NodeDesc conv;
    conv.name = "conv0";
    conv.op_type = "Conv2D";
    conv.inputs = {"input", "weight", "bias"};
    conv.outputs = {"conv_out"};
    conv.attributes["stride_h"] = static_cast<int64_t>(1);
    conv.attributes["stride_w"] = static_cast<int64_t>(1);
    conv.attributes["pad_h"] = static_cast<int64_t>(0);
    conv.attributes["pad_w"] = static_cast<int64_t>(0);

    NodeDesc relu;
    relu.name = "relu0";
    relu.op_type = "ReLU";
    relu.inputs = {"conv_out"};
    relu.outputs = {"relu_out"};

    model.graph.values = {input, weight, bias, conv_out, relu_out};
    model.graph.nodes = {conv, relu};
    return model;
}

ModelDesc BuildConvReluDeadTailModelDesc() {
    ModelDesc model = BuildConvReluModelDesc();
    model.name = "conv_relu_dead_tail_graph";
    model.graph.outputs = {"conv_out"};
    return model;
}

ModelDesc BuildSigmoidMulModelDesc(bool reverse_mul_inputs) {
    ModelDesc model;
    model.name = "sigmoid_mul_graph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = {"output"};

    ValueDesc input;
    input.tensor.name = "input";
    input.tensor.dims = {4};
    input.tensor.data_type = DataType::FP32;

    ValueDesc sigmoid_out;
    sigmoid_out.tensor.name = "sigmoid_out";
    sigmoid_out.tensor.dims = {4};
    sigmoid_out.tensor.data_type = DataType::FP32;

    ValueDesc output;
    output.tensor.name = "output";
    output.tensor.dims = {4};
    output.tensor.data_type = DataType::FP32;

    NodeDesc sigmoid;
    sigmoid.name = "sigmoid0";
    sigmoid.op_type = "Sigmoid";
    sigmoid.inputs = {"input"};
    sigmoid.outputs = {"sigmoid_out"};

    NodeDesc mul;
    mul.name = "mul0";
    mul.op_type = "Mul";
    mul.inputs = reverse_mul_inputs ? std::vector<std::string>{"sigmoid_out", "input"}
                                    : std::vector<std::string>{"input", "sigmoid_out"};
    mul.outputs = {"output"};

    model.graph.values = {input, sigmoid_out, output};
    model.graph.nodes = {sigmoid, mul};
    return model;
}

ModelDesc BuildIdentityRelayModelDesc(bool identity_output_is_graph_output = false) {
    ModelDesc model;
    model.name = "identity_relay_graph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = {identity_output_is_graph_output ? "identity_out" : "output"};

    ValueDesc input;
    input.tensor.name = "input";
    input.tensor.dims = {2, 2};
    input.tensor.data_type = DataType::FP32;

    ValueDesc identity_out;
    identity_out.tensor.name = "identity_out";
    identity_out.tensor.dims = {2, 2};
    identity_out.tensor.data_type = DataType::FP32;

    ValueDesc output;
    output.tensor.name = "output";
    output.tensor.dims = {2, 2};
    output.tensor.data_type = DataType::FP32;

    NodeDesc identity;
    identity.name = "identity0";
    identity.op_type = "Identity";
    identity.inputs = {"input"};
    identity.outputs = {"identity_out"};

    NodeDesc relu;
    relu.name = "relu0";
    relu.op_type = "ReLU";
    relu.inputs = {"identity_out"};
    relu.outputs = {"output"};

    model.graph.values = {input, identity_out, output};
    model.graph.nodes = identity_output_is_graph_output ? std::vector<NodeDesc>{identity} : std::vector<NodeDesc>{identity, relu};
    return model;
}

ModelDesc BuildMatMulAddModelDesc(bool keep_matmul_output_live = false) {
    ModelDesc model;
    model.name = "matmul_add_graph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"a", "b", "bias"};
    model.graph.outputs = keep_matmul_output_live ? std::vector<std::string>{"output", "matmul_out"}
                                                  : std::vector<std::string>{"output"};

    auto value = [](const std::string& name, std::vector<int64_t> dims) {
        ValueDesc desc;
        desc.tensor.name = name;
        desc.tensor.dims = std::move(dims);
        desc.tensor.data_type = DataType::FP32;
        return desc;
    };

    model.graph.values = {value("a", {2, 3}), value("b", {3, 4}), value("bias", {4}), value("matmul_out", {2, 4}),
                          value("output", {2, 4})};

    NodeDesc matmul;
    matmul.name = "matmul0";
    matmul.op_type = "MatMul";
    matmul.inputs = {"a", "b"};
    matmul.outputs = {"matmul_out"};

    NodeDesc add;
    add.name = "add0";
    add.op_type = "Add";
    add.inputs = {"matmul_out", "bias"};
    add.outputs = {"output"};

    model.graph.nodes = {matmul, add};
    return model;
}

ModelDesc BuildNoOpReshapeModelDesc() {
    ModelDesc model;
    model.name = "no_op_reshape_graph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = {"output"};

    auto value = [](const std::string& name, std::vector<int64_t> dims) {
        ValueDesc desc;
        desc.tensor.name = name;
        desc.tensor.dims = std::move(dims);
        desc.tensor.data_type = DataType::FP32;
        return desc;
    };

    model.graph.values = {value("input", {2, 2}), value("reshape_out", {2, 2}), value("output", {2, 2})};

    NodeDesc reshape;
    reshape.name = "reshape0";
    reshape.op_type = "Reshape";
    reshape.inputs = {"input"};
    reshape.outputs = {"reshape_out"};
    reshape.attributes["shape"] = std::vector<int64_t>{2, 2};

    NodeDesc relu;
    relu.name = "relu0";
    relu.op_type = "ReLU";
    relu.inputs = {"reshape_out"};
    relu.outputs = {"output"};

    model.graph.nodes = {reshape, relu};
    return model;
}

ModelDesc BuildIdentityTransposeModelDesc() {
    ModelDesc model;
    model.name = "identity_transpose_graph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = {"output"};

    auto value = [](const std::string& name, std::vector<int64_t> dims) {
        ValueDesc desc;
        desc.tensor.name = name;
        desc.tensor.dims = std::move(dims);
        desc.tensor.data_type = DataType::FP32;
        return desc;
    };

    model.graph.values = {value("input", {1, 2, 3}), value("transpose_out", {1, 2, 3}), value("output", {1, 2, 3})};

    NodeDesc transpose;
    transpose.name = "transpose0";
    transpose.op_type = "Transpose";
    transpose.inputs = {"input"};
    transpose.outputs = {"transpose_out"};
    transpose.attributes["perm"] = std::vector<int64_t>{0, 1, 2};

    NodeDesc relu;
    relu.name = "relu0";
    relu.op_type = "ReLU";
    relu.inputs = {"transpose_out"};
    relu.outputs = {"output"};

    model.graph.nodes = {transpose, relu};
    return model;
}

ModelDesc BuildReshapeChainModelDesc(bool keep_inner_output_live = false) {
    ModelDesc model;
    model.name = "reshape_chain_graph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = keep_inner_output_live ? std::vector<std::string>{"output", "reshape0_out"}
                                                 : std::vector<std::string>{"output"};

    auto value = [](const std::string& name, std::vector<int64_t> dims) {
        ValueDesc desc;
        desc.tensor.name = name;
        desc.tensor.dims = std::move(dims);
        desc.tensor.data_type = DataType::FP32;
        return desc;
    };

    model.graph.values = {value("input", {2, 3}), value("reshape0_out", {1, 6}), value("output", {3, 2})};

    NodeDesc reshape0;
    reshape0.name = "reshape0";
    reshape0.op_type = "Reshape";
    reshape0.inputs = {"input"};
    reshape0.outputs = {"reshape0_out"};
    reshape0.attributes["shape"] = std::vector<int64_t>{1, 6};

    NodeDesc reshape1;
    reshape1.name = "reshape1";
    reshape1.op_type = "Reshape";
    reshape1.inputs = {"reshape0_out"};
    reshape1.outputs = {"output"};
    reshape1.attributes["shape"] = std::vector<int64_t>{3, 2};

    model.graph.nodes = {reshape0, reshape1};
    return model;
}

ModelDesc BuildYoloDecodePatternModelDesc() {
    ModelDesc model;
    model.name = "yolo_decode_pattern";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"raw"};
    model.graph.outputs = {"output"};

    auto value = [](const std::string& name, std::vector<int64_t> dims, bool constant = false) {
        ValueDesc desc;
        desc.tensor.name = name;
        desc.tensor.dims = std::move(dims);
        desc.tensor.data_type = DataType::FP32;
        desc.constant = constant;
        return desc;
    };

    model.graph.values = {
        value("raw", {1, 12, 1, 1}),
        value("xy_scale", {1}, true),
        value("grid", {1, 2, 1, 1, 2}, true),
        value("stride", {1}, true),
        value("wh_scale", {1}, true),
        value("anchor", {1, 2, 1, 1, 2}, true),
        value("reshaped", {1, 2, 6, 1, 1}),
        value("transposed", {1, 2, 1, 1, 6}),
        value("sigmoid", {1, 2, 1, 1, 6}),
        value("split_xy", {1, 2, 1, 1, 2}),
        value("split_wh", {1, 2, 1, 1, 2}),
        value("split_rest", {1, 2, 1, 1, 2}),
        value("mul_xy", {1, 2, 1, 1, 2}),
        value("add_xy", {1, 2, 1, 1, 2}),
        value("decoded_xy", {1, 2, 1, 1, 2}),
        value("mul_wh", {1, 2, 1, 1, 2}),
        value("pow_wh", {1, 2, 1, 1, 2}),
        value("decoded_wh", {1, 2, 1, 1, 2}),
        value("concat", {1, 2, 1, 1, 6}),
        value("output", {1, 2, 6}),
    };

    NodeDesc reshape0;
    reshape0.name = "reshape0";
    reshape0.op_type = "Reshape";
    reshape0.inputs = {"raw"};
    reshape0.outputs = {"reshaped"};
    reshape0.attributes["shape"] = std::vector<int64_t>{1, 2, 6, 1, 1};

    NodeDesc transpose;
    transpose.name = "transpose0";
    transpose.op_type = "Transpose";
    transpose.inputs = {"reshaped"};
    transpose.outputs = {"transposed"};
    transpose.attributes["perm"] = std::vector<int64_t>{0, 1, 3, 4, 2};

    NodeDesc sigmoid;
    sigmoid.name = "sigmoid0";
    sigmoid.op_type = "Sigmoid";
    sigmoid.inputs = {"transposed"};
    sigmoid.outputs = {"sigmoid"};

    NodeDesc split;
    split.name = "split0";
    split.op_type = "Split";
    split.inputs = {"sigmoid"};
    split.outputs = {"split_xy", "split_wh", "split_rest"};
    split.attributes["axis"] = static_cast<int64_t>(4);
    split.attributes["split_sizes"] = std::vector<int64_t>{2, 2, 2};

    NodeDesc mul_xy;
    mul_xy.name = "mul_xy";
    mul_xy.op_type = "Mul";
    mul_xy.inputs = {"split_xy", "xy_scale"};
    mul_xy.outputs = {"mul_xy"};

    NodeDesc add_xy;
    add_xy.name = "add_xy";
    add_xy.op_type = "Add";
    add_xy.inputs = {"mul_xy", "grid"};
    add_xy.outputs = {"add_xy"};

    NodeDesc mul_xy_stride;
    mul_xy_stride.name = "mul_xy_stride";
    mul_xy_stride.op_type = "Mul";
    mul_xy_stride.inputs = {"add_xy", "stride"};
    mul_xy_stride.outputs = {"decoded_xy"};

    NodeDesc mul_wh;
    mul_wh.name = "mul_wh";
    mul_wh.op_type = "Mul";
    mul_wh.inputs = {"split_wh", "wh_scale"};
    mul_wh.outputs = {"mul_wh"};

    NodeDesc pow_wh;
    pow_wh.name = "pow_wh";
    pow_wh.op_type = "Pow";
    pow_wh.inputs = {"mul_wh"};
    pow_wh.outputs = {"pow_wh"};
    pow_wh.attributes["exponent"] = 2.0f;

    NodeDesc mul_wh_anchor;
    mul_wh_anchor.name = "mul_wh_anchor";
    mul_wh_anchor.op_type = "Mul";
    mul_wh_anchor.inputs = {"pow_wh", "anchor"};
    mul_wh_anchor.outputs = {"decoded_wh"};

    NodeDesc concat;
    concat.name = "concat0";
    concat.op_type = "Concat";
    concat.inputs = {"decoded_xy", "decoded_wh", "split_rest"};
    concat.outputs = {"concat"};
    concat.attributes["axis"] = static_cast<int64_t>(4);

    NodeDesc reshape1;
    reshape1.name = "reshape1";
    reshape1.op_type = "Reshape";
    reshape1.inputs = {"concat"};
    reshape1.outputs = {"output"};
    reshape1.attributes["shape"] = std::vector<int64_t>{1, 2, 6};

    model.graph.nodes = {reshape0, transpose, sigmoid, split, mul_xy, add_xy, mul_xy_stride,
                         mul_wh,   pow_wh,    mul_wh_anchor, concat, reshape1};
    return model;
}

void BindSiluGraphInputs(StaticGraph* graph) {
    auto input_tensor = std::make_shared<Tensor>();
    input_tensor->Assign<float>({-2.0f, -1.0f, 0.0f, 2.0f}, {4});
    ASSERT_EQ(graph->SetTensor("input", input_tensor), 0);
}

void BindIdentityRelayInputs(StaticGraph* graph) {
    auto input_tensor = std::make_shared<Tensor>();
    input_tensor->Assign<float>({-1.0f, 2.0f, -3.0f, 4.0f}, {2, 2});
    ASSERT_EQ(graph->SetTensor("input", input_tensor), 0);
}

void BindMatMulAddInputs(StaticGraph* graph) {
    auto a = std::make_shared<Tensor>();
    a->Assign<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    auto b = std::make_shared<Tensor>();
    b->Assign<float>({1.0f, 0.0f, 2.0f, 1.0f, 0.0f, 1.0f, 3.0f, 2.0f, 1.0f, 0.0f, 1.0f, 4.0f}, {3, 4});
    auto bias = std::make_shared<Tensor>();
    bias->Assign<float>({0.5f, -1.0f, 1.5f, 0.0f}, {4});
    ASSERT_EQ(graph->SetTensor("a", a), 0);
    ASSERT_EQ(graph->SetTensor("b", b), 0);
    ASSERT_EQ(graph->SetTensor("bias", bias), 0);
}

void BindNoOpGraphInputs(StaticGraph* graph, const std::vector<int64_t>& dims) {
    auto input_tensor = std::make_shared<Tensor>();
    const int64_t numel = std::accumulate(dims.begin(), dims.end(), int64_t{1}, std::multiplies<int64_t>());
    std::vector<float> values(static_cast<size_t>(numel), 1.0f);
    input_tensor->Assign<float>(values, dims);
    ASSERT_EQ(graph->SetTensor("input", input_tensor), 0);
}

void BindConvReluGraphInputs(StaticGraph* graph) {
    auto input_tensor = std::make_shared<Tensor>();
    input_tensor->Assign<float>({
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    }, {3, 3});

    auto weight_tensor = std::make_shared<Tensor>();
    weight_tensor->Assign<float>({
        1, 0,
        0, -1,
    }, {2, 2});

    auto bias_tensor = std::make_shared<Tensor>();
    bias_tensor->Assign<float>({
        1, 1,
        1, 1,
    }, {2, 2});

    ASSERT_EQ(graph->SetTensor("input", input_tensor), 0);
    ASSERT_EQ(graph->SetTensor("weight", weight_tensor), 0);
    ASSERT_EQ(graph->SetTensor("bias", bias_tensor), 0);
}

void BindYoloDecodePatternInputs(StaticGraph* graph) {
    auto raw = std::make_shared<Tensor>();
    raw->Assign<float>(
        {
            0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f,
            2.0f, -2.0f, 0.5f, -0.5f, 0.0f, 4.0f,
        },
        {1, 12, 1, 1});
    auto xy_scale = std::make_shared<Tensor>();
    xy_scale->Assign<float>({2.0f}, {1});
    auto grid = std::make_shared<Tensor>();
    grid->Assign<float>({10.0f, 20.0f, 30.0f, 40.0f}, {1, 2, 1, 1, 2});
    auto stride = std::make_shared<Tensor>();
    stride->Assign<float>({8.0f}, {1});
    auto wh_scale = std::make_shared<Tensor>();
    wh_scale->Assign<float>({2.0f}, {1});
    auto anchor = std::make_shared<Tensor>();
    anchor->Assign<float>({4.0f, 6.0f, 8.0f, 10.0f}, {1, 2, 1, 1, 2});

    ASSERT_EQ(graph->SetTensor("raw", raw), 0);
    ASSERT_EQ(graph->SetTensor("xy_scale", xy_scale), 0);
    ASSERT_EQ(graph->SetTensor("grid", grid), 0);
    ASSERT_EQ(graph->SetTensor("stride", stride), 0);
    ASSERT_EQ(graph->SetTensor("wh_scale", wh_scale), 0);
    ASSERT_EQ(graph->SetTensor("anchor", anchor), 0);
}

class RecordingPass : public GraphPass {
   public:
    explicit RecordingPass(std::vector<std::string>* trace, std::string label)
        : trace_(trace), label_(std::move(label)) {}

    const std::string& name() const override { return label_; }

    int32_t Run(StaticGraph* graph) override {
        if (graph == nullptr || trace_ == nullptr) {
            return -1;
        }
        trace_->push_back(label_);
        return 0;
    }

   private:
    std::vector<std::string>* trace_;
    std::string label_;
};

}  // namespace

TEST(static_graph_pass_test, BuildCapturesNodeMetadataAndUseDef) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildConvReluModelDesc()), 0);
    BindConvReluGraphInputs(&graph);

    ASSERT_EQ(graph.Build(), 0);
    ASSERT_EQ(graph.NodeSize(), 2U);

    const auto* conv_node = graph.GetNode("conv0");
    ASSERT_NE(conv_node, nullptr);
    EXPECT_EQ(conv_node->op_type, "Conv2D");
    EXPECT_EQ(conv_node->outputs, std::vector<std::string>({"conv_out"}));

    const auto* relu_node = graph.GetNode("relu0");
    ASSERT_NE(relu_node, nullptr);
    EXPECT_EQ(relu_node->inputs, std::vector<std::string>({"conv_out"}));

    EXPECT_EQ(graph.GetProducer("conv_out"), "conv0");
    EXPECT_EQ(graph.GetUsers("conv_out"), std::vector<std::string>({"relu0"}));
}

TEST(static_graph_pass_test, BuildCapturesProducerAndUsersForGraphValues) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildConvReluModelDesc()), 0);
    BindConvReluGraphInputs(&graph);

    ASSERT_EQ(graph.Build(), 0);
    EXPECT_EQ(graph.GetProducer("input"), "");
    EXPECT_EQ(graph.GetUsers("input"), std::vector<std::string>({"conv0"}));
    EXPECT_EQ(graph.GetProducer("relu_out"), "relu0");
    EXPECT_TRUE(graph.GetUsers("relu_out").empty());
}

TEST(static_graph_pass_test, ApplyPassesRunsRegisteredPassesInOrder) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildConvReluModelDesc()), 0);
    BindConvReluGraphInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    std::vector<std::string> trace;
    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<RecordingPass>(&trace, "first"));
    pass_manager->AddPass(std::make_unique<RecordingPass>(&trace, "second"));

    graph.SetPassManager(pass_manager);
    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_EQ(trace, (std::vector<std::string>{"first", "second"}));
}

TEST(static_graph_pass_test, RemoveNodeUpdatesUseDefAndOperatorView) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildConvReluDeadTailModelDesc()), 0);
    BindConvReluGraphInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    ASSERT_TRUE(graph.RemoveNode("relu0"));
    EXPECT_EQ(graph.NodeSize(), 1U);
    EXPECT_EQ(graph.OperatorSize(), 1U);
    EXPECT_TRUE(graph.GetUsers("conv_out").empty());
    EXPECT_EQ(graph.GetProducer("relu_out"), "");
    EXPECT_EQ(graph.GetNode("relu0"), nullptr);
}

TEST(static_graph_pass_test, ApplyPassesUsesInstalledPassManager) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildConvReluModelDesc()), 0);
    BindConvReluGraphInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    std::vector<std::string> trace;
    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<RecordingPass>(&trace, "installed"));
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_EQ(trace, (std::vector<std::string>{"installed"}));
}

TEST(static_graph_pass_test, SetModelPreservesPreinstalledPassManager) {
    StaticGraph graph;
    std::vector<std::string> trace;
    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<RecordingPass>(&trace, "preinstalled"));
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.SetModel(BuildConvReluModelDesc()), 0);
    BindConvReluGraphInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);
    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_EQ(trace, (std::vector<std::string>{"preinstalled"}));
}

TEST(static_graph_pass_test, ReplaceInputValueRebuildsAffectedNode) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildConvReluModelDesc()), 0);
    BindConvReluGraphInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    ASSERT_TRUE(graph.ReplaceInputValue("relu0", "conv_out", "input"));
    EXPECT_TRUE(graph.GetUsers("conv_out").empty());
    EXPECT_EQ(graph.GetUsers("input"), std::vector<std::string>({"conv0", "relu0"}));

    auto output_tensor = graph.GetTensor("relu_out");
    ASSERT_NE(output_tensor, nullptr);
    EXPECT_EQ(output_tensor->dims().data(), std::vector<int64_t>({3, 3}));
}

TEST(static_graph_pass_test, DeadNodeEliminationPassRemovesUnusedTailNodes) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildConvReluDeadTailModelDesc()), 0);
    BindConvReluGraphInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<DeadNodeEliminationPass>());
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_EQ(graph.NodeSize(), 1U);
    EXPECT_EQ(graph.OperatorSize(), 1U);
    EXPECT_EQ(graph.GetNode("relu0"), nullptr);
}

TEST(static_graph_pass_test, SigmoidMulFusionPassRewritesSiluPattern) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildSigmoidMulModelDesc(false)), 0);
    BindSiluGraphInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<SigmoidMulFusionPass>());
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_EQ(graph.NodeSize(), 1U);
    EXPECT_EQ(graph.OperatorSize(), 1U);
    EXPECT_EQ(graph.GetNode("sigmoid0"), nullptr);

    const auto* fused = graph.GetNode("mul0");
    ASSERT_NE(fused, nullptr);
    EXPECT_EQ(fused->op_type, "SiLU");
    EXPECT_EQ(fused->inputs, std::vector<std::string>({"input"}));
    EXPECT_EQ(fused->outputs, std::vector<std::string>({"output"}));
    EXPECT_TRUE(graph.GetUsers("sigmoid_out").empty());
    EXPECT_EQ(graph.GetUsers("input"), std::vector<std::string>({"mul0"}));
    EXPECT_EQ(graph.GetProducer("output"), "mul0");
}

TEST(static_graph_pass_test, SigmoidMulFusionPassAcceptsReversedMulInputs) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildSigmoidMulModelDesc(true)), 0);
    BindSiluGraphInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<SigmoidMulFusionPass>());
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    const auto* fused = graph.GetNode("mul0");
    ASSERT_NE(fused, nullptr);
    EXPECT_EQ(fused->op_type, "SiLU");
    EXPECT_EQ(fused->inputs, std::vector<std::string>({"input"}));
}

TEST(static_graph_pass_test, IdentityEliminationPassRemovesInteriorIdentity) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildIdentityRelayModelDesc(false)), 0);
    BindIdentityRelayInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<IdentityEliminationPass>());
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_EQ(graph.NodeSize(), 1U);
    EXPECT_EQ(graph.GetNode("identity0"), nullptr);
    const auto* relu = graph.GetNode("relu0");
    ASSERT_NE(relu, nullptr);
    EXPECT_EQ(relu->inputs, std::vector<std::string>({"input"}));
}

TEST(static_graph_pass_test, IdentityEliminationPassPreservesGraphOutputIdentity) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildIdentityRelayModelDesc(true)), 0);
    BindIdentityRelayInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<IdentityEliminationPass>());
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_EQ(graph.NodeSize(), 1U);
    EXPECT_NE(graph.GetNode("identity0"), nullptr);
}

TEST(static_graph_pass_test, MatMulAddFusionPassRewritesToGemm) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildMatMulAddModelDesc(false)), 0);
    BindMatMulAddInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<MatMulAddFusionPass>());
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_EQ(graph.NodeSize(), 1U);
    EXPECT_EQ(graph.GetNode("matmul0"), nullptr);
    const auto* fused = graph.GetNode("add0");
    ASSERT_NE(fused, nullptr);
    EXPECT_EQ(fused->op_type, "Gemm");
    EXPECT_EQ(fused->inputs, (std::vector<std::string>{"a", "b", "bias"}));
}

TEST(static_graph_pass_test, MatMulAddFusionPassSkipsWhenMatMulOutputIsGraphOutput) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildMatMulAddModelDesc(true)), 0);
    BindMatMulAddInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<MatMulAddFusionPass>());
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_NE(graph.GetNode("matmul0"), nullptr);
    const auto* add = graph.GetNode("add0");
    ASSERT_NE(add, nullptr);
    EXPECT_EQ(add->op_type, "Add");
}

TEST(static_graph_pass_test, DefaultPassManagerIncludesGeneralFusionPasses) {
    auto pass_manager = feather::CreateDefaultPassManager();
    ASSERT_NE(pass_manager, nullptr);
    EXPECT_EQ(pass_manager->PassCount(), 6U);
}

TEST(static_graph_pass_test, YoloPassManagerAddsDecodeFusionOnTopOfDefaultPasses) {
    auto pass_manager = feather::CreateYoloPassManager();
    ASSERT_NE(pass_manager, nullptr);
    EXPECT_EQ(pass_manager->PassCount(), 8U);
}

TEST(static_graph_pass_test, ReshapeChainEliminationPassRemovesInnerReshape) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildReshapeChainModelDesc(false)), 0);
    BindNoOpGraphInputs(&graph, {2, 3});
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<ReshapeChainEliminationPass>());
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_EQ(graph.GetNode("reshape0"), nullptr);
    const auto* reshape1 = graph.GetNode("reshape1");
    ASSERT_NE(reshape1, nullptr);
    EXPECT_EQ(reshape1->inputs, std::vector<std::string>({"input"}));
}

TEST(static_graph_pass_test, ReshapeChainEliminationPassPreservesGraphOutputIntermediate) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildReshapeChainModelDesc(true)), 0);
    BindNoOpGraphInputs(&graph, {2, 3});
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<ReshapeChainEliminationPass>());
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_NE(graph.GetNode("reshape0"), nullptr);
    EXPECT_NE(graph.GetNode("reshape1"), nullptr);
}

TEST(static_graph_pass_test, NoOpEliminationPassRemovesSameShapeReshape) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildNoOpReshapeModelDesc()), 0);
    BindNoOpGraphInputs(&graph, {2, 2});
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<NoOpEliminationPass>());
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_EQ(graph.GetNode("reshape0"), nullptr);
    const auto* relu = graph.GetNode("relu0");
    ASSERT_NE(relu, nullptr);
    EXPECT_EQ(relu->inputs, std::vector<std::string>({"input"}));
}

TEST(static_graph_pass_test, NoOpEliminationPassRemovesIdentityTranspose) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildIdentityTransposeModelDesc()), 0);
    BindNoOpGraphInputs(&graph, {1, 2, 3});
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<NoOpEliminationPass>());
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_EQ(graph.GetNode("transpose0"), nullptr);
    const auto* relu = graph.GetNode("relu0");
    ASSERT_NE(relu, nullptr);
    EXPECT_EQ(relu->inputs, std::vector<std::string>({"input"}));
}

TEST(static_graph_pass_test, ApplyPassesUsesDefaultPassManagerWhenNotOverridden) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildIdentityRelayModelDesc(false)), 0);
    BindIdentityRelayInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_EQ(graph.GetNode("identity0"), nullptr);
    const auto* relu = graph.GetNode("relu0");
    ASSERT_NE(relu, nullptr);
    EXPECT_EQ(relu->inputs, std::vector<std::string>({"input"}));
}

TEST(static_graph_pass_test, YoloDecodeFusionPassRewritesDecodeScalePattern) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildYoloDecodePatternModelDesc()), 0);
    BindYoloDecodePatternInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<YoloDecodeFusionPass>());
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_EQ(graph.NodeSize(), 1U);
    EXPECT_EQ(graph.OperatorSize(), 1U);

    const auto* fused = graph.GetNode("reshape1");
    ASSERT_NE(fused, nullptr);
    EXPECT_EQ(fused->op_type, "YoloDecode");
    EXPECT_EQ(fused->inputs,
              (std::vector<std::string>{"raw", "xy_scale", "grid", "stride", "wh_scale", "anchor"}));
    EXPECT_EQ(fused->outputs, std::vector<std::string>({"output"}));
    EXPECT_EQ(graph.GetProducer("output"), "reshape1");
    EXPECT_EQ(graph.GetUsers("raw"), std::vector<std::string>({"reshape1"}));
    EXPECT_EQ(graph.GetNode("concat0"), nullptr);
    EXPECT_EQ(graph.GetNode("transpose0"), nullptr);
}
