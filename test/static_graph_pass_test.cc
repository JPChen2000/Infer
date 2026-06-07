#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/graph_pass.h"
#include "core/dead_node_elimination_pass.h"
#include "core/static_graph.h"
#include "core/tensor.h"
#include "model/model_format.h"

using feather::DataType;
using feather::DeadNodeEliminationPass;
using feather::GraphPass;
using feather::PassManager;
using feather::StaticGraph;
using feather::Tensor;
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
