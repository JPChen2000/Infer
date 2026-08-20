#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "core/static_graph.h"
#include "model/model_format.h"
#include "pass/qwen_rms_norm_fusion_pass.h"
#include "util/bf16.h"

namespace {

using feather::DataType;
using feather::DeviceType;
using feather::PassManager;
using feather::StaticGraph;
using feather::Tensor;
using feather::model::ModelDesc;
using feather::model::NodeDesc;
using feather::model::ValueDesc;

ValueDesc Value(const std::string& name, std::vector<int64_t> dims, DataType dtype, bool constant = false) {
    ValueDesc value;
    value.tensor.name = name;
    value.tensor.dims = std::move(dims);
    value.tensor.data_type = dtype;
    value.constant = constant;
    return value;
}

NodeDesc Node(const std::string& name, const std::string& op_type, std::vector<std::string> inputs,
             std::vector<std::string> outputs) {
    NodeDesc node;
    node.name = name;
    node.op_type = op_type;
    node.inputs = std::move(inputs);
    node.outputs = std::move(outputs);
    return node;
}

ModelDesc BuildRmsNormModel(bool shared_square_output, bool direct_fp32_input) {
    ModelDesc model;
    model.name = "qwen_rms_norm_graph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = shared_square_output ? std::vector<std::string>{"output", "square"}
                                               : std::vector<std::string>{"output"};

    const auto input_dtype = direct_fp32_input ? DataType::FP32 : DataType::BF16;
    model.graph.values = {
        Value("input", {1, 1, 4}, input_dtype),
        Value("weight", {4}, DataType::BF16, true),
        Value("epsilon", {1}, DataType::FP32, true),
        Value("one", {1}, DataType::FP32, true),
        Value("input_fp32", {1, 1, 4}, DataType::FP32),
        Value("square", {1, 1, 4}, DataType::FP32),
        Value("mean", {1, 1, 1}, DataType::FP32),
        Value("mean_eps", {1, 1, 1}, DataType::FP32),
        Value("root", {1, 1, 1}, DataType::FP32),
        Value("normalized", {1, 1, 4}, DataType::FP32),
        Value("weight_fp32", {4}, DataType::FP32),
        Value("scale", {4}, DataType::FP32),
        Value("scaled", {1, 1, 4}, DataType::FP32),
        Value("output", {1, 1, 4}, DataType::BF16),
    };

    if (!direct_fp32_input) {
        auto cast = Node("input_cast", "Cast", {"input"}, {"input_fp32"});
        cast.attributes["to"] = static_cast<int64_t>(1);
        model.graph.nodes.push_back(std::move(cast));
    } else {
        model.graph.nodes.push_back(Node("input_relay", "Identity", {"input"}, {"input_fp32"}));
    }
    model.graph.nodes.push_back(Node("square_node", "Mul", {"input_fp32", "input_fp32"}, {"square"}));
    auto mean = Node("mean_node", "ReduceMean", {"square"}, {"mean"});
    mean.attributes["axes"] = std::vector<int64_t>{-1};
    mean.attributes["keepdims"] = static_cast<int64_t>(1);
    model.graph.nodes.push_back(std::move(mean));
    model.graph.nodes.push_back(Node("eps_node", "Add", {"mean", "epsilon"}, {"mean_eps"}));
    model.graph.nodes.push_back(Node("sqrt_node", "Sqrt", {"mean_eps"}, {"root"}));
    model.graph.nodes.push_back(Node("div_node", "Div", {"input_fp32", "root"}, {"normalized"}));
    auto weight_cast = Node("weight_cast", "Cast", {"weight"}, {"weight_fp32"});
    weight_cast.attributes["to"] = static_cast<int64_t>(1);
    model.graph.nodes.push_back(std::move(weight_cast));
    model.graph.nodes.push_back(Node("scale_node", "Add", {"weight_fp32", "one"}, {"scale"}));
    model.graph.nodes.push_back(Node("scale_mul", "Mul", {"normalized", "scale"}, {"scaled"}));
    auto output_cast = Node("output_cast", "Cast", {"scaled"}, {"output"});
    output_cast.attributes["to"] = static_cast<int64_t>(16);
    model.graph.nodes.push_back(std::move(output_cast));
    return model;
}

ModelDesc BuildGatedRmsNormModel() {
    ModelDesc model;
    model.name = "qwen_gated_rms_norm_graph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"core_bf16"};
    model.graph.outputs = {"output"};
    model.graph.values = {
        Value("core_bf16", {1, 2, 4}, DataType::BF16),
        Value("weight", {4}, DataType::BF16, true),
        Value("epsilon", {1}, DataType::FP32, true),
        Value("core_fp32", {1, 2, 4}, DataType::FP32),
        Value("square", {1, 2, 4}, DataType::FP32),
        Value("mean", {1, 2, 1}, DataType::FP32),
        Value("mean_eps", {1, 2, 1}, DataType::FP32),
        Value("root", {1, 2, 1}, DataType::FP32),
        Value("normalized", {1, 2, 4}, DataType::FP32),
        Value("output", {1, 2, 4}, DataType::FP32),
    };

    auto input_cast = Node("input_cast", "Cast", {"core_bf16"}, {"core_fp32"});
    input_cast.attributes["to"] = static_cast<int64_t>(1);
    model.graph.nodes.push_back(std::move(input_cast));
    model.graph.nodes.push_back(Node("square_node", "Mul", {"core_fp32", "core_fp32"}, {"square"}));
    auto mean = Node("mean_node", "ReduceMean", {"square"}, {"mean"});
    mean.attributes["axes"] = std::vector<int64_t>{-1};
    mean.attributes["keepdims"] = static_cast<int64_t>(1);
    model.graph.nodes.push_back(std::move(mean));
    model.graph.nodes.push_back(Node("eps_node", "Add", {"mean", "epsilon"}, {"mean_eps"}));
    model.graph.nodes.push_back(Node("sqrt_node", "Sqrt", {"mean_eps"}, {"root"}));
    model.graph.nodes.push_back(Node("div_node", "Div", {"core_fp32", "root"}, {"normalized"}));
    model.graph.nodes.push_back(Node("scale_mul", "Mul", {"normalized", "weight"}, {"output"}));
    return model;
}

void BindInputs(StaticGraph* graph, bool direct_fp32_input) {
    if (direct_fp32_input) {
        auto input = std::make_shared<Tensor>();
        input->Assign<float>({1.0f, -2.0f, 0.5f, 3.0f}, {1, 1, 4});
        ASSERT_EQ(graph->SetTensor("input", input), 0);
    } else {
        std::vector<feather::BFloat16> values;
        for (float value : {1.0f, -2.0f, 0.5f, 3.0f}) {
            values.push_back(feather::BFloat16{feather::FloatToBFloat16(value)});
        }
        auto input = std::make_shared<Tensor>();
        input->Assign<feather::BFloat16>(values, {1, 1, 4});
        ASSERT_EQ(graph->SetTensor("input", input), 0);
    }
    auto weight = std::make_shared<Tensor>();
    std::vector<feather::BFloat16> weights;
    for (float value : {0.5f, 1.0f, -0.75f, 2.0f}) {
        weights.push_back(feather::BFloat16{feather::FloatToBFloat16(value)});
    }
    weight->Assign<feather::BFloat16>(weights, {4});
    ASSERT_EQ(graph->SetTensor("weight", weight), 0);
    auto epsilon = std::make_shared<Tensor>();
    epsilon->Assign<float>({1.0e-6f}, {1});
    ASSERT_EQ(graph->SetTensor("epsilon", epsilon), 0);
    auto one = std::make_shared<Tensor>();
    one->Assign<float>({1.0f}, {1});
    ASSERT_EQ(graph->SetTensor("one", one), 0);
}

TEST(qwen_rms_norm_fusion_pass_test, FusesBf16RmsNormChainIntoSingleOperator) {
    StaticGraph graph;
    graph.SetKernelDevice(DeviceType::X86);
    ASSERT_EQ(graph.SetModel(BuildRmsNormModel(false, false)), 0);
    BindInputs(&graph, false);
    ASSERT_EQ(graph.Build(), 0);

    auto passes = std::make_shared<PassManager>();
    passes->AddPass(std::make_unique<feather::QwenRmsNormFusionPass>());
    graph.SetPassManager(passes);
    ASSERT_EQ(graph.ApplyPasses(), 0);

    ASSERT_EQ(graph.NodeSize(), 1U);
    const auto* fused = graph.GetNode("output_cast");
    ASSERT_NE(fused, nullptr);
    EXPECT_EQ(fused->op_type, "QwenRmsNorm");
    EXPECT_EQ(fused->inputs, (std::vector<std::string>{"input", "weight", "epsilon"}));
    const auto model_node = std::find_if(graph.model().graph.nodes.begin(), graph.model().graph.nodes.end(),
                                         [](const auto& node) { return node.name == "output_cast"; });
    ASSERT_NE(model_node, graph.model().graph.nodes.end());
    ASSERT_TRUE(model_node->attributes.count("weight_offset") != 0);
    EXPECT_FLOAT_EQ(std::get<float>(model_node->attributes.at("weight_offset")), 1.0f);
    EXPECT_EQ(graph.GetNode("scale_mul"), nullptr);
    EXPECT_EQ(graph.GetNode("square_node"), nullptr);
}

TEST(qwen_rms_norm_fusion_pass_test, SkipsChainWhenIntermediateIsGraphOutput) {
    StaticGraph graph;
    graph.SetKernelDevice(DeviceType::X86);
    ASSERT_EQ(graph.SetModel(BuildRmsNormModel(true, false)), 0);
    BindInputs(&graph, false);
    ASSERT_EQ(graph.Build(), 0);

    auto passes = std::make_shared<PassManager>();
    passes->AddPass(std::make_unique<feather::QwenRmsNormFusionPass>());
    graph.SetPassManager(passes);
    ASSERT_EQ(graph.ApplyPasses(), 0);

    EXPECT_NE(graph.GetNode("output_cast"), nullptr);
    EXPECT_NE(graph.GetNode("square_node"), nullptr);
    EXPECT_EQ(graph.GetNode("output_cast")->op_type, "Cast");
}

TEST(qwen_rms_norm_fusion_pass_test, FusesFp32ChainWithoutFinalCast) {
    auto model = BuildRmsNormModel(false, true);
    model.graph.nodes.pop_back();
    model.graph.outputs = {"scaled"};

    StaticGraph graph;
    graph.SetKernelDevice(DeviceType::X86);
    ASSERT_EQ(graph.SetModel(model), 0);
    BindInputs(&graph, true);
    ASSERT_EQ(graph.Build(), 0);

    auto passes = std::make_shared<PassManager>();
    passes->AddPass(std::make_unique<feather::QwenRmsNormFusionPass>());
    graph.SetPassManager(passes);
    ASSERT_EQ(graph.ApplyPasses(), 0);

    ASSERT_EQ(graph.NodeSize(), 2U);
    const auto* fused = graph.GetNode("scale_mul");
    ASSERT_NE(fused, nullptr);
    EXPECT_EQ(fused->op_type, "QwenRmsNorm");
    EXPECT_EQ(fused->inputs, (std::vector<std::string>{"input_fp32", "weight", "epsilon"}));
}

TEST(qwen_rms_norm_fusion_pass_test, FusesGatedNormWithDirectWeightAndFp32Output) {
    StaticGraph graph;
    graph.SetKernelDevice(DeviceType::X86);
    ASSERT_EQ(graph.SetModel(BuildGatedRmsNormModel()), 0);

    std::vector<feather::BFloat16> input_values;
    for (float value : {1.0f, -2.0f, 0.5f, 3.0f, -1.5f, 2.0f, 0.25f, -0.75f}) {
        input_values.push_back(feather::BFloat16{feather::FloatToBFloat16(value)});
    }
    auto input = std::make_shared<Tensor>();
    input->Assign<feather::BFloat16>(input_values, {1, 2, 4});
    ASSERT_EQ(graph.SetTensor("core_bf16", input), 0);
    auto weight = std::make_shared<Tensor>();
    std::vector<feather::BFloat16> weight_values;
    for (float value : {0.5f, 1.0f, -0.75f, 2.0f}) {
        weight_values.push_back(feather::BFloat16{feather::FloatToBFloat16(value)});
    }
    weight->Assign<feather::BFloat16>(weight_values, {4});
    ASSERT_EQ(graph.SetTensor("weight", weight), 0);
    auto epsilon = std::make_shared<Tensor>();
    epsilon->Assign<float>({1.0e-6f}, {1});
    ASSERT_EQ(graph.SetTensor("epsilon", epsilon), 0);
    ASSERT_EQ(graph.Build(), 0);

    auto passes = std::make_shared<PassManager>();
    passes->AddPass(std::make_unique<feather::QwenRmsNormFusionPass>());
    graph.SetPassManager(passes);
    ASSERT_EQ(graph.ApplyPasses(), 0);

    ASSERT_EQ(graph.NodeSize(), 1U);
    const auto* fused = graph.GetNode("scale_mul");
    ASSERT_NE(fused, nullptr);
    EXPECT_EQ(fused->op_type, "QwenRmsNorm");
    EXPECT_EQ(fused->inputs, (std::vector<std::string>{"core_bf16", "weight", "epsilon"}));
}

}  // namespace
