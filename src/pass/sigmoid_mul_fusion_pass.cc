#include "pass/sigmoid_mul_fusion_pass.h"

#include <algorithm>
#include <string>
#include <vector>

#include "core/static_graph.h"

namespace feather {

namespace {

const std::string kSigmoidMulFusionPassName = "SigmoidMulFusionPass";

bool IsGraphOutput(const StaticGraph& graph, const std::string& value_name) {
    const auto& outputs = graph.model().graph.outputs;
    return std::find(outputs.begin(), outputs.end(), value_name) != outputs.end();
}

bool MatchSiluPattern(const StaticGraph& graph, const StaticNode& mul_node, std::string* sigmoid_node_name,
                      std::string* activation_input_name) {
    if (sigmoid_node_name == nullptr || activation_input_name == nullptr || mul_node.removed ||
        mul_node.op_type != "Mul" || mul_node.inputs.size() != 2 || mul_node.outputs.size() != 1) {
        return false;
    }

    for (size_t sigmoid_input_index = 0; sigmoid_input_index < 2; ++sigmoid_input_index) {
        const auto& sigmoid_output = mul_node.inputs[sigmoid_input_index];
        const auto& passthrough_input = mul_node.inputs[1 - sigmoid_input_index];
        const auto producer_name = graph.GetProducer(sigmoid_output);
        if (producer_name.empty()) {
            continue;
        }
        const auto* sigmoid_node = graph.GetNode(producer_name);
        if (sigmoid_node == nullptr || sigmoid_node->op_type != "Sigmoid" || sigmoid_node->inputs.size() != 1 ||
            sigmoid_node->outputs.size() != 1 || sigmoid_node->outputs[0] != sigmoid_output ||
            sigmoid_node->inputs[0] != passthrough_input) {
            continue;
        }
        if (IsGraphOutput(graph, sigmoid_output)) {
            continue;
        }
        const auto users = graph.GetUsers(sigmoid_output);
        if (users.size() != 1 || users[0] != mul_node.name) {
            continue;
        }
        *sigmoid_node_name = sigmoid_node->name;
        *activation_input_name = passthrough_input;
        return true;
    }
    return false;
}

}  // namespace

const std::string& SigmoidMulFusionPass::name() const { return kSigmoidMulFusionPassName; }

int32_t SigmoidMulFusionPass::Run(StaticGraph* graph) {
    if (graph == nullptr) {
        return -1;
    }

    std::vector<std::string> mul_node_names;
    for (const auto& node : graph->nodes()) {
        if (!node.removed && node.op_type == "Mul") {
            mul_node_names.push_back(node.name);
        }
    }

    for (const auto& mul_node_name : mul_node_names) {
        const auto* mul_node = graph->GetNode(mul_node_name);
        if (mul_node == nullptr) {
            continue;
        }

        std::string sigmoid_node_name;
        std::string activation_input_name;
        if (!MatchSiluPattern(*graph, *mul_node, &sigmoid_node_name, &activation_input_name)) {
            continue;
        }
        if (!graph->ReplaceNodeOp(mul_node_name, "SiLU", {activation_input_name})) {
            return -1;
        }
        if (!graph->RemoveNode(sigmoid_node_name)) {
            return -1;
        }
    }
    return 0;
}

}  // namespace feather
