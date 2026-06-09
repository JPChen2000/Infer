#include "pass/no_op_elimination_pass.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include "core/static_graph.h"
#include "model/model_format.h"

namespace feather {

namespace {

const std::string kNoOpEliminationPassName = "NoOpEliminationPass";

bool IsGraphOutput(const StaticGraph& graph, const std::string& value_name) {
    const auto& outputs = graph.model().graph.outputs;
    return std::find(outputs.begin(), outputs.end(), value_name) != outputs.end();
}

const model::NodeDesc* FindModelNode(const StaticGraph& graph, const std::string& name) {
    for (const auto& node : graph.model().graph.nodes) {
        if (node.name == name) {
            return &node;
        }
    }
    return nullptr;
}

std::vector<int64_t> GetVectorAttribute(const model::NodeDesc& node, const std::string& key) {
    const auto it = node.attributes.find(key);
    if (it == node.attributes.end()) {
        return {};
    }
    if (const auto value = std::get_if<std::vector<int64_t>>(&it->second); value != nullptr) {
        return *value;
    }
    return {};
}

bool IsIdentityPermutation(const std::vector<int64_t>& perm) {
    if (perm.empty()) {
        return false;
    }
    for (size_t i = 0; i < perm.size(); ++i) {
        if (perm[i] != static_cast<int64_t>(i)) {
            return false;
        }
    }
    return true;
}

bool ReplaceAllUsers(StaticGraph* graph, const std::string& from, const std::string& to) {
    const auto users = graph->GetUsers(from);
    for (const auto& user : users) {
        if (!graph->ReplaceInputValue(user, from, to)) {
            return false;
        }
    }
    return true;
}

}  // namespace

const std::string& NoOpEliminationPass::name() const { return kNoOpEliminationPassName; }

int32_t NoOpEliminationPass::Run(StaticGraph* graph) {
    if (graph == nullptr) {
        return -1;
    }

    std::vector<std::string> removable_nodes;
    for (const auto& node : graph->nodes()) {
        if (node.removed || node.inputs.size() != 1 || node.outputs.size() != 1) {
            continue;
        }
        if (IsGraphOutput(*graph, node.outputs[0])) {
            continue;
        }

        const auto input_tensor = graph->GetTensor(node.inputs[0]);
        const auto output_tensor = graph->GetTensor(node.outputs[0]);
        if (input_tensor == nullptr || output_tensor == nullptr) {
            continue;
        }

        bool removable = false;
        if (node.op_type == "Reshape") {
            removable = input_tensor->dims().data() == output_tensor->dims().data();
        } else if (node.op_type == "Transpose") {
            const auto* model_node = FindModelNode(*graph, node.name);
            if (model_node != nullptr) {
                removable = IsIdentityPermutation(GetVectorAttribute(*model_node, "perm"));
            }
        }

        if (removable) {
            removable_nodes.push_back(node.name);
        }
    }

    for (const auto& node_name : removable_nodes) {
        const auto* node = graph->GetNode(node_name);
        if (node == nullptr) {
            continue;
        }
        if (!ReplaceAllUsers(graph, node->outputs[0], node->inputs[0])) {
            return -1;
        }
        if (!graph->RemoveNode(node_name)) {
            return -1;
        }
    }

    return 0;
}

}  // namespace feather
