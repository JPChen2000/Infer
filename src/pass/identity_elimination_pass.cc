#include "pass/identity_elimination_pass.h"

#include <algorithm>
#include <vector>

#include "core/static_graph.h"

namespace feather {

namespace {

const std::string kIdentityEliminationPassName = "IdentityEliminationPass";

bool IsGraphOutput(const StaticGraph& graph, const std::string& value_name) {
    const auto& outputs = graph.model().graph.outputs;
    return std::find(outputs.begin(), outputs.end(), value_name) != outputs.end();
}

}  // namespace

const std::string& IdentityEliminationPass::name() const { return kIdentityEliminationPassName; }

int32_t IdentityEliminationPass::Run(StaticGraph* graph) {
    if (graph == nullptr) {
        return -1;
    }

    std::vector<std::string> identity_node_names;
    for (const auto& node : graph->nodes()) {
        if (!node.removed && node.op_type == "Identity") {
            identity_node_names.push_back(node.name);
        }
    }

    for (const auto& node_name : identity_node_names) {
        const auto* node = graph->GetNode(node_name);
        if (node == nullptr || node->inputs.size() != 1 || node->outputs.size() != 1) {
            continue;
        }
        const std::string& input_name = node->inputs[0];
        const std::string& output_name = node->outputs[0];
        if (IsGraphOutput(*graph, output_name)) {
            continue;
        }
        const auto users = graph->GetUsers(output_name);
        bool ok = true;
        for (const auto& user : users) {
            if (!graph->ReplaceInputValue(user, output_name, input_name)) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            return -1;
        }
        if (!graph->RemoveNode(node_name)) {
            return -1;
        }
    }

    return 0;
}

}  // namespace feather
