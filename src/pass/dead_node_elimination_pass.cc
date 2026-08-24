#include "pass/dead_node_elimination_pass.h"

#include <algorithm>
#include <vector>

#include "core/static_graph.h"

namespace feather {

namespace {

const std::string kDeadNodeEliminationPassName = "DeadNodeEliminationPass";

bool IsGraphOutput(const StaticGraph& graph, const std::string& value_name) {
    return graph.IsGraphOutputValue(value_name);
}

}  // namespace

const std::string& DeadNodeEliminationPass::name() const { return kDeadNodeEliminationPassName; }

int32_t DeadNodeEliminationPass::Run(StaticGraph* graph) {
    if (graph == nullptr) {
        return -1;
    }

    bool changed = true;
    while (changed) {
        changed = false;
        std::vector<std::string> removable_nodes;
        for (const auto& node : graph->nodes()) {
            if (node.removed) {
                continue;
            }

            bool has_live_output = false;
            for (const auto& output_name : node.outputs) {
                if (IsGraphOutput(*graph, output_name) || !graph->GetUsers(output_name).empty()) {
                    has_live_output = true;
                    break;
                }
            }

            if (!has_live_output) {
                removable_nodes.push_back(node.name);
            }
        }

        for (const auto& node_name : removable_nodes) {
            if (graph->RemoveNode(node_name)) {
                changed = true;
            }
        }
    }

    return 0;
}

}  // namespace feather
