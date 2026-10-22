// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pass/reshape_chain_elimination_pass.h"

#include <algorithm>
#include <string>
#include <vector>

#include "core/static_graph.h"

namespace feather {

namespace {

const std::string kReshapeChainEliminationPassName = "ReshapeChainEliminationPass";

bool IsGraphOutput(const StaticGraph& graph, const std::string& value_name) {
    return graph.IsGraphOutputValue(value_name);
}

}  // namespace

const std::string& ReshapeChainEliminationPass::name() const { return kReshapeChainEliminationPassName; }

int32_t ReshapeChainEliminationPass::Run(StaticGraph* graph) {
    if (graph == nullptr) {
        return -1;
    }

    std::vector<std::string> reshape_node_names;
    for (const auto& node : graph->nodes()) {
        if (!node.removed && node.op_type == "Reshape") {
            reshape_node_names.push_back(node.name);
        }
    }

    for (const auto& node_name : reshape_node_names) {
        const auto* outer_reshape = graph->GetNode(node_name);
        if (outer_reshape == nullptr || outer_reshape->inputs.size() != 1 || outer_reshape->outputs.size() != 1) {
            continue;
        }

        const std::string& inner_output = outer_reshape->inputs[0];
        if (IsGraphOutput(*graph, inner_output)) {
            continue;
        }
        const auto producer_name = graph->GetProducer(inner_output);
        if (producer_name.empty()) {
            continue;
        }
        const auto* inner_reshape = graph->GetNode(producer_name);
        if (inner_reshape == nullptr || inner_reshape->op_type != "Reshape" || inner_reshape->inputs.size() != 1 ||
            inner_reshape->outputs.size() != 1) {
            continue;
        }
        const auto users = graph->GetUsers(inner_output);
        if (users.size() != 1 || users[0] != outer_reshape->name) {
            continue;
        }

        if (!graph->ReplaceInputValue(outer_reshape->name, inner_output, inner_reshape->inputs[0])) {
            return -1;
        }
        if (!graph->RemoveNode(inner_reshape->name)) {
            return -1;
        }
    }

    return 0;
}

}  // namespace feather
