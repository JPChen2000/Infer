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

#include "pass/identity_elimination_pass.h"

#include <algorithm>
#include <vector>

#include "core/static_graph.h"

namespace feather {

namespace {

const std::string kIdentityEliminationPassName = "IdentityEliminationPass";

bool IsGraphOutput(const StaticGraph& graph, const std::string& value_name) {
    return graph.IsGraphOutputValue(value_name);
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
