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

#include "core/graph_lowering.h"

#include "core/execution_plan.h"

namespace feather {

Status GraphLowering::LowerStatus(const StaticGraph& static_graph, RuntimeGraph* runtime_graph) const {
    if (!runtime_graph) return Status::Error(StatusCode::kInvalidArgument, "runtime graph is null");
    std::shared_ptr<const CompiledGraph> compiled;
    const auto status = CompiledGraph::Compile(static_graph, &compiled);
    return status.ok() ? compiled->LowerInto(runtime_graph, static_graph) : status;
}

int32_t GraphLowering::Lower(const StaticGraph& static_graph, RuntimeGraph* runtime_graph) const {
    return LowerStatus(static_graph, runtime_graph).ok() ? 0 : -1;
}

}  // namespace feather
