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

#ifndef FEATHER_CORE_GRAPH_LOWERING_H
#define FEATHER_CORE_GRAPH_LOWERING_H

#include "core/graph.h"
#include "core/static_graph.h"

namespace feather {

class GraphLowering {
   public:
    int32_t Lower(const StaticGraph& static_graph, RuntimeGraph* runtime_graph) const;
    Status LowerStatus(const StaticGraph& static_graph, RuntimeGraph* runtime_graph) const;
};

}  // namespace feather

#endif  // FEATHER_CORE_GRAPH_LOWERING_H
