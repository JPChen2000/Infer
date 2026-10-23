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

#ifndef FEATHER_CORE_STATEFUL_SESSION_H
#define FEATHER_CORE_STATEFUL_SESSION_H

#include "core/graph.h"

namespace feather {
struct StateBinding {
    std::string input_name;
    std::string output_name;
    int cache_axis{-1};
    bool stable_storage{false};
};

Status ApplyStateBinding(RuntimeGraph& graph, const StateBinding& binding, DeviceType device);

// Owns state bindings, while storage and backend resources belong to the graph.
class StatefulSession {
 public:
    explicit StatefulSession(RuntimeGraph& graph, DeviceType device = DeviceType::COMMON) : graph_(graph), device_(device) {}
    Status Bind(StateBinding binding);
    // Validate the complete batch and retain aliased outputs before committing.
    Status Advance();
    Status Reset();
 private:
    RuntimeGraph& graph_;
    DeviceType device_;
    std::vector<StateBinding> bindings_;
};
}  // namespace feather
#endif
