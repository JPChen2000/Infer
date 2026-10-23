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

#ifndef FEATHER_CORE_EXECUTION_PLAN_H
#define FEATHER_CORE_EXECUTION_PLAN_H

#include "core/operator_semantics.h"
#include "core/graph.h"
#include "core/static_graph.h"

namespace feather {

class ExecutionSession;
struct SessionOptions {
    RuntimeThreadMode thread_mode{RuntimeThreadMode::kSerialGraph};
    size_t thread_count{1};
    bool reuse_memory{true};
    // Mutable host storage only; shared immutable weights are excluded.
    size_t host_memory_limit{static_cast<size_t>(-1)};
    size_t device_workspace_limit{static_cast<size_t>(-1)};
    size_t fp8_weight_cache_limit{0};
    std::function<std::unique_ptr<BackendContext>()> backend_factory;
    // Bind the exact caller-owned mutable input objects across shape plans.
    // View aliases that need separate Tensor objects are currently rejected.
    std::unordered_map<std::string, std::shared_ptr<Tensor>> mutable_state;
    // Optional source for equal immutable constants only. No source session
    // lifetime is required after creation; storage owns the shared resources.
    const ExecutionSession* immutable_weight_source{nullptr};
};

struct PlannedAllocation {
    size_t group{0};
    size_t offset{0};
    size_t bytes{0};
    size_t first_node{0};
    size_t last_node{0};
};

struct MemoryPlan {
    size_t arena_bytes{0};
    size_t unshared_bytes{0};
    std::vector<PlannedAllocation> allocations;
};

struct SessionMemoryUsage {
    size_t mutable_host_bytes{0};
    size_t shared_weight_bytes{0};
    size_t reused_weight_bytes{0};
    size_t arena_bytes{0};
};

class ExecutionSession;

// A snapshot of optimized semantics, fixed tensor descriptors and parameter
// recipes. It owns no mutable execution kernels, stream or activation arena.
class CompiledGraph {
 public:
    static Status Compile(const StaticGraph& graph, std::shared_ptr<const CompiledGraph>* result,
                          const std::unordered_set<std::string>& external_state_inputs = {});
    Status CreateSession(std::unique_ptr<ExecutionSession>* session, const SessionOptions& options = {}) const;
    const MemoryPlan& memory_plan() const { return memory_plan_; }
    size_t NodeSize() const { return nodes_.size(); }
 private:
    struct Value {
        std::string name;
        std::shared_ptr<Tensor> prototype;
        size_t group{0};
        size_t bytes{0};
        size_t offset{0};
        bool seed{false};
    };
    struct Node {
        std::string name, op_type;
        std::vector<std::string> inputs, outputs;
        DeviceType device{DeviceType::UNKNOWN};
        DataType dtype{DataType::UNKNOWN};
        DataLayout layout{DataLayout::ND};
        KernelBase::ParameterBinder bind;
        KernelBase::Factory factory;
        OperatorSemantics semantics;
    };
    friend class GraphLowering;
    Status LowerInto(RuntimeGraph* runtime, const StaticGraph& graph) const;
    Status InstallRuntime(RuntimeGraph& runtime, const KernelBase::TensorBindings& bindings) const;
    Status BuildMemoryPlan();
    std::vector<Value> values_;
    std::vector<Node> nodes_;
    std::vector<std::string> inputs_, outputs_;
    MemoryPlan memory_plan_;
    std::unordered_set<std::string> required_state_inputs_;
    std::unordered_map<std::string, std::vector<uint8_t>> control_values_;
};

class ExecutionSession {
 public:
    explicit ExecutionSession(std::unique_ptr<BackendContext> backend = nullptr) : runtime_(std::move(backend)) {}
    Status Run() { return runtime_.RunStatus(); }
    Status BindInput(const std::string& name, std::shared_ptr<Tensor> tensor);
    Status ReadOutput(const std::string& name, std::shared_ptr<Tensor>* output);
    RuntimeGraph& runtime() { return runtime_; }
    const RuntimeGraph& runtime() const { return runtime_; }
    bool ReusesMemory() const { return reuses_memory_; }
    const SessionMemoryUsage& memory_usage() const { return memory_usage_; }
 private:
    friend class CompiledGraph;
    RuntimeGraph runtime_;
    bool reuses_memory_{false};
    SessionMemoryUsage memory_usage_;
    std::unordered_set<std::string> inputs_, outputs_;
};

}  // namespace feather
#endif
