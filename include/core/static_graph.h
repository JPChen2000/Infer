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

#ifndef FEATHER_CORE_STATIC_GRAPH_H
#define FEATHER_CORE_STATIC_GRAPH_H

#include <cstddef>
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/operator_semantics.h"
#include "core/operator.h"
#include "core/status.h"
#include "model/model_format.h"
#include "pass/graph_pass.h"

namespace feather {

struct StaticNode {
    std::string name;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::shared_ptr<OpBase> op;
    OperatorSemantics semantics;
    bool removed{false};
};

class StaticGraph {
   public:
    StaticGraph() = default;

    int32_t SetModel(const model::ModelDesc& model);
    int32_t SetTensor(const std::string& name, std::shared_ptr<Tensor> tensor);
    bool SetValueDataType(const std::string& name, DataType data_type);
    std::shared_ptr<Tensor> GetTensor(const std::string& name) const;
    void SetKernelDevice(DeviceType device) { kernel_device_ = device == DeviceType::UNKNOWN ? GetHostRuntimeDevice() : device; }
    void SetFallbackPolicy(KernelFallbackPolicy policy) { fallback_policy_ = policy; }
    void SetWorkspaceLimit(size_t bytes) { workspace_limit_ = bytes; }
    DeviceType KernelDevice() const { return kernel_device_; }

    int32_t Build();
    Status BuildStatus();
    int32_t Check() const;
    Status CheckStatus() const;
    Status ValidateStructureStatus() const;
    Status ValidateBuildDescriptorsStatus() const;
    int32_t ApplyPasses();
    Status ApplyPassesStatus();
    void SetPassManager(std::shared_ptr<PassManager> pass_manager);

    size_t OperatorSize() const;
    size_t NodeSize() const;
    const model::ModelDesc& model() const { return model_; }

    const StaticNode* GetNode(const std::string& name) const;
    std::string GetProducer(const std::string& value_name) const;
    std::vector<std::string> GetUsers(const std::string& value_name) const;
    bool IsGraphOutputValue(const std::string& value_name) const;
    bool RemoveNode(const std::string& node_name);
    // Removes a graph-output relay and binds its output name to the source
    // tensor. The relay must have one input, one graph-output value, and no
    // graph users so no dependency edge is lost.
    bool RemoveNodeWithOutputAlias(const std::string& node_name);
    bool ReplaceInputValue(const std::string& node_name, const std::string& from, const std::string& to);
    bool ReplaceNodeOp(const std::string& node_name, const std::string& op_type,
                       const std::vector<std::string>& inputs);
    bool ReplaceNodeDesc(const model::NodeDesc& desc);
    // Replaces a node while appending every output from one directly absorbed node.
    // The replacement must retain the target's original output prefix.
    bool ReplaceNodeDescAndAbsorbNode(const model::NodeDesc& desc, const std::string& absorbed_node_name);

    const std::unordered_map<std::string, std::shared_ptr<Tensor>>& tensors() const { return tensors_; }
    const std::unordered_set<std::string>& bound_tensor_names() const { return bound_tensor_names_; }
    const std::unordered_set<std::string>& build_time_values() const { return build_time_values_; }
    const std::unordered_map<std::string, std::vector<uint8_t>>& control_values() const { return control_values_; }
    const std::vector<StaticNode>& nodes() const { return nodes_; }

   private:
    void ClearGraphState();
    void RegisterValueUse(const std::string& value_name, const std::string& node_name);
    void UnregisterValueUse(const std::string& value_name, const std::string& node_name);
    bool RebuildNode(size_t node_index);
    Status RecordControlValues();
    void RecordBuildDescriptors();
    void RecordBuildDescriptor(const std::string& name);

    model::ModelDesc model_;
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors_;
    std::unordered_set<std::string> bound_tensor_names_;
    std::vector<StaticNode> nodes_;
    std::unordered_map<std::string, size_t> node_index_by_name_;
    std::unordered_map<std::string, std::string> producer_by_value_;
    std::unordered_map<std::string, std::vector<std::string>> users_by_value_;
    std::unordered_map<std::string, std::string> output_aliases_;
    std::shared_ptr<PassManager> pass_manager_;
    DeviceType kernel_device_{GetHostRuntimeDevice()};
    KernelFallbackPolicy fallback_policy_{KernelFallbackPolicy::kAllowCommon};
    size_t workspace_limit_{static_cast<size_t>(-1)};
    std::unordered_set<std::string> build_time_values_;
    std::unordered_map<std::string, std::vector<uint8_t>> control_values_;
    struct BuildDescriptor {
        model::TensorDesc descriptor;
        std::shared_ptr<const Tensor> tensor;
    };
    std::unordered_map<std::string, BuildDescriptor> build_descriptors_;
    bool built_{false};
};

}  // namespace feather

#endif  // FEATHER_CORE_STATIC_GRAPH_H
