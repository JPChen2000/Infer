// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#ifndef FEATHER_CORE_OPERATOR_SEMANTICS_H
#define FEATHER_CORE_OPERATOR_SEMANTICS_H
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/status.h"
#include "core/tensor.h"
namespace feather {
struct OutputAlias {
    size_t output{0}, input{0};
    bool whole_tensor{false};
};
// Value semantics are copied into each built node. Later registrations cannot
// change an existing graph's shape dependencies or scheduling requirements.
struct OperatorSemantics {
    std::vector<size_t> shape_value_inputs;
    bool build_time_evaluable{false};
    bool descriptor_only_evaluation{false};
    std::vector<size_t> mutable_inputs, external_inputs, exclusive_inputs;
    std::vector<OutputAlias> aliases;
    std::vector<size_t> bounded_outputs;
    bool terminal_outputs{false};
    bool serial_execution{false};
    std::optional<size_t> input_count;
    std::function<Status(const std::vector<std::shared_ptr<Tensor>>&)> validate_runtime;
};
OperatorSemantics GetOperatorSemantics(const std::string& type);
void RegisterOperatorSemantics(const std::string& type, OperatorSemantics semantics);
struct OperatorEffectNode {
    std::string name;
    std::vector<std::string> inputs, outputs;
    OperatorSemantics semantics;
};
Status ValidateOperatorEffects(const std::vector<OperatorEffectNode>& nodes,
                               const std::unordered_map<std::string, std::shared_ptr<Tensor>>& tensors,
                               const std::vector<std::string>* graph_inputs,
                               const std::vector<std::string>& graph_outputs, bool serial_execution);
// Revalidate only properties that may change when callers rebind storage.
Status ValidateOperatorBindings(const std::vector<OperatorEffectNode>& nodes,
                                const std::unordered_map<std::string, std::shared_ptr<Tensor>>& tensors,
                                const std::vector<std::string>* graph_inputs);
// Returns the declared whole-tensor relay source, or an empty name for views.
std::string WholeTensorAliasSource(const OperatorSemantics& semantics, const std::vector<std::string>& inputs,
                                   const std::vector<std::string>& outputs, const std::string& output);
}  // namespace feather
#endif
