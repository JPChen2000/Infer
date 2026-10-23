// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#include "core/operator_semantics.h"

#include <algorithm>
#include <cstdint>
#include <mutex>
namespace feather {
namespace {
using Registry = std::unordered_map<std::string, OperatorSemantics>;
Registry Builtins() {
    Registry result;
    for (auto type : {"Reshape", "Expand", "Unsqueeze", "Squeeze", "Split", "ReduceMax", "TopK"})
        result[type].shape_value_inputs = {1};
    result["Slice"].shape_value_inputs = {1, 2, 3, 4};
    result["Resize"].shape_value_inputs = {2, 3};
    result["ConstantOfShape"].shape_value_inputs = {0};
    for (auto type :
         {"Shape", "Unsqueeze", "Squeeze", "Concat", "Cast", "ConstantOfShape", "Slice", "Mul", "Equal", "Where"})
        result[type].build_time_evaluable = true;
    result["Shape"].descriptor_only_evaluation = true;
    result["Identity"].aliases = {{0, 0, true}};
    for (auto type : {"Reshape", "Flatten", "Squeeze", "Unsqueeze", "Slice"}) result[type].aliases = {{0, 0, false}};
    result["NonMaxSuppression"].bounded_outputs = {0};
    result["NonMaxSuppression"].terminal_outputs = true;
    return result;
}
Registry& Entries() {
    static Registry entries = Builtins();
    return entries;
}
std::mutex& Mutex() {
    static std::mutex mutex;
    return mutex;
}
bool Contains(const std::vector<std::string>& values, const std::string& name) {
    return std::find(values.begin(), values.end(), name) != values.end();
}
bool Overlaps(const Tensor& a, const Tensor& b) {
    if (&a == &b) return true;
    size_t ab = 0, bb = 0;
    if (!CheckedTensorBytes(a.dims().data(), a.data_type(), &ab) ||
        !CheckedTensorBytes(b.dims().data(), b.data_type(), &bb))
        return true;
    if (!ab || !bb) return false;
    if (a.storage() && a.storage() == b.storage())
        return a.offset() <= b.offset() ? b.offset() - a.offset() < ab : a.offset() - b.offset() < bb;
    if (!a.storage() || !b.storage() || !a.storage()->materialized() || !b.storage()->materialized()) return false;
    auto ap = reinterpret_cast<uintptr_t>(a.raw_data()), bp = reinterpret_cast<uintptr_t>(b.raw_data());
    return ap <= bp ? bp - ap < ab : ap - bp < bb;
}
}  // namespace
OperatorSemantics GetOperatorSemantics(const std::string& type) {
    std::lock_guard<std::mutex> lock(Mutex());
    auto found = Entries().find(type);
    return found == Entries().end() ? OperatorSemantics{} : found->second;
}
void RegisterOperatorSemantics(const std::string& type, OperatorSemantics semantics) {
    std::lock_guard<std::mutex> lock(Mutex());
    Entries()[type] = std::move(semantics);
}
std::string WholeTensorAliasSource(const OperatorSemantics& semantics, const std::vector<std::string>& inputs,
                                   const std::vector<std::string>& outputs, const std::string& output) {
    for (const auto& alias : semantics.aliases)
        if (alias.whole_tensor && alias.input < inputs.size() && alias.output < outputs.size() &&
            outputs[alias.output] == output)
            return inputs[alias.input];
    return {};
}
Status ValidateOperatorBindings(const std::vector<OperatorEffectNode>& nodes,
                                const std::unordered_map<std::string, std::shared_ptr<Tensor>>& tensors,
                                const std::vector<std::string>* graph_inputs) {
    auto fail = [](const std::string& message) { return Status::Error(StatusCode::kInvalidArgument, message); };
    auto tensor = [&](const std::string& name) {
        auto it = tensors.find(name);
        return it == tensors.end() ? nullptr : it->second;
    };
    for (const auto& node : nodes) {
        const auto& s = node.semantics;
        for (auto slot : s.mutable_inputs) {
            if (slot >= node.inputs.size()) return fail("missing mutable input: " + node.name);
            const auto written = tensor(node.inputs[slot]);
            if (!written || written->is_immutable()) return fail("state must be mutable: " + node.inputs[slot]);
            for (size_t i = 0; i < node.inputs.size(); ++i)
                if (i != slot) {
                    const auto other = tensor(node.inputs[i]);
                    if (other && Overlaps(*written, *other))
                        return fail("mutable input aliases another input: " + node.name);
                }
            if (graph_inputs)
                for (const auto& input : *graph_inputs)
                    if (input != node.inputs[slot]) {
                        const auto other = tensor(input);
                        if (other && Overlaps(*written, *other))
                            return fail("mutable input aliases graph input: " + node.name);
                    }
        }
    }
    return Status::Ok();
}

Status ValidateOperatorEffects(const std::vector<OperatorEffectNode>& nodes,
                               const std::unordered_map<std::string, std::shared_ptr<Tensor>>& tensors,
                               const std::vector<std::string>* graph_inputs,
                               const std::vector<std::string>& graph_outputs, bool serial) {
    auto fail = [](const std::string& message) { return Status::Error(StatusCode::kBuildFailed, message); };
    auto bindings = ValidateOperatorBindings(nodes, tensors, graph_inputs);
    if (!bindings.ok()) return bindings;
    for (const auto& node : nodes) {
        const auto& s = node.semantics;
        if (s.input_count && *s.input_count != node.inputs.size()) return fail("input count mismatch: " + node.name);
        if (s.serial_execution && !serial) return fail("operator requires serial execution: " + node.name);
        for (auto slot : s.external_inputs) {
            if (slot >= node.inputs.size() || node.inputs[slot].empty())
                return fail("missing external input: " + node.name);
            const auto& name = node.inputs[slot];
            if (graph_inputs && !Contains(*graph_inputs, name)) return fail("state must be a graph input: " + name);
            for (const auto& other : nodes)
                if (Contains(other.outputs, name)) return fail("state must be external: " + name);
        }
        for (auto slot : s.exclusive_inputs) {
            if (slot >= node.inputs.size()) return fail("missing exclusive input: " + node.name);
            for (const auto& other : nodes)
                if (other.name != node.name && Contains(other.inputs, node.inputs[slot]))
                    return fail("state requires an exclusive consumer: " + node.inputs[slot]);
        }
        for (const auto& alias : s.aliases)
            if (alias.input >= node.inputs.size() || alias.output >= node.outputs.size())
                return fail("invalid alias declaration: " + node.name);
        for (auto slot : s.bounded_outputs)
            if (slot >= node.outputs.size()) return fail("invalid bounded output: " + node.name);
        if (s.terminal_outputs)
            for (const auto& output : node.outputs) {
                if (!Contains(graph_outputs, output))
                    return fail("bounded selection must be a graph output: " + output);
                for (const auto& other : nodes)
                    if (Contains(other.inputs, output))
                        return fail("data-dependent output consumers are unsupported: " + other.name);
            }
    }
    return Status::Ok();
}
}  // namespace feather
