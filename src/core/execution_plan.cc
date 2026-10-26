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

#include "core/execution_plan.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <queue>

namespace feather {
namespace {
Status BuildError(const std::string& message) { return Status::Error(StatusCode::kBuildFailed, message); }

bool TensorRangesOverlap(const Tensor& a, const Tensor& b) {
    if (&a == &b) return true;
    size_t a_bytes = 0, b_bytes = 0;
    if (!CheckedTensorBytes(a.dims().data(), a.data_type(), &a_bytes) ||
        !CheckedTensorBytes(b.dims().data(), b.data_type(), &b_bytes))
        return true;
    if (a_bytes == 0 || b_bytes == 0) return false;
    if (a.storage() && a.storage() == b.storage()) {
        const size_t first = a.offset(), second = b.offset();
        return first <= second ? second - first < a_bytes : first - second < b_bytes;
    }
    // Caller bindings can wrap overlapping addresses in distinct Storage
    // objects. Keep deferred allocations lazy while checking such bindings.
    if (!a.storage() || !b.storage() || !a.storage()->materialized() || !b.storage()->materialized() ||
        !a.IsInitialized() || !b.IsInitialized())
        return false;
    const auto first = reinterpret_cast<std::uintptr_t>(a.raw_data());
    const auto second = reinterpret_cast<std::uintptr_t>(b.raw_data());
    return first <= second ? second - first < a_bytes : first - second < b_bytes;
}

std::shared_ptr<Tensor> Descriptor(const Tensor& source) {
    auto result = std::make_shared<Tensor>();
    result->Resize(source.dims());
    result->set_data_type(source.data_type());
    result->set_layout(source.layout());
    result->set_quantization(source.quantization());
    result->set_immutable(source.is_immutable());
    return result;
}

std::shared_ptr<Buffer> FrozenBytes(const Tensor& tensor, size_t bytes) {
    // Mapped weights already enforce copy-on-write, so keep the mapping alive.
    if (tensor.storage() && tensor.storage()->read_only()) {
        return std::make_shared<Buffer>(tensor.raw_data(), bytes, tensor.storage()->buffer(), true);
    }
    Tensor copy(std::make_shared<Buffer>(bytes));
    if (bytes) std::memcpy(copy.mutable_data(bytes), tensor.raw_data(), bytes);
    return std::make_shared<Buffer>(copy.raw_data(), bytes, copy.storage()->buffer(), true);
}
}

Status CompiledGraph::Compile(const StaticGraph& graph, std::shared_ptr<const CompiledGraph>* result,
                              const std::unordered_set<std::string>& external_state_inputs) {
    if (!result) return Status::Error(StatusCode::kInvalidArgument, "null compiled graph destination");
    const auto checked = graph.ValidateStructureStatus();
    if (!checked.ok()) return checked;
    const auto descriptors = graph.ValidateBuildDescriptorsStatus();
    if (!descriptors.ok()) return descriptors;
    try {
        auto plan = std::shared_ptr<CompiledGraph>(new CompiledGraph);
        plan->inputs_ = graph.model().graph.inputs;
        plan->outputs_ = graph.model().graph.outputs;
        plan->required_state_inputs_ = external_state_inputs;
        plan->control_values_ = graph.control_values();
        for (const auto& control : plan->control_values_) {
            const auto tensor = graph.GetTensor(control.first);
            size_t bytes = 0;
            if (!tensor || !tensor->IsInitialized() || tensor->memory_size() < control.second.size() ||
                !CheckedTensorBytes(tensor->dims().data(), tensor->data_type(), &bytes) ||
                bytes != control.second.size() ||
                (control.second.size() &&
                 std::memcmp(tensor->raw_data(), control.second.data(), control.second.size()) != 0))
                return BuildError("build control values changed before compilation: " + control.first);
            if (!graph.GetProducer(control.first).empty() && !graph.build_time_values().count(control.first))
                return BuildError("data-dependent shape controls are unsupported: " + control.first);
            if (external_state_inputs.count(control.first))
                return BuildError("shape controls cannot be mutable external state: " + control.first);
        }
        std::unordered_set<const Tensor*> external_state_tensors;
        for (const auto& name : external_state_inputs) {
            const auto tensor = graph.GetTensor(name);
            if (!tensor || tensor->is_immutable() ||
                std::find(plan->inputs_.begin(), plan->inputs_.end(), name) == plan->inputs_.end())
                return BuildError("external state must be a mutable graph input: " + name);
            external_state_tensors.insert(tensor.get());
            for (const auto& alias : graph.tensors())
                if (alias.second && alias.second != tensor && alias.second->storage() == tensor->storage())
                    return BuildError("external state view aliases are unsupported: " + name);
            for (const auto& node : graph.nodes())
                if (!node.removed)
                    for (const auto& output : node.outputs) {
                        const auto produced = graph.GetTensor(output);
                        if (output == name || (produced && produced->storage() == tensor->storage()))
                            return BuildError("external state aliases a produced value: " + name);
                    }
        }
        std::vector<OperatorEffectNode> effects;
        for (const auto& node : graph.nodes())
            if (!node.removed) effects.push_back({node.name, node.inputs, node.outputs, node.semantics});
        auto effect_status = ValidateOperatorEffects(effects, graph.tensors(), &plan->inputs_, plan->outputs_, true);
        if (!effect_status.ok()) return effect_status;
        KernelBase::TensorBindings frozen;
        std::unordered_map<const Storage*, size_t> groups;
        for (const auto& item : graph.tensors()) {
            const auto& source = item.second;
            if (!source) return BuildError("null tensor: " + item.first);
            size_t bytes = 0;
            if (!CheckedTensorBytes(source->dims().data(), source->data_type(), &bytes))
                return BuildError("invalid tensor descriptor: " + item.first);
            auto found = frozen.find(source.get());
            std::shared_ptr<Tensor> prototype;
            const bool seed =
                !external_state_tensors.count(source.get()) &&
                (graph.build_time_values().count(item.first) != 0 || graph.bound_tensor_names().count(item.first) != 0);
            if (found != frozen.end()) prototype = found->second;
            else {
                prototype = Descriptor(*source);
                if (seed) {
                    if (!source->IsInitialized() || source->memory_size() < bytes)
                        return BuildError("uninitialized seed: " + item.first);
                    prototype->ResetBuffer(FrozenBytes(*source, bytes), bytes);
                }
                frozen[source.get()] = prototype;
            }
            if (seed && !prototype->IsInitialized()) prototype->ResetBuffer(FrozenBytes(*source, bytes), bytes);
            auto group = groups.emplace(source->storage().get(), groups.size()).first->second;
            plan->values_.push_back({item.first, prototype, group, bytes, source->offset(), seed});
        }
        // Stable records make allocation decisions reproducible across hash seeds.
        std::sort(plan->values_.begin(), plan->values_.end(), [](const Value& a, const Value& b) { return a.name < b.name; });
        for (const auto& node : graph.nodes()) {
            if (node.removed) continue;
            const auto* kernel = node.op ? node.op->kernel() : nullptr;
            if (!kernel) return BuildError("missing kernel schema: " + node.name);
            auto snapshot = kernel->SnapshotParameters();
            auto temporary = std::make_unique<DeferredKernel>();
            if (!snapshot || !kernel->InstanceFactory()) return BuildError("kernel cannot be snapshotted: " + node.name);
            auto status = snapshot(*temporary, frozen);
            if (!status.ok()) return status.WithContext("compile node=" + node.name);
            plan->nodes_.push_back({node.name, node.op_type, node.inputs, node.outputs, kernel->device(),
                                    kernel->data_type(), kernel->layout(), temporary->SnapshotParameters(),
                                    kernel->InstanceFactory(), node.semantics});
        }
        // A fusion may make an earlier node depend on a later node.
        std::unordered_map<std::string, size_t> producers;
        for (size_t i = 0; i < plan->nodes_.size(); ++i)
            for (const auto& output : plan->nodes_[i].outputs)
                if (!producers.emplace(output, i).second) return BuildError("duplicate producer: " + output);
        std::vector<size_t> pending(plan->nodes_.size());
        std::vector<std::vector<size_t>> users(plan->nodes_.size());
        for (size_t i = 0; i < plan->nodes_.size(); ++i) {
            for (const auto& input : plan->nodes_[i].inputs) {
                auto producer = producers.find(input);
                if (producer != producers.end()) { ++pending[i]; users[producer->second].push_back(i); }
                else if (!input.empty() && !graph.GetTensor(input)) return BuildError("missing input: " + input);
            }
        }
        std::priority_queue<size_t, std::vector<size_t>, std::greater<size_t>> ready;
        for (size_t i = 0; i < pending.size(); ++i) if (!pending[i]) ready.push(i);
        std::vector<Node> ordered;
        while (!ready.empty()) {
            const auto i = ready.top(); ready.pop(); ordered.push_back(std::move(plan->nodes_[i]));
            for (const auto user : users[i]) if (--pending[user] == 0) ready.push(user);
        }
        if (ordered.size() != plan->nodes_.size()) return BuildError("cycle in optimized graph");
        plan->nodes_ = std::move(ordered);
        const auto memory = plan->BuildMemoryPlan();
        if (!memory.ok()) return memory;
        *result = std::move(plan);
        return Status::Ok();
    } catch (const std::exception& error) {
        return BuildError(error.what()).WithContext("compile");
    }
}

Status CompiledGraph::BuildMemoryPlan() {
    struct Lifetime { size_t bytes{0}, first{0}, last{0}; bool produced{false}, pinned{false}, used{false}; };
    std::unordered_map<size_t, Lifetime> lifetimes;
    std::unordered_map<std::string, const Value*> values;
    for (const auto& value : values_) {
        values[value.name] = &value;
        if (value.offset > std::numeric_limits<size_t>::max() - value.bytes) return BuildError("storage range overflow");
        auto& life = lifetimes[value.group];
        life.bytes = std::max(life.bytes, value.offset + value.bytes);
        life.pinned = life.pinned || value.seed;
    }
    for (size_t i = 0; i < nodes_.size(); ++i) {
        for (const auto& output : nodes_[i].outputs) {
            if (!values.count(output)) return BuildError("missing output descriptor: " + output);
            auto& life = lifetimes[values[output]->group];
            life.first = life.produced ? std::min(life.first, i) : i;
            life.last = std::max(life.last, i); life.produced = life.used = true;
        }
        for (const auto& input : nodes_[i].inputs) {
            if (input.empty()) continue;
            if (!values.count(input)) return BuildError("missing input descriptor: " + input);
            auto& life = lifetimes[values[input]->group];
            life.last = std::max(life.last, i); life.used = true;
        }
    }
    for (const auto& name : outputs_) {
        if (!values.count(name)) return BuildError("missing graph output: " + name);
        auto& life = lifetimes[values[name]->group]; life.last = nodes_.size(); life.used = true;
    }
    std::vector<PlannedAllocation> intervals;
    for (const auto& item : lifetimes) {
        const auto& life = item.second;
        if (!life.used || life.pinned || !life.produced || !life.bytes) continue;
        intervals.push_back({item.first, 0, life.bytes, life.first, life.last});
    }
    std::sort(intervals.begin(), intervals.end(), [](const PlannedAllocation& a, const PlannedAllocation& b) {
        return a.first_node != b.first_node ? a.first_node < b.first_node : a.group < b.group;
    });
    struct Slot { size_t offset, bytes, last; };
    std::vector<Slot> slots;
    for (auto interval : intervals) {
        // Buffer's base is sufficiently aligned for native types; view offsets
        // preserve that alignment. Arena slots use 64-byte size boundaries.
        if (interval.bytes > std::numeric_limits<size_t>::max() - 63) return BuildError("arena size overflow");
        const size_t rounded = (interval.bytes + 63) / 64 * 64;
        Slot* best = nullptr;
        for (auto& slot : slots)
            if (slot.last < interval.first_node && slot.bytes >= rounded && (!best || slot.bytes < best->bytes)) best = &slot;
        if (best) { interval.offset = best->offset; best->last = interval.last_node; }
        else {
            if (memory_plan_.arena_bytes > std::numeric_limits<size_t>::max() - rounded) return BuildError("arena size overflow");
            interval.offset = memory_plan_.arena_bytes;
            slots.push_back({interval.offset, rounded, interval.last_node}); memory_plan_.arena_bytes += rounded;
        }
        if (memory_plan_.unshared_bytes > std::numeric_limits<size_t>::max() - rounded) return BuildError("arena size overflow");
        memory_plan_.unshared_bytes += rounded;
        memory_plan_.allocations.push_back(interval);
    }
    return Status::Ok();
}

Status CompiledGraph::InstallRuntime(RuntimeGraph& runtime, const KernelBase::TensorBindings& bindings) const {
    std::unordered_set<std::string> produced, bounded;
    for (const auto& node : nodes_) {
        produced.insert(node.outputs.begin(), node.outputs.end());
        for (auto slot : node.semantics.bounded_outputs) bounded.insert(node.outputs.at(slot));
    }
    for (const auto& value : values_) {
        auto tensor = runtime.GetTensor(value.name);
        if (!tensor) return BuildError("missing runtime binding: " + value.name);
        runtime.contracts_.push_back({value.name, tensor, value.prototype, value.bytes, !produced.count(value.name),
                                      bounded.count(value.name) != 0});
        auto control = control_values_.find(value.name);
        if (control != control_values_.end())
            runtime.contracts_.back().fixed_values = std::make_shared<const std::vector<uint8_t>>(control->second);
        if (tensor->is_immutable()) runtime.Backend().MarkPersistent(*tensor, true);
    }
    for (const auto& node : nodes_) {
        for (auto slot : node.semantics.mutable_inputs)
            runtime.Backend().MarkPersistent(*runtime.GetTensor(node.inputs.at(slot)), true);
        RuntimeNode executable;
        executable.name = node.name;
        executable.op_type = node.op_type;
        executable.inputs = node.inputs;
        executable.outputs = node.outputs;
        executable.kernel_device = node.device;
        executable.semantics = node.semantics;
        if (node.device == DeviceType::X86 && !HostSupportsX86Kernels())
            return BuildError("compiled x86 backend is unavailable: " + node.name);
        executable.kernel = node.factory ? node.factory() : nullptr;
        if (!executable.kernel || executable.kernel->device() != node.device)
            return BuildError("compiled backend is unavailable: " + node.name);
        auto status = node.bind(*executable.kernel, bindings);
        if (!status.ok()) return status.WithContext("instantiate node=" + node.name);
        status = executable.kernel->Supports();
        if (!status.ok()) return status.WithContext("instantiate node=" + node.name);
        runtime.AddNode(std::move(executable));
    }
    runtime.CaptureTensorBindings();
    return Status::Ok();
}

Status CompiledGraph::LowerInto(RuntimeGraph* runtime, const StaticGraph& graph) const {
    if (!runtime) return BuildError("null runtime destination");
    try {
        auto scope = runtime->Backend().Enter();
        auto mode = runtime->ThreadMode();
        runtime->Clear();
        runtime->SetThreadMode(mode);
        runtime->SetOutputNames(outputs_);
        KernelBase::TensorBindings bindings;
        for (const auto& value : values_) {
            auto tensor = graph.GetTensor(value.name);
            if (!tensor || runtime->SetTensor(value.name, tensor) != 0)
                return BuildError("missing legacy tensor: " + value.name);
            bindings[value.prototype.get()] = tensor;
        }
        for (const auto& node : nodes_)
            if (!runtime->Backend().Capabilities(node.device).parallel_nodes)
                runtime->SetThreadMode(RuntimeThreadMode::kSerialGraph);
        auto status = InstallRuntime(*runtime, bindings);
        if (!status.ok()) return status;
        status = runtime->FinalizeStatus();
        return status.ok() ? runtime->CheckStatus() : status;
    } catch (const std::exception& error) {
        return BuildError(error.what()).WithContext("lower graph");
    }
}

Status CompiledGraph::CreateSession(std::unique_ptr<ExecutionSession>* result, const SessionOptions& options) const {
    if (!result) return Status::Error(StatusCode::kInvalidArgument, "null session destination");
    try {
        auto session = std::make_unique<ExecutionSession>(options.backend_factory ? options.backend_factory() : nullptr);
        auto& runtime = session->runtime_;
        auto scope = runtime.Backend().Enter();
        runtime.SetThreadMode(options.thread_mode); runtime.SetThreadCount(options.thread_count);
        runtime.SetOutputNames(outputs_);
        session->outputs_.insert(outputs_.begin(), outputs_.end());
        std::unordered_set<std::string> produced, cuda_inputs;
        for (const auto& node : nodes_)
            if (node.device == DeviceType::CUDA) cuda_inputs.insert(node.inputs.begin(), node.inputs.end());
        for (const auto& node : nodes_) for (const auto& name : node.outputs) produced.insert(name);
        KernelBase::TensorBindings bindings;
        std::unordered_map<size_t, std::shared_ptr<const Storage>> storage;
        bool can_reuse = true;
        for (const auto& node : nodes_) {
            if (node.semantics.serial_execution && options.thread_mode != RuntimeThreadMode::kSerialGraph)
                return BuildError("operator requires serial graph execution: " + node.name);
            const auto caps = runtime.Backend().Capabilities(node.device);
            if (!caps.supported) return BuildError("backend cannot execute node=" + node.name);
            if (!caps.parallel_nodes && options.thread_mode != RuntimeThreadMode::kSerialGraph)
                return BuildError("backend requires serial execution: " + node.name);
            can_reuse = can_reuse && caps.activation_reuse;
        }
        session->reuses_memory_ = options.reuse_memory && can_reuse && options.thread_mode == RuntimeThreadMode::kSerialGraph;
        std::unordered_map<size_t, size_t> group_bytes;
        std::unordered_map<size_t, bool> immutable_groups;
        std::unordered_map<size_t, std::vector<const Value*>> group_values;
        std::unordered_set<size_t> seeded_groups;
        for (const auto& value : values_) {
            group_bytes[value.group] = std::max(group_bytes[value.group], value.offset + value.bytes);
            group_values[value.group].push_back(&value);
            if (value.seed) {
                if (seeded_groups.insert(value.group).second) immutable_groups[value.group] = true;
                immutable_groups[value.group] = immutable_groups[value.group] && value.prototype->is_immutable();
            }
        }
        // External state is validated before allocations or weight imports.
        std::unordered_map<std::string, const Value*> named_values;
        for (const auto& value : values_) named_values.emplace(value.name, &value);
        for (const auto& name : required_state_inputs_)
            if (!options.mutable_state.count(name)) return BuildError("missing external state binding: " + name);
        std::unordered_map<const Tensor*, std::shared_ptr<Tensor>> external_prototypes;
        std::unordered_set<size_t> external_groups;
        for (const auto& item : options.mutable_state) {
            const auto found = named_values.find(item.first);
            if (found == named_values.end() || std::find(inputs_.begin(), inputs_.end(), item.first) == inputs_.end() ||
                produced.count(item.first))
                return BuildError("state binding is not a graph input: " + item.first);
            const auto& value = *found->second;
            const auto& tensor = item.second;
            const auto& descriptor = value.prototype;
            if (!tensor || !tensor->IsInitialized() || tensor->is_immutable() || descriptor->is_immutable() ||
                tensor->storage()->read_only() || tensor->dims() != descriptor->dims() ||
                tensor->data_type() != descriptor->data_type() || tensor->layout() != descriptor->layout() ||
                !HasSameQuantization(tensor->quantization(), descriptor->quantization()) ||
                tensor->memory_size() < value.bytes || tensor->offset() > tensor->storage()->size() ||
                value.bytes > tensor->storage()->size() - tensor->offset())
                return BuildError("incompatible external state: " + item.first);
            for (const auto* alias : group_values[value.group])
                if (alias->prototype != descriptor)
                    return BuildError("external state view aliases are unsupported: " + item.first);
            auto inserted = external_prototypes.emplace(descriptor.get(), tensor);
            if (!inserted.second && inserted.first->second != tensor)
                return BuildError("external state aliases require the same Tensor: " + item.first);
            for (const auto& other : options.mutable_state) {
                if (other.first >= item.first || !other.second) continue;
                const auto other_value = named_values.find(other.first);
                if (other_value != named_values.end() && other_value->second->prototype == descriptor &&
                    other.second == tensor)
                    continue;
                if (TensorRangesOverlap(*tensor, *other.second))
                    return BuildError("overlapping external state bindings: " + item.first + " and " + other.first);
            }
            external_groups.insert(value.group);
        }
        for (const auto& external : external_prototypes)
            for (const auto& value : values_)
                if (value.prototype->is_immutable() && value.prototype->IsInitialized() &&
                    TensorRangesOverlap(*external.second, *value.prototype))
                    return BuildError("external state overlaps immutable weight: " + value.name);
        auto& usage = session->memory_usage_;
        std::unordered_set<size_t> arena_groups;
        if (session->reuses_memory_) {
            usage.arena_bytes = memory_plan_.arena_bytes;
            for (const auto& allocation : memory_plan_.allocations) arena_groups.insert(allocation.group);
        }
        usage.mutable_host_bytes = usage.arena_bytes;
        for (const auto& group : group_bytes) {
            auto& total = immutable_groups[group.first] ? usage.shared_weight_bytes : usage.mutable_host_bytes;
            if (arena_groups.count(group.first)) continue;
            if (group.second > std::numeric_limits<size_t>::max() - total) return BuildError("session storage size overflow");
            total += group.second;
        }
        if (usage.mutable_host_bytes > options.host_memory_limit) return BuildError("session host memory budget exceeded");
        runtime.Backend().SetWorkspaceLimit(options.device_workspace_limit);
        runtime.Backend().SetFp8WeightCacheLimit(options.fp8_weight_cache_limit);
        if (session->reuses_memory_ && memory_plan_.arena_bytes) {
            Tensor arena(std::make_shared<Buffer>(memory_plan_.arena_bytes));
            auto* arena_data = static_cast<char*>(arena.mutable_data(memory_plan_.arena_bytes));
            for (const auto& allocation : memory_plan_.allocations) {
                auto buffer = std::make_shared<Buffer>(arena_data + allocation.offset, allocation.bytes,
                                                       arena.storage()->buffer());
                storage[allocation.group] = std::make_shared<Storage>(std::move(buffer));
            }
        }
        // Keep constant views on the same read-only allocation. This also
        // preserves mapped-weight sharing when the seed covers the full group.
        for (const auto& group : immutable_groups) {
            if (!group.second) continue;
            std::shared_ptr<const Storage> frozen;
            for (const auto* value : group_values[group.first]) {
                if (value->seed && value->offset == 0 && value->bytes >= group_bytes[group.first]) {
                    frozen = value->prototype->storage(); break;
                }
            }
            if (!frozen) {
                Tensor buffer(std::make_shared<Buffer>(group_bytes[group.first]));
                auto* buffer_data = static_cast<char*>(buffer.mutable_data(group_bytes[group.first]));
                for (const auto* value : group_values[group.first]) if (value->seed && value->bytes)
                        std::memcpy(buffer_data + value->offset, value->prototype->raw_data(), value->bytes);
                auto readonly = std::make_shared<Buffer>(buffer.raw_data(), group_bytes[group.first],
                                                         buffer.storage()->buffer(), true);
                frozen = std::make_shared<Storage>(std::move(readonly));
            }
            if (options.immutable_weight_source) {
                for (const auto* value : group_values[group.first]) {
                    if (!value->seed || value->offset != 0 || value->bytes != group_bytes[group.first]) continue;
                    auto source = options.immutable_weight_source->runtime().GetTensor(value->name);
                    if (!source || !source->is_immutable() || !source->IsInitialized() ||
                        !source->storage()->read_only() || source->dims() != value->prototype->dims() ||
                        source->data_type() != value->prototype->data_type() ||
                        source->layout() != value->prototype->layout() ||
                        !HasSameQuantization(source->quantization(), value->prototype->quantization()) ||
                        source->memory_size() != value->bytes)
                        continue;
                    if (source->raw_data() != frozen->data() &&
                        std::memcmp(source->raw_data(), frozen->data(), value->bytes) != 0)
                        continue;
                    // Share the entire alias group, retaining independent COW.
                    auto buffer =
                        std::make_shared<Buffer>(source->raw_data(), value->bytes, source->storage()->buffer(), true);
                    frozen = std::make_shared<Storage>(std::move(buffer));
                    break;
                }
            }
            storage[group.first] = std::make_shared<Storage>(frozen->buffer());
        }
        for (const auto& value : values_) {
            auto found = bindings.find(value.prototype.get());
            std::shared_ptr<Tensor> tensor;
            if (found != bindings.end()) tensor = found->second;
            else if (external_prototypes.count(value.prototype.get())) {
                tensor = external_prototypes.at(value.prototype.get());
                bindings[value.prototype.get()] = tensor;
            } else {
                tensor = Descriptor(*value.prototype);
                auto& allocation = storage[value.group];
                if (!allocation) allocation = std::make_shared<Storage>(group_bytes[value.group]);
                tensor->ResetStorage(allocation, value.bytes, value.offset);
                if (value.seed && !immutable_groups[value.group] && value.bytes)
                    std::memcpy(tensor->mutable_data(value.bytes), value.prototype->raw_data(), value.bytes);
                bindings[value.prototype.get()] = tensor;
            }
            if (runtime.SetTensor(value.name, tensor) != 0) return BuildError("session value binding failed: " + value.name);
            // Names identify candidates, never establish equality: shape constants
            // can have the same name but different contents in another plan.
            if (options.immutable_weight_source && tensor->is_immutable()) {
                auto& source_runtime = options.immutable_weight_source->runtime();
                auto source = source_runtime.GetTensor(value.name);
                if (source && source->is_immutable() && source->IsInitialized() && source->storage()->read_only() &&
                    source->dims() == tensor->dims() && source->data_type() == tensor->data_type() &&
                    source->layout() == tensor->layout() &&
                    HasSameQuantization(source->quantization(), tensor->quantization()) &&
                    source->memory_size() == value.bytes &&
                    (source->raw_data() == tensor->raw_data() ||
                     std::memcmp(source->raw_data(), tensor->raw_data(), value.bytes) == 0)) {
                    if (cuda_inputs.count(value.name)) {
                        auto shared =
                            runtime.Backend().ShareImmutableWeight(source_runtime.Backend(), *source, *tensor);
                        if (!shared.ok()) return shared.WithContext("share weight=" + value.name);
                    }
                    usage.reused_weight_bytes += value.bytes;
                }
            }
            if (!produced.count(value.name) && !tensor->is_immutable()) session->inputs_.insert(value.name);
            if (tensor->is_immutable() || external_groups.count(value.group))
                runtime.Backend().MarkPersistent(*tensor, true);
        }
        auto installed = InstallRuntime(runtime, bindings);
        if (!installed.ok()) return installed;
        runtime.uses_memory_plan_ = session->reuses_memory_;
        auto status = runtime.FinalizeStatus();
        if (!status.ok()) return status;
        runtime.structure_locked_ = true;
        *result = std::move(session);
        return Status::Ok();
    } catch (const std::exception& error) { return BuildError(error.what()).WithContext("instantiate session"); }
}

Status ExecutionSession::Run() {
    try {
        auto scope = runtime_.Backend().EnterExecution();
        return runtime_.RunStatus();
    } catch (const std::exception& error) {
        return Status::Error(StatusCode::kExecutionFailed, error.what()).WithContext("execute session");
    }
}

Status ExecutionSession::BindInput(const std::string& name, std::shared_ptr<Tensor> tensor) {
    const auto bindings = runtime_.CheckTensorBindings();
    if (!bindings.ok()) return bindings;
    auto current = inputs_.count(name) ? runtime_.GetTensor(name) : nullptr;
    if (!current || !tensor || !tensor->IsInitialized() || current->dims() != tensor->dims() ||
        current->data_type() != tensor->data_type() || current->layout() != tensor->layout() ||
        !HasSameQuantization(current->quantization(), tensor->quantization()) ||
        tensor->memory_size() < current->memory_size())
        return Status::Error(StatusCode::kInvalidArgument, "incompatible session input: " + name);
    for (const auto& contract : runtime_.contracts_) {
        if (contract.tensor == current && contract.fixed_values && contract.bytes &&
            std::memcmp(tensor->raw_data(), contract.fixed_values->data(), contract.bytes) != 0)
            return Status::Error(StatusCode::kInvalidArgument, "compiled control values changed: " + name);
    }
    // Copy into the session's storage so existing reshape/slice views remain valid.
    if (current.get() != tensor.get() && current->memory_size())
        std::memmove(current->mutable_data(current->memory_size()), tensor->raw_data(), current->memory_size());
    runtime_.Backend().HostWritten(*current);
    return Status::Ok();
}

Status ExecutionSession::ReadOutput(const std::string& name, std::shared_ptr<Tensor>* output) {
    if (!output) return Status::Error(StatusCode::kInvalidArgument, "null output destination");
    const auto checked = runtime_.CheckStatus();
    if (!checked.ok()) return checked;
    auto tensor = outputs_.count(name) ? runtime_.GetTensor(name) : nullptr;
    if (!tensor) return Status::Error(StatusCode::kNotFound, "unknown output: " + name);
    const auto status = runtime_.Backend().ReadHost(*tensor);
    if (status.ok()) *output = std::move(tensor);
    return status;
}
}  // namespace feather
