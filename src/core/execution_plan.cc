#include "core/execution_plan.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <queue>

namespace feather {
namespace {
Status BuildError(const std::string& message) { return Status::Error(StatusCode::kBuildFailed, message); }

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
        return std::make_shared<Buffer>(const_cast<void*>(tensor.raw_data()), bytes,
                                       tensor.storage(), true);
    }
    auto copy = std::make_shared<Buffer>(bytes);
    if (bytes) std::memcpy(copy->data(), tensor.raw_data(), bytes);
    return std::make_shared<Buffer>(copy->data(), bytes, copy, true);
}
}

Status CompiledGraph::Compile(const StaticGraph& graph, std::shared_ptr<const CompiledGraph>* result) {
    if (!result) return Status::Error(StatusCode::kInvalidArgument, "null compiled graph destination");
    const auto checked = graph.ValidateStructureStatus();
    if (!checked.ok()) return checked;
    try {
        auto plan = std::shared_ptr<CompiledGraph>(new CompiledGraph);
        plan->outputs_ = graph.model().graph.outputs;
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
            const bool seed = graph.build_time_values().count(item.first) != 0 ||
                              graph.bound_tensor_names().count(item.first) != 0;
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
            plan->nodes_.push_back({node.name, node.op_type, node.inputs, node.outputs,
                                   kernel->device(), kernel->data_type(), kernel->layout(), temporary->SnapshotParameters(), kernel->InstanceFactory()});
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

Status CompiledGraph::CreateSession(std::unique_ptr<ExecutionSession>* result, const SessionOptions& options) const {
    if (!result) return Status::Error(StatusCode::kInvalidArgument, "null session destination");
    try {
        auto session = std::make_unique<ExecutionSession>(options.backend_factory ? options.backend_factory() : nullptr);
        auto& runtime = session->runtime_;
        auto scope = runtime.Backend().Enter();
        runtime.SetThreadMode(options.thread_mode); runtime.SetThreadCount(options.thread_count);
        runtime.SetOutputNames(outputs_);
        session->outputs_.insert(outputs_.begin(), outputs_.end());
        std::unordered_set<std::string> produced;
        for (const auto& node : nodes_) for (const auto& name : node.outputs) produced.insert(name);
        KernelBase::TensorBindings bindings;
        std::unordered_map<size_t, std::shared_ptr<Storage>> storage;
        bool can_reuse = true;
        for (const auto& node : nodes_) {
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
        if (session->reuses_memory_ && memory_plan_.arena_bytes) {
            auto arena = std::make_shared<Buffer>(memory_plan_.arena_bytes);
            for (const auto& allocation : memory_plan_.allocations) {
                auto buffer = std::make_shared<Buffer>(static_cast<char*>(arena->data()) + allocation.offset, allocation.bytes, arena);
                storage[allocation.group] = std::make_shared<Storage>(std::move(buffer));
            }
        }
        // Keep constant views on the same read-only allocation. This also
        // preserves mapped-weight sharing when the seed covers the full group.
        for (const auto& group : immutable_groups) {
            if (!group.second) continue;
            std::shared_ptr<Storage> frozen;
            for (const auto* value : group_values[group.first]) {
                if (value->seed && value->offset == 0 && value->bytes >= group_bytes[group.first]) {
                    frozen = value->prototype->storage(); break;
                }
            }
            if (!frozen) {
                auto buffer = std::make_shared<Buffer>(group_bytes[group.first]);
                for (const auto* value : group_values[group.first]) if (value->seed && value->bytes)
                    std::memcpy(static_cast<char*>(buffer->data()) + value->offset, value->prototype->raw_data(), value->bytes);
                auto readonly = std::make_shared<Buffer>(buffer->data(), buffer->size(), buffer, true);
                frozen = std::make_shared<Storage>(std::move(readonly));
            }
            storage[group.first] = std::make_shared<Storage>(frozen->buffer());
        }
        for (const auto& value : values_) {
            auto found = bindings.find(value.prototype.get());
            std::shared_ptr<Tensor> tensor;
            if (found != bindings.end()) tensor = found->second;
            else {
                tensor = Descriptor(*value.prototype);
                auto& allocation = storage[value.group];
                if (!allocation) allocation = std::make_shared<Storage>(group_bytes[value.group]);
                tensor->ResetStorage(allocation, value.bytes, value.offset);
                if (value.seed && !immutable_groups[value.group] && value.bytes)
                    std::memcpy(tensor->mutable_data(value.bytes), value.prototype->raw_data(), value.bytes);
                bindings[value.prototype.get()] = tensor;
            }
            if (!produced.count(value.name) && !tensor->is_immutable()) session->inputs_.insert(value.name);
            if (runtime.SetTensor(value.name, tensor) != 0) return BuildError("session value binding failed: " + value.name);
            runtime.contracts_.push_back({value.name, tensor, value.prototype, value.bytes, !produced.count(value.name)});
            if (tensor->is_immutable()) runtime.Backend().MarkPersistent(*tensor, true);
        }
        for (const auto& node : nodes_) {
            RuntimeNode executable;
            executable.name = node.name; executable.op_type = node.op_type;
            executable.inputs = node.inputs; executable.outputs = node.outputs; executable.kernel_device = node.device;
            if (node.device == DeviceType::X86 && !HostSupportsX86Kernels()) return BuildError("compiled x86 backend is unavailable: " + node.name);
            executable.kernel = node.factory ? node.factory() : nullptr;
            if (!executable.kernel || executable.kernel->device() != node.device)
                return BuildError("compiled backend is unavailable: " + node.name);
            auto status = node.bind(*executable.kernel, bindings);
            if (!status.ok()) return status.WithContext("instantiate node=" + node.name);
            status = executable.kernel->Supports();
            if (!status.ok()) return status.WithContext("instantiate node=" + node.name);
            runtime.AddNode(std::move(executable));
        }
        runtime.uses_memory_plan_ = session->reuses_memory_;
        auto status = runtime.FinalizeStatus();
        if (!status.ok()) return status;
        runtime.structure_locked_ = true;
        *result = std::move(session);
        return Status::Ok();
    } catch (const std::exception& error) { return BuildError(error.what()).WithContext("instantiate session"); }
}

Status ExecutionSession::BindInput(const std::string& name, std::shared_ptr<Tensor> tensor) {
    auto current = inputs_.count(name) ? runtime_.GetTensor(name) : nullptr;
    if (!current || !tensor || !tensor->IsInitialized() || current->dims() != tensor->dims() ||
        current->data_type() != tensor->data_type() || current->layout() != tensor->layout() ||
        !HasSameQuantization(current->quantization(), tensor->quantization()) ||
        tensor->memory_size() < current->memory_size())
        return Status::Error(StatusCode::kInvalidArgument, "incompatible session input: " + name);
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
