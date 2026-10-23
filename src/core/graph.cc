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

#include "core/graph.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <queue>
#include <utility>

#include "util/threading.h"
#include "util/timer.h"
#include "util/types.h"

namespace feather {

RuntimeGraph::RuntimeGraph() : RuntimeGraph(CreateBackendContext()) {}

RuntimeGraph::RuntimeGraph(std::unique_ptr<BackendContext> backend)
    : thread_mode_(RuntimeThreadMode::kParallelGraph),
      configured_thread_count_(DefaultThreadCount()), backend_(backend ? std::move(backend) : CreateBackendContext()) {}

RuntimeGraph::~RuntimeGraph() {
    structure_locked_ = false;
    try { Clear(); } catch (...) { /* Destructors must not propagate backend failures. */ }
}

std::string RuntimeNode::ProfileLabel() const {
    return "RuntimeNode::" + name + "[" + op_type + "]";
}

int32_t RuntimeNode::Run() {
    if (kernel == nullptr || !kernel->OwnsParam()) {
        return -1;
    }
    AutoTimer timer(ProfileLabel());
    return kernel->compute();
}

int32_t RuntimeGraph::SetTensor(const std::string& name, std::shared_ptr<Tensor> tensor) {
    if (structure_locked_) return -1;  // Session inputs must use BindInput.
    if (name.empty() || tensor == nullptr) {
        return -1;
    }
    const auto existing = tensors_.find(name);
    if (bindings_installed_ && existing != tensors_.end()) {
        if (!CheckTensorBindings().ok()) return -1;
        for (const auto& node : nodes_) {
            if (std::find(node.outputs.begin(), node.outputs.end(), name) != node.outputs.end()) return -1;
        }
        const auto& current = existing->second;
        if (current->is_immutable() || current->dims() != tensor->dims() ||
            current->data_type() != tensor->data_type() || current->layout() != tensor->layout() ||
            !HasSameQuantization(current->quantization(), tensor->quantization()) ||
            !tensor->IsInitialized() || tensor->memory_size() < current->memory_size()) return -1;
        if (current != tensor) {
            if (current->memory_size())
                std::memmove(current->mutable_data(current->memory_size()), tensor->raw_data(), current->memory_size());
            backend_->HostWritten(*current);
        }
    } else {
        if (finalized_) return -1;
        backend_->RegisterTensor(tensor);
        tensors_[name] = std::move(tensor);
    }
    RefreshOutputTensorPointers();
    return 0;
}

std::shared_ptr<Tensor> RuntimeGraph::GetTensor(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        return nullptr;
    }
    return it->second;
}

const RuntimeNode* RuntimeGraph::GetNode(const std::string& name) const {
    auto it = node_index_by_name_.find(name);
    if (it == node_index_by_name_.end()) {
        return nullptr;
    }
    return &nodes_[it->second];
}

const RuntimeNode* RuntimeGraph::GetProducerNode(const std::string& value_name) const {
    for (const auto& node : nodes_) {
        if (std::find(node.outputs.begin(), node.outputs.end(), value_name) != node.outputs.end()) return &node;
    }
    return nullptr;
}

void RuntimeGraph::SetProfilingEnabled(bool enabled) {
    profiling_enabled_ = enabled;
    if (!profiling_enabled_) {
        profile_summaries_.clear();
        profile_index_by_node_.clear();
    } else {
        // Qwen decode has more than a thousand runtime nodes. Reserve once so
        // turning profiling on does not allocate on the per-node hot path.
        profile_summaries_.reserve(nodes_.size());
        profile_index_by_node_.reserve(nodes_.size());
    }
}

void RuntimeGraph::RecordNodeProfile(const std::string& node_name, const std::string& op_type, double elapsed_ms) {
    if (!profiling_enabled_) {
        return;
    }

    std::lock_guard<std::mutex> lock(profile_mutex_);
    const auto index = profile_index_by_node_.find(node_name);
    if (index == profile_index_by_node_.end()) {
        profile_index_by_node_.emplace(node_name, profile_summaries_.size());
        profile_summaries_.push_back(
            RuntimeProfileSummary{node_name, op_type, 1, elapsed_ms, elapsed_ms, elapsed_ms, elapsed_ms});
        return;
    }

    auto& summary = profile_summaries_[index->second];
    summary.call_count += 1;
    summary.total_ms += elapsed_ms;
    summary.avg_ms = summary.total_ms / static_cast<double>(summary.call_count);
    summary.min_ms = std::min(summary.min_ms, elapsed_ms);
    summary.max_ms = std::max(summary.max_ms, elapsed_ms);
}

void RuntimeGraph::SetThreadMode(RuntimeThreadMode mode) {
    if (structure_locked_ && mode != thread_mode_) throw std::logic_error("session execution mode is fixed");
    if (uses_memory_plan_ && mode != RuntimeThreadMode::kSerialGraph)
        throw std::logic_error("create a new session to change a memory-planned graph's execution mode");
    if (thread_mode_ != mode) finalized_ = false;
    thread_mode_ = mode;
}

void RuntimeGraph::SetThreadCount(size_t count) {
    if (structure_locked_ && std::max<size_t>(1, count) != configured_thread_count_)
        throw std::logic_error("session thread count is fixed");
    if (configured_thread_count_ != std::max<size_t>(1, count)) finalized_ = false;
    configured_thread_count_ = std::max<size_t>(1, count);
}

void RuntimeGraph::SetOutputNames(std::vector<std::string> output_names) {
    if (structure_locked_) throw std::logic_error("session outputs are fixed");
    output_names_.clear();
    for (auto& name : output_names) {
        output_names_.insert(std::move(name));
    }
    RefreshOutputTensorPointers();
}

void RuntimeGraph::Clear() {
    if (structure_locked_) throw std::logic_error("session graph is immutable");
    auto context_scope = backend_->Enter();
    backend_->Clear();
    finalized_ = false;
    bindings_installed_ = false;
    uses_memory_plan_ = false;
    nodes_.clear();
    contracts_.clear();
    binding_contracts_.clear();
    effect_nodes_.clear();
    effect_inputs_.clear();
    effect_outputs_.clear();
    tensors_.clear();
    node_index_by_name_.clear();
    value_id_by_name_.clear();
    value_name_by_id_.clear();
    remaining_uses_.clear();
    output_names_.clear();
    output_tensor_ptrs_.clear();
    thread_pool_.reset();
    worker_count_ = 1;
    profiling_enabled_ = false;
    profile_summaries_.clear();
    profile_index_by_node_.clear();
}

int32_t RuntimeGraph::Check() const {
    return CheckStatus().ok() ? 0 : -1;
}

Status RuntimeGraph::CheckStatus() const {
    const auto bindings = CheckTensorBindings();
    if (!bindings.ok()) return bindings;
    for (const auto& value : contracts_) {
        const auto found = tensors_.find(value.name);
        if (found == tensors_.end() || found->second != value.tensor)
            return Status::Error(StatusCode::kInvalidArgument, "compiled tensor binding changed: " + value.name);
        bool shape_matches = value.tensor->dims() == value.descriptor->dims();
        if (value.bounded_leading_dimension) {
            const auto& actual = value.tensor->dims().data();
            const auto& capacity = value.descriptor->dims().data();
            shape_matches = actual.size() == capacity.size() && !actual.empty() && actual[0] >= 0 &&
                            actual[0] <= capacity[0] &&
                            std::equal(actual.begin() + 1, actual.end(), capacity.begin() + 1);
        }
        const bool has_capacity = value.bounded_leading_dimension
                                      ? value.tensor->storage() &&
                                            value.tensor->offset() <= value.tensor->storage()->size() &&
                                            value.bytes <= value.tensor->storage()->size() - value.tensor->offset()
                                      : value.tensor->memory_size() >= value.bytes;
        if (!shape_matches || value.tensor->data_type() != value.descriptor->data_type() ||
            value.tensor->layout() != value.descriptor->layout() || !value.tensor->IsInitialized() || !has_capacity ||
            (value.fixed_quantization &&
             !HasSameQuantization(value.tensor->quantization(), value.descriptor->quantization())))
            return Status::Error(StatusCode::kInvalidArgument, "compiled tensor contract changed: " + value.name);
        if (value.fixed_values && value.bytes &&
            std::memcmp(value.tensor->raw_data(), value.fixed_values->data(), value.bytes) != 0)
            return Status::Error(StatusCode::kInvalidArgument, "compiled control values changed: " + value.name);
    }
    for (const auto& node : nodes_) {
        if (node.kernel == nullptr || !node.kernel->OwnsParam()) {
            return Status::Error(StatusCode::kBuildFailed, "runtime node has no kernel: " + node.name);
        }
    }
    return Status::Ok();
}

Status RuntimeGraph::CheckTensorBindings() const {
    for (const auto& item : binding_contracts_) {
        const auto found = tensors_.find(item.first);
        const auto& expected = item.second;
        if (found == tensors_.end() || found->second != expected.tensor ||
            (expected.tensor->storage() ? expected.tensor->storage()->id() : 0) != expected.storage_id ||
            expected.tensor->offset() != expected.offset)
            return Status::Error(StatusCode::kInvalidArgument, "runtime storage view changed: " + item.first);
    }
    return Status::Ok();
}

void RuntimeGraph::CaptureTensorBindings() {
    for (const auto& item : tensors_) {
        const auto& tensor = item.second;
        binding_contracts_.try_emplace(item.first, TensorBindingContract{
            tensor, tensor->storage() ? tensor->storage()->id() : 0, tensor->offset()});
    }
    bindings_installed_ = true;
}

void RuntimeGraph::RecordStateStorageSwap(const Tensor& first, const Tensor& second) {
    // The state updater validates the complete batch before it can enter this
    // private boundary. Only the exact swapped Tensor objects change contracts.
    for (auto& item : binding_contracts_) {
        auto& binding = item.second;
        if (binding.tensor.get() != &first && binding.tensor.get() != &second) continue;
        binding.storage_id = binding.tensor->storage() ? binding.tensor->storage()->id() : 0;
        binding.offset = binding.tensor->offset();
    }
}

int32_t RuntimeGraph::Run() {
    return RunStatus().ok() ? 0 : -1;
}

Status RuntimeGraph::RunStatus() {
    if (!finalized_) return Status::Error(StatusCode::kBuildFailed, "runtime graph must be finalized before execution");
    try {
        auto context_scope = backend_->Enter();
        const auto check_status = CheckStatus();
        if (!check_status.ok()) {
            return check_status;
        }
        const auto effects = ValidateOperatorBindings(effect_nodes_, tensors_, &effect_inputs_);
        if (!effects.ok()) return effects.WithContext("execution preflight");
        for (const auto& node : nodes_)
            if (node.semantics->validate_runtime) {
                std::vector<std::shared_ptr<Tensor>> inputs;
                for (const auto& name : node.inputs) inputs.push_back(GetTensor(name));
                auto status = node.semantics->validate_runtime(inputs);
                if (!status.ok()) return status.WithContext("preflight node=" + node.name);
            }
        ResetPendingDependencies();
        ResetRemainingUses();
        execution_status_ = Status::Ok();
        const auto status = RunSerial();
        return status == 0 ? Status::Ok() : (execution_status_.ok()
            ? Status::Error(StatusCode::kExecutionFailed, "runtime graph execution failed") : execution_status_);
    } catch (const std::exception& error) {
        return Status::Error(StatusCode::kExecutionFailed, error.what()).WithContext("execute graph");
    }
}

void RuntimeGraph::AddNode(RuntimeNode node) {
    if (structure_locked_) throw std::logic_error("session graph is immutable");
    finalized_ = false;
    if (node.kernel) bindings_installed_ = true;
    if (!node.semantics) node.semantics = GetOperatorSemantics(node.op_type);
    if (node.input_ids.empty()) {
        node.input_ids.reserve(node.inputs.size());
        for (const auto& name : node.inputs) {
            if (name.empty()) continue;  // ONNX omitted optional inputs have no dependency.
            node.input_ids.push_back(GetOrCreateValueId(name));
        }
    }
    if (node.output_ids.empty()) {
        node.output_ids.reserve(node.outputs.size());
        for (const auto& name : node.outputs) {
            node.output_ids.push_back(GetOrCreateValueId(name));
        }
    }
    node_index_by_name_[node.name] = nodes_.size();
    nodes_.push_back(std::move(node));
}

ValueId RuntimeGraph::GetOrCreateValueId(const std::string& name) {
    auto it = value_id_by_name_.find(name);
    if (it != value_id_by_name_.end()) {
        return it->second;
    }
    if (structure_locked_) throw std::logic_error("session values are fixed");
    const auto id = static_cast<ValueId>(value_name_by_id_.size());
    value_id_by_name_.emplace(name, id);
    value_name_by_id_.push_back(name);
    return id;
}

ValueId RuntimeGraph::GetValueId(const std::string& name) const {
    auto it = value_id_by_name_.find(name);
    return it == value_id_by_name_.end() ? kInvalidValueId : it->second;
}

int32_t RuntimeGraph::Finalize() {
    return FinalizeStatus().ok() ? 0 : -1;
}

Status RuntimeGraph::FinalizeStatus() {
    if (structure_locked_) return Status::Error(StatusCode::kInvalidArgument, "session graph is already finalized");
    finalized_ = false;
    try {
        auto context_scope = backend_->Enter();
        const auto bindings = CheckTensorBindings();
        if (!bindings.ok()) return bindings;
        const auto status = BuildDependencies();
        if (status != 0) {
            thread_pool_.reset();
            worker_count_ = 1;
            return Status::Error(StatusCode::kBuildFailed, "runtime graph dependency construction failed");
        }

        effect_nodes_.clear();
        effect_inputs_.clear();
        effect_outputs_.assign(output_names_.begin(), output_names_.end());
        std::unordered_set<std::string> produced;
        for (const auto& node : nodes_) {
            // Topology-only placeholder nodes are accepted by Finalize for
            // scheduler setup; CheckStatus rejects them before execution.
            if (node.kernel) effect_nodes_.push_back({node.name, node.inputs, node.outputs, *node.semantics});
            produced.insert(node.outputs.begin(), node.outputs.end());
        }
        for (const auto& value : tensors_)
            if (!produced.count(value.first)) effect_inputs_.push_back(value.first);
        const auto effect_status = ValidateOperatorEffects(effect_nodes_, tensors_, &effect_inputs_, effect_outputs_,
                                                           thread_mode_ == RuntimeThreadMode::kSerialGraph);
        if (!effect_status.ok()) return effect_status;
        for (const auto& node : nodes_) if (node.kernel)
            for (auto slot : node.semantics->mutable_inputs)
                backend_->MarkPersistent(*GetTensor(node.inputs.at(slot)), true);

        // Prepare immutable-weight kernels before the first timed inference. This
        // keeps one-time packing/reformatting out of decode latency while leaving
        // kernels that do not need preparation unchanged.
        for (const auto& node : nodes_) {
            const auto caps = backend_->Capabilities(node.kernel_device);
            if (!caps.supported || (!caps.parallel_nodes && thread_mode_ != RuntimeThreadMode::kSerialGraph))
                return Status::Error(StatusCode::kUnsupported, "backend execution mode unavailable: " + node.name);
            if (node.kernel != nullptr && node.kernel->Prepare() != 0) {
                thread_pool_.reset();
                worker_count_ = 1;
                return Status::Error(StatusCode::kBuildFailed, "runtime kernel preparation failed");
            }
        }

        CaptureTensorBindings();
        finalized_ = true;
        if (thread_mode_ == RuntimeThreadMode::kSerialGraph) {
            thread_pool_.reset();
            worker_count_ = 1;
            return Status::Ok();
        }

        worker_count_ = std::max<size_t>(1, configured_thread_count_);
        worker_count_ = std::min(worker_count_, nodes_.size());
        if (worker_count_ <= 1) {
            thread_pool_.reset();
            worker_count_ = 1;
            return Status::Ok();
        }

        thread_pool_ = std::make_unique<ThreadPoolNv>(worker_count_);
        return Status::Ok();
    } catch (const std::exception& error) {
        finalized_ = false;
        thread_pool_.reset(); worker_count_ = 1;
        return Status::Error(StatusCode::kBuildFailed, error.what()).WithContext("finalize graph");
    }
}

size_t RuntimeGraph::NodeSize() const { return nodes_.size(); }

size_t RuntimeGraph::WorkerCount() const { return worker_count_; }

int32_t RuntimeGraph::BuildDependencies() {
    std::unordered_set<std::string> names;
    std::unordered_map<ValueId, size_t> producer_by_value;
    for (size_t i = 0; i < nodes_.size(); ++i) {
        auto& node = nodes_[i];
        node.predecessors.clear(); node.successors.clear(); node.pending_dependencies = 0;
        if (node.name.empty() || !names.insert(node.name).second) return -1;
        for (const auto value_id : node.output_ids) {
            if (!producer_by_value.emplace(value_id, i).second) return -1;
        }
    }
    for (size_t i = 0; i < nodes_.size(); ++i) {
        for (const auto value_id : nodes_[i].input_ids) {
            if (value_id >= value_name_by_id_.size()) return -1;
            const auto producer = producer_by_value.find(value_id);
            if (producer == producer_by_value.end()) {
                if (GetTensor(value_name_by_id_[value_id]) == nullptr) return -1;
                continue;
            }
            if (producer->second == i) return -1;
            nodes_[i].predecessors.push_back(producer->second);
            nodes_[producer->second].successors.push_back(i);
        }
    }
    std::vector<size_t> pending(nodes_.size());
    std::queue<size_t> ready;
    for (size_t i = 0; i < nodes_.size(); ++i) {
        pending[i] = nodes_[i].pending_dependencies = nodes_[i].predecessors.size();
        if (pending[i] == 0) ready.push(i);
    }
    size_t visited = 0;
    while (!ready.empty()) {
        const auto i = ready.front(); ready.pop(); ++visited;
        for (const auto user : nodes_[i].successors) if (--pending[user] == 0) ready.push(user);
    }
    return visited == nodes_.size() ? 0 : -1;
}

void RuntimeGraph::ResetPendingDependencies() {
    for (auto& node : nodes_) {
        node.pending_dependencies = node.predecessors.size();
    }
}

void RuntimeGraph::ResetRemainingUses() {
    remaining_uses_.clear();
    for (const auto& node : nodes_) {
        for (const auto value_id : node.input_ids) {
            ++remaining_uses_[value_id];
        }
    }
}

bool RuntimeGraph::ShouldKeepTensorDevice(const std::string& value_name, const std::shared_ptr<Tensor>& tensor) const {
    if (tensor == nullptr) {
        return true;
    }
    if (output_names_.find(value_name) != output_names_.end()) {
        return true;
    }
    if (output_tensor_ptrs_.find(tensor.get()) != output_tensor_ptrs_.end()) {
        return true;
    }
    if (backend_->IsPersistent(*tensor)) {
        return true;
    }
    return false;
}

void RuntimeGraph::RefreshOutputTensorPointers() {
    output_tensor_ptrs_.clear();
    for (const auto& output_name : output_names_) {
        const auto it = tensors_.find(output_name);
        if (it != tensors_.end() && it->second != nullptr) {
            output_tensor_ptrs_.insert(it->second.get());
        }
    }
}

void RuntimeGraph::ReleaseUnusedInputs(const RuntimeNode& node) {
    std::lock_guard<std::mutex> lock(uses_mutex_);
    for (const auto value_id : node.input_ids) {
        auto it = remaining_uses_.find(value_id);
        if (it == remaining_uses_.end()) {
            continue;
        }
        if (it->second == 0) {
            continue;
        }
        --it->second;
        if (it->second != 0) {
            continue;
        }
        if (value_id >= value_name_by_id_.size()) {
            continue;
        }
        const auto& value_name = value_name_by_id_[value_id];
        auto tensor = GetTensor(value_name);
        if (ShouldKeepTensorDevice(value_name, tensor)) {
            continue;
        }
        if (tensor != nullptr) {
            backend_->Release(*tensor);
        }
    }
}

int32_t RuntimeGraph::PrepareNodeForRun(const RuntimeNode& node) {
    if (!backend_->Capabilities(node.kernel_device).host_memory) return 0;
    for (const auto& name : node.inputs) {
        auto tensor = GetTensor(name);
        if (tensor && tensor->IsInitialized()) {
            const auto status = backend_->ReadHost(*tensor);
            if (!status.ok()) {
                std::lock_guard<std::mutex> lock(error_mutex_);
                execution_status_ = status.WithContext("prepare node=" + node.name + " input=" + name);
                return -1;
            }
        }
    }
    return 0;
}

int32_t RuntimeGraph::FinalizeNodeRun(const RuntimeNode& node, int32_t status) {
    if (status != 0 || !backend_->Capabilities(node.kernel_device).host_memory) return status;
    for (const auto& name : node.outputs) {
        auto tensor = GetTensor(name);
        if (tensor) backend_->HostWritten(*tensor);
    }
    if (node.semantics)
        for (const auto slot : node.semantics->mutable_inputs)
            if (auto tensor = GetTensor(node.inputs[slot])) backend_->HostWritten(*tensor);
    return status;
}

int32_t RuntimeGraph::RunNode(size_t index) {
    try {
        auto context_scope = backend_->Enter();
        if (index >= nodes_.size()) {
            return -1;
        }
        auto& node = nodes_[index];
        const auto begin = std::chrono::steady_clock::now();
        auto status = PrepareNodeForRun(node);
        if (status == 0) {
            try {
                const auto supported = node.kernel->Supports();
                if (!supported.ok()) {
                    std::lock_guard<std::mutex> lock(error_mutex_);
                    execution_status_ = supported.WithContext("node=" + node.name + " op=" + node.op_type);
                    status = -1;
                } else status = node.Run();
            } catch (const std::exception& error) {
                std::lock_guard<std::mutex> lock(error_mutex_);
                execution_status_ = Status::Error(StatusCode::kExecutionFailed,
                    "node=" + node.name + " op=" + node.op_type + ": " + error.what());
                status = -1;
            }
        }
        status = FinalizeNodeRun(node, status);
        if (profiling_enabled_ && status == 0) {
            status = !backend_->Capabilities(node.kernel_device).host_memory && !backend_->Synchronize().ok() ? -1 : 0;
        }
        if (status == 0) {
            ReleaseUnusedInputs(node);
        }
        if (status != 0) {
            std::lock_guard<std::mutex> lock(error_mutex_);
            if (execution_status_.ok()) execution_status_ = Status::Error(StatusCode::kExecutionFailed,
                "node=" + node.name + " op=" + node.op_type + " device=" + std::to_string(static_cast<int>(node.kernel_device)));
        }
        const auto end = std::chrono::steady_clock::now();
        RecordNodeProfile(node.name, node.op_type, std::chrono::duration<double, std::milli>(end - begin).count());
        return status;
    } catch (const std::exception& error) {
        std::lock_guard<std::mutex> lock(error_mutex_);
        const auto name = index < nodes_.size() ? nodes_[index].name : std::to_string(index);
        execution_status_ = Status::Error(StatusCode::kExecutionFailed, error.what()).WithContext("node=" + name);
        return -1;
    }
}

int32_t RuntimeGraph::RunSerial() {
    if (nodes_.size() < 2 || worker_count_ == 1 || thread_pool_ == nullptr) {
        // Memory lifetimes use the lowest ready node index at compile time.
        // Keep the same order even when independent branches become ready.
        std::priority_queue<size_t, std::vector<size_t>, std::greater<size_t>> ready;
        size_t completed = 0;
        for (size_t i = 0; i < nodes_.size(); ++i) {
            if (nodes_[i].pending_dependencies == 0) {
                ready.push(i);
            }
        }

        while (!ready.empty()) {
            const size_t index = ready.top();
            ready.pop();
            const auto status = RunNode(index);
            if (status != 0) {
                return status;
            }
            ++completed;
            for (const auto succ_index : nodes_[index].successors) {
                auto& succ = nodes_[succ_index];
                if (succ.pending_dependencies == 0) {
                    return -1;
                }
                --succ.pending_dependencies;
                if (succ.pending_dependencies == 0) {
                    ready.push(succ_index);
                }
            }
        }

        return completed == nodes_.size() ? 0 : -1;
    }

    std::mutex mutex;
    std::condition_variable cv;
    std::queue<size_t> ready;
    size_t completed = 0;
    size_t in_flight = 0;
    bool stop = false;
    int32_t error_status = 0;

    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (nodes_[i].pending_dependencies == 0) {
            ready.push(i);
        }
    }
    if (ready.empty() && !nodes_.empty()) {
        return -1;
    }

    auto run_node = [&](int /*tid*/, size_t index) -> int {
        const auto status = RunNode(index);
        {
            std::lock_guard<std::mutex> lock(mutex);
            --in_flight;
            if (status != 0) {
                error_status = status;
                stop = true;
            } else {
                ++completed;
                for (const auto succ_index : nodes_[index].successors) {
                    auto& succ = nodes_[succ_index];
                    if (succ.pending_dependencies == 0) {
                        error_status = -1;
                        stop = true;
                        break;
                    }
                    --succ.pending_dependencies;
                    if (succ.pending_dependencies == 0) {
                        ready.push(succ_index);
                    }
                }
                if (completed == nodes_.size()) {
                    stop = true;
                }
            }
        }
        cv.notify_all();
        return status;
    };

    while (completed < nodes_.size()) {
        std::vector<std::future<int>> futures;
        bool launched = false;
        {
            std::unique_lock<std::mutex> lock(mutex);
            if (error_status != 0) {
                break;
            }
            while (!ready.empty() && in_flight < worker_count_) {
                const size_t index = ready.front();
                ready.pop();
                ++in_flight;
                futures.push_back(thread_pool_->enqueue(run_node, index));
            }
            launched = !futures.empty();
            if (in_flight == 0 && ready.empty()) {
                break;
            }
            if (!launched) {
                cv.wait(lock, [&] { return stop || error_status != 0 || !ready.empty() || in_flight == 0; });
            }
        }

        for (auto& future : futures) {
            const auto status = future.get();
            if (status != 0) {
                error_status = status;
            }
        }
    }

    if (error_status != 0) {
        return error_status;
    }
    return completed == nodes_.size() ? 0 : -1;
}

}  // namespace feather
