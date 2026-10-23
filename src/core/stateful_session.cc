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

#include "core/stateful_session.h"

#include <cstdint>
#include <cstring>

namespace feather {
// Defined only in the state implementation: callers cannot bless arbitrary
// public Tensor storage replacement as a new compiled allocation contract.
class StateBindingAccess {
 public:
    static Status Check(const RuntimeGraph& graph) { return graph.CheckTensorBindings(); }
    static void Swapped(RuntimeGraph& graph, const Tensor& first, const Tensor& second) {
        graph.RecordStateStorageSwap(first, second);
    }
};
namespace {
Status ValidateTensors(const StateBinding& binding, const std::shared_ptr<Tensor>& input,
                       const std::shared_ptr<Tensor>& output) {
    if (!input || !output || !input->IsInitialized() || !output->IsInitialized() || input->is_immutable() ||
        input->data_type() != output->data_type() ||
        NormalizeDataLayout(input->layout()) != NormalizeDataLayout(output->layout()))
        return Status::Error(StatusCode::kInvalidArgument, "unavailable or incompatible state: " + binding.input_name);
    if (!HasSameQuantization(input->quantization(), output->quantization()))
        return Status::Error(StatusCode::kInvalidArgument, "state quantization mismatch: " + binding.input_name);
    const auto& a = input->dims().data();
    const auto& b = output->dims().data();
    if (a.size() != b.size()) return Status::Error(StatusCode::kShapeMismatch, "state rank mismatch: " + binding.input_name);
    if (binding.cache_axis < 0) {
        if (a != b) return Status::Error(StatusCode::kShapeMismatch, "state shape mismatch: " + binding.input_name);
    } else {
        const auto axis = static_cast<size_t>(binding.cache_axis);
        if (axis >= a.size() || a[axis] <= 1 || b[axis] != 1)
            return Status::Error(StatusCode::kShapeMismatch, "invalid state append axis: " + binding.input_name);
        for (size_t i = 0; i < a.size(); ++i) if (i != axis && a[i] != b[i])
            return Status::Error(StatusCode::kShapeMismatch, "state append shape mismatch: " + binding.input_name);
    }
    size_t input_bytes = 0, output_bytes = 0;
    if (!CheckedTensorBytes(a, input->data_type(), &input_bytes) ||
        !CheckedTensorBytes(b, output->data_type(), &output_bytes) || input->memory_size() < input_bytes ||
        output->memory_size() < output_bytes)
        return Status::Error(StatusCode::kInvalidArgument, "undersized state storage: " + binding.input_name);
    return Status::Ok();
}

Status Validate(RuntimeGraph& graph, const StateBinding& binding, std::shared_ptr<Tensor>* input,
                std::shared_ptr<Tensor>* output) {
    *input = graph.GetTensor(binding.input_name);
    *output = graph.GetTensor(binding.output_name);
    return ValidateTensors(binding, *input, *output);
}

std::shared_ptr<Tensor> StateOutput(RuntimeGraph& graph, const StateBinding& binding,
                                    const std::shared_ptr<Tensor>& input, std::shared_ptr<Tensor> output) {
    if (binding.cache_axis < 0) {
        // Legacy swapping follows the producer to keep its next write away
        // from the recurrent input. Stable copies use the retained output view;
        // the producer's device cache may already have been released.
        const auto* producer = graph.GetProducerNode(binding.output_name);
        if (!binding.stable_storage && producer && producer->semantics) {
            auto source = graph.GetTensor(
                WholeTensorAliasSource(*producer->semantics, producer->inputs, producer->outputs, binding.output_name));
            if (source && source->dims() == input->dims() && source->data_type() == input->data_type()) output = source;
        }
    }
    return output;
}

bool Overlaps(const Tensor& a, const Tensor& b) {
    size_t a_bytes = 0, b_bytes = 0;
    (void)CheckedTensorBytes(a.dims().data(), a.data_type(), &a_bytes);
    (void)CheckedTensorBytes(b.dims().data(), b.data_type(), &b_bytes);
    if (!a_bytes || !b_bytes) return false;
    if (a.storage() == b.storage())
        return a.offset() <= b.offset() ? b.offset() - a.offset() < a_bytes : a.offset() - b.offset() < b_bytes;
    const auto first = reinterpret_cast<uintptr_t>(a.raw_data());
    const auto second = reinterpret_cast<uintptr_t>(b.raw_data());
    return first <= second ? second - first < a_bytes : first - second < b_bytes;
}

Status ApplyValidatedState(RuntimeGraph& graph, const StateBinding& binding, DeviceType device,
                           const std::shared_ptr<Tensor>& input, const std::shared_ptr<Tensor>& output) {
    const auto capabilities = graph.Backend().Capabilities(device);
    if (binding.cache_axis < 0) {
        if (input.get() == output.get() ||
            (input->storage() == output->storage() && input->offset() == output->offset()))
            return Status::Ok();
        if (!capabilities.host_memory) {
            if (binding.stable_storage)
                return graph.Backend().CopyState(*input, *output).WithContext(binding.input_name);
            return graph.Backend().SwapState(*input, *output).WithContext(binding.input_name);
        }
        if (graph.UsesMemoryPlan() || binding.stable_storage) {
            size_t bytes = 0;
            (void)CheckedTensorBytes(input->dims().data(), input->data_type(), &bytes);
            if (bytes) std::memmove(input->mutable_data(input->memory_size()), output->raw_data(), bytes);
        } else {
            input->SwapStorage(*output);
            StateBindingAccess::Swapped(graph, *input, *output);
            graph.Backend().HostWritten(*output);
        }
        graph.Backend().HostWritten(*input);
        return Status::Ok();
    }
    if (!capabilities.host_memory)
        return graph.Backend().AppendState(*input, *output, binding.cache_axis).WithContext(binding.input_name);
    const auto& dims = input->dims().data(); const auto axis = static_cast<size_t>(binding.cache_axis);
    size_t outer = 1, inner = 1;
    for (size_t i = 0; i < axis; ++i) outer *= static_cast<size_t>(dims[i]);
    for (size_t i = axis + 1; i < dims.size(); ++i) inner *= static_cast<size_t>(dims[i]);
    const auto element = DataTypeBytes(input->data_type());
    if (!element) return Status::Error(StatusCode::kUnsupported, "packed state append is unsupported");
    const size_t block = inner * element, slots = static_cast<size_t>(dims[axis]);
    auto* destination = static_cast<char*>(input->mutable_data(input->memory_size()));
    const auto* source = static_cast<const char*>(output->raw_data());
    std::vector<char> snapshot;
    if (input->storage() == output->storage()) {
        snapshot.assign(source, source + output->memory_size()); source = snapshot.data();
    }
    for (size_t i = 0; i < outer; ++i) {
        auto* cache = destination + i * slots * block;
        std::memmove(cache, cache + block, (slots - 1) * block);
        std::memcpy(cache + (slots - 1) * block, source + i * block, block);
    }
    graph.Backend().HostWritten(*input);
    return Status::Ok();
}
Status ApplyStateBindings(RuntimeGraph& graph, const std::vector<StateBinding>& bindings, DeviceType device) {
    const auto contract = StateBindingAccess::Check(graph);
    if (!contract.ok()) return contract;
    struct Update {
        StateBinding binding;
        std::shared_ptr<Tensor> input, output;
        bool snapshot{false};
    };
    std::vector<Update> updates;
    for (const auto& binding : bindings) {
        Update update{binding, nullptr, nullptr};
        auto status = Validate(graph, binding, &update.input, &update.output);
        if (!status.ok()) return status;
        update.output = StateOutput(graph, binding, update.input, update.output);
        status = ValidateTensors(binding, update.input, update.output);
        if (!status.ok()) return status;
        // Constants may supply recurrent state, but exchanging their storage
        // would mutate immutable weights and break retained alias views.
        if (update.output->is_immutable()) update.binding.stable_storage = true;
        updates.push_back(std::move(update));
    }
    if (updates.empty()) return Status::Ok();
    auto scope = graph.Backend().Enter();
    const auto capabilities = graph.Backend().Capabilities(device);
    if (!capabilities.supported) return Status::Error(StatusCode::kUnsupported, "state backend unavailable");
    if (capabilities.host_memory)
        for (const auto& update : updates)
            if (update.binding.cache_axis >= 0 && !DataTypeBytes(update.input->data_type()))
                return Status::Error(StatusCode::kUnsupported, "packed state append is unsupported");
    std::unordered_map<const Tensor*, std::shared_ptr<const Storage>> device_storage;
    if (!capabilities.host_memory) {
        for (const auto& update : updates) {
            device_storage.emplace(update.input.get(), graph.Backend().DeviceStorage(*update.input));
            device_storage.emplace(update.output.get(), graph.Backend().DeviceStorage(*update.output));
        }
    }
    const auto overlaps = [&](const Tensor& a, const Tensor& b) {
        if (Overlaps(a, b)) return true;
        if (capabilities.host_memory) return false;
        const auto& first = device_storage.at(&a);
        const auto& second = device_storage.at(&b);
        return first && first == second;
    };
    for (size_t i = 0; i < updates.size(); ++i) {
        auto& update = updates[i];
        if (overlaps(*update.input, *update.output) && (capabilities.host_memory || update.binding.cache_axis >= 0)) {
            update.snapshot = true;
            update.binding.stable_storage = true;
        }
        for (size_t j = 0; j < updates.size(); ++j) {
            if (i == j) continue;
            if (overlaps(*update.input, *updates[j].input))
                return Status::Error(StatusCode::kInvalidArgument,
                                     "overlapping state destinations: " + update.binding.input_name);
            if (overlaps(*update.output, *updates[j].input) || overlaps(*update.output, *updates[j].output)) {
                update.snapshot = true;
                update.binding.stable_storage = true;
                updates[j].binding.stable_storage = true;
            }
        }
    }
    // A backend swap may exchange cached device allocations, so a host copy
    // cannot snapshot device aliases. Reject that unsupported batch up front.
    if (!capabilities.host_memory) {
        for (const auto& update : updates)
            if (update.snapshot &&
                !(update.binding.cache_axis < 0 && update.input->storage() == update.output->storage() &&
                  update.input->offset() == update.output->offset()))
                return Status::Error(StatusCode::kUnsupported,
                                     "overlapping device state updates: " + update.binding.input_name);
    } else {
        for (auto& update : updates)
            if (update.snapshot) {
                size_t bytes = 0;
                (void)CheckedTensorBytes(update.output->dims().data(), update.output->data_type(), &bytes);
                auto snapshot = std::make_shared<Tensor>(bytes);
                snapshot->Resize(update.output->dims());
                snapshot->set_data_type(update.output->data_type());
                snapshot->set_layout(update.output->layout());
                snapshot->set_quantization(update.output->quantization());
                if (bytes) std::memcpy(snapshot->mutable_data(bytes), update.output->raw_data(), bytes);
                update.output = std::move(snapshot);
            }
    }
    for (const auto& update : updates) {
        const auto status = ApplyValidatedState(graph, update.binding, device, update.input, update.output);
        if (!status.ok()) return status;
    }
    return Status::Ok();
}

}  // namespace

Status ApplyStateBinding(RuntimeGraph& graph, const StateBinding& binding, DeviceType device) {
    return ApplyStateBindings(graph, {binding}, device);
}

Status StatefulSession::Bind(StateBinding binding) {
    std::shared_ptr<Tensor> input, output;
    auto status = Validate(graph_, binding, &input, &output);
    if (!status.ok()) return status;
    for (const auto& existing : bindings_) if (existing.input_name == binding.input_name)
        return Status::Error(StatusCode::kInvalidArgument, "duplicate state binding: " + binding.input_name);
    graph_.Backend().MarkPersistent(*input, true); graph_.Backend().MarkPersistent(*output, true);
    const auto* producer = graph_.GetProducerNode(binding.output_name);
    if (producer && producer->semantics) {
        auto source = graph_.GetTensor(
            WholeTensorAliasSource(*producer->semantics, producer->inputs, producer->outputs, binding.output_name));
        if (source) graph_.Backend().MarkPersistent(*source, true);
    }
    bindings_.push_back(std::move(binding)); return Status::Ok();
}
Status StatefulSession::Advance() { return ApplyStateBindings(graph_, bindings_, device_); }
Status StatefulSession::Reset() {
    const auto contract = StateBindingAccess::Check(graph_);
    if (!contract.ok()) return contract;
    for (const auto& binding : bindings_) {
        auto input = graph_.GetTensor(binding.input_name);
        if (!input) return Status::Error(StatusCode::kNotFound, "missing state: " + binding.input_name);
        std::memset(input->mutable_data(input->memory_size()), 0, input->memory_size());
        graph_.Backend().HostWritten(*input);
    }
    return Status::Ok();
}
}  // namespace feather
