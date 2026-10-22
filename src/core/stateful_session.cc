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

#include <cstring>

namespace feather {
namespace {
Status Validate(RuntimeGraph& graph, const StateBinding& binding, std::shared_ptr<Tensor>* input, std::shared_ptr<Tensor>* output) {
    *input = graph.GetTensor(binding.input_name); *output = graph.GetTensor(binding.output_name);
    if (!*input || !*output || !(*input)->IsInitialized() || !(*output)->IsInitialized() ||
        (*input)->is_immutable() || (*input)->data_type() != (*output)->data_type() || (*input)->layout() != (*output)->layout())
        return Status::Error(StatusCode::kInvalidArgument, "unavailable or incompatible state: " + binding.input_name);
    if (!HasSameQuantization((*input)->quantization(), (*output)->quantization()))
        return Status::Error(StatusCode::kInvalidArgument, "state quantization mismatch: " + binding.input_name);
    const auto& a = (*input)->dims().data(); const auto& b = (*output)->dims().data();
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
    if (!CheckedTensorBytes(a, (*input)->data_type(), &input_bytes) ||
        !CheckedTensorBytes(b, (*output)->data_type(), &output_bytes) ||
        (*input)->memory_size() < input_bytes || (*output)->memory_size() < output_bytes)
        return Status::Error(StatusCode::kInvalidArgument, "undersized state storage: " + binding.input_name);
    return Status::Ok();
}
}

Status ApplyStateBinding(RuntimeGraph& graph, const StateBinding& binding, DeviceType device) {
    std::shared_ptr<Tensor> input, output;
    auto status = Validate(graph, binding, &input, &output);
    if (!status.ok()) return status;
    auto scope = graph.Backend().Enter();
    const auto capabilities = graph.Backend().Capabilities(device);
    if (!capabilities.supported) return Status::Error(StatusCode::kUnsupported, "state backend unavailable");
    if (binding.cache_axis < 0) {
        // Identity is a view; swap its producer so the next execution cannot
        // accidentally write into the recurrent input through the relay.
        const auto* producer = graph.GetProducerNode(binding.output_name);
        if (producer && producer->op_type == "Identity" && producer->inputs.size() == 1) {
            auto source = graph.GetTensor(producer->inputs[0]);
            if (source && source->dims() == input->dims() && source->data_type() == input->data_type()) output = source;
        }
        if (input.get() == output.get() || input->storage() == output->storage()) return Status::Ok();
        if (!capabilities.host_memory) return graph.Backend().SwapState(*input, *output).WithContext(binding.input_name);
        if (graph.UsesMemoryPlan()) {
            if (input->memory_size()) std::memmove(input->mutable_data(input->memory_size()), output->raw_data(), input->memory_size());
        } else input->SwapStorage(*output);
        graph.Backend().HostWritten(*input); graph.Backend().HostWritten(*output);
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

Status StatefulSession::Bind(StateBinding binding) {
    std::shared_ptr<Tensor> input, output;
    auto status = Validate(graph_, binding, &input, &output);
    if (!status.ok()) return status;
    for (const auto& existing : bindings_) if (existing.input_name == binding.input_name)
        return Status::Error(StatusCode::kInvalidArgument, "duplicate state binding: " + binding.input_name);
    graph_.Backend().MarkPersistent(*input, true); graph_.Backend().MarkPersistent(*output, true);
    const auto* producer = graph_.GetProducerNode(binding.output_name);
    if (producer && producer->op_type == "Identity" && producer->inputs.size() == 1) {
        auto source = graph_.GetTensor(producer->inputs[0]);
        if (source) graph_.Backend().MarkPersistent(*source, true);
    }
    bindings_.push_back(std::move(binding)); return Status::Ok();
}
Status StatefulSession::Advance() {
    for (const auto& binding : bindings_) {
        auto status = ApplyStateBinding(graph_, binding, device_);
        if (!status.ok()) return status;
    }
    return Status::Ok();
}
Status StatefulSession::Reset() {
    for (const auto& binding : bindings_) {
        auto input = graph_.GetTensor(binding.input_name);
        if (!input) return Status::Error(StatusCode::kNotFound, "missing state: " + binding.input_name);
        std::memset(input->mutable_data(input->memory_size()), 0, input->memory_size());
        graph_.Backend().HostWritten(*input);
    }
    return Status::Ok();
}
}  // namespace feather
