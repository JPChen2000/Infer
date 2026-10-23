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

#include "demo/qwen_runner.h"
#include "demo/internal/qwen_execution_plan.h"
#include "demo/qwen_context.h"
#include "demo/qwen_sequence.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <numeric>
#include <sstream>
#include <unordered_set>

#include "util/bf16.h"
#include "util/fp16.h"
#ifdef FEATHER_WITH_CUDA
#include "src/kernel/cuda/runtime.h"
#endif

namespace feather {
namespace demo {

namespace {

DeviceType ResolveBackendDevice(QwenBackend backend) {
    switch (backend) {
        case QwenBackend::kCuda:
            return DeviceType::CUDA;
        case QwenBackend::kX86:
            return DeviceType::X86;
        case QwenBackend::kHost:
        case QwenBackend::kCommon:
        default:
            // Qwen's BF16 reference path is implemented by Common kernels.
            return DeviceType::COMMON;
    }
}

int64_t Numel(const std::vector<int64_t>& dims) {
    int64_t result = 1;
    for (const auto dim : dims) {
        if (dim <= 0 || result > std::numeric_limits<int64_t>::max() / dim) {
            return 0;
        }
        result *= dim;
    }
    return result;
}

size_t TensorBytes(const Tensor& tensor) {
    const auto element_bytes = DataTypeBytes(tensor.data_type());
    if (element_bytes == 0 || tensor.numel() <= 0) {
        return 0;
    }
    return static_cast<size_t>(tensor.numel()) * element_bytes;
}

#ifdef FEATHER_WITH_CUDA
int SyncTensorFromCuda(const std::shared_ptr<Tensor>& tensor) {
    if (tensor == nullptr || !tensor->IsInitialized()) {
        return -1;
    }
    const size_t bytes = TensorBytes(*tensor);
    return bytes == 0 ? -1 : kernel::cuda_detail::SyncTensorToHost(tensor.get(), bytes, tensor->raw_data());
}
#endif

std::shared_ptr<Tensor> CreateZeroTensor(const model::TensorDesc& desc) {
    const int64_t numel = Numel(desc.dims);
    const size_t element_bytes = DataTypeBytes(desc.data_type);
    if (numel <= 0 || element_bytes == 0 || static_cast<uint64_t>(numel) >
                                             std::numeric_limits<size_t>::max() / element_bytes) {
        return nullptr;
    }
    const size_t bytes = static_cast<size_t>(numel) * element_bytes;
    auto tensor = std::make_shared<Tensor>(bytes);
    tensor->Resize(desc.dims);
    tensor->set_data_type(desc.data_type);
    tensor->set_layout(desc.layout);
    std::memset(tensor->mutable_data(bytes), 0, bytes);
    return tensor;
}

bool HasName(const std::vector<std::string>& names, const std::string& name) {
    return std::find(names.begin(), names.end(), name) != names.end();
}

bool IsBf16NaN(uint16_t bits) {
    return (bits & 0x7f80u) == 0x7f80u && (bits & 0x007fu) != 0;
}

bool IsFp16NaN(uint16_t bits) {
    return (bits & 0x7c00u) == 0x7c00u && (bits & 0x03ffu) != 0;
}

template <typename IsNaN, typename IsZero, typename SortableKey>
int64_t SelectGreedy16(const uint16_t* values, int64_t count, IsNaN is_nan, IsZero is_zero,
                       SortableKey sortable_key) {
    int64_t best = -1;
    uint16_t best_bits = 0;
    for (int64_t index = 0; index < count; ++index) {
        const uint16_t bits = values[index];
        if (is_nan(bits)) {
            continue;
        }
        if (best < 0) {
            best = index;
            best_bits = bits;
            continue;
        }
        // IEEE treats both signed zeroes as equal. Keep the first index just
        // like the scalar `value > best_value` comparison does.
        if (is_zero(bits) && is_zero(best_bits)) {
            continue;
        }
        if (sortable_key(bits) > sortable_key(best_bits)) {
            best = index;
            best_bits = bits;
        }
    }
    return best < 0 ? 0 : best;
}

int64_t SelectGreedyFp32(const float* values, int64_t count) {
    int64_t best = -1;
    float best_value = 0.0f;
    for (int64_t index = 0; index < count; ++index) {
        const float value = values[index];
        if (std::isnan(value)) {
            continue;
        }
        if (best < 0 || value > best_value) {
            best = index;
            best_value = value;
        }
    }
    return best < 0 ? 0 : best;
}

int64_t SelectGreedyFp16(const uint16_t* values, int64_t count) {
    return SelectGreedy16(values, count, IsFp16NaN, [](uint16_t bits) { return (bits & 0x7fffu) == 0; },
                          [](uint16_t bits) {
                              return (bits & 0x8000u) != 0 ? static_cast<uint16_t>(~bits)
                                                           : static_cast<uint16_t>(bits ^ 0x8000u);
                          });
}

int64_t SelectGreedyBf16(const BFloat16* values, int64_t count) {
    return SelectGreedy16(
        reinterpret_cast<const uint16_t*>(values), count, IsBf16NaN,
        [](uint16_t bits) { return (bits & 0x7fffu) == 0; }, [](uint16_t bits) {
            return (bits & 0x8000u) != 0 ? static_cast<uint16_t>(~bits)
                                         : static_cast<uint16_t>(bits ^ 0x8000u);
        });
}

}  // namespace

using internal::QwenExecutionPlan;
using internal::QwenDecodeState;
using internal::QwenModelResources;
using internal::QwenModelSource;

QwenExecutionPlan::~QwenExecutionPlan() {
#ifdef FEATHER_WITH_CUDA
    kernel::cuda_detail::ExecutionContextScope context_scope(Runtime().CudaContext());
    ResetCudaGraphs();
#endif
}

void QwenExecutionPlan::SetRuntimeProfilingEnabled(bool enabled) {
#ifdef FEATHER_WITH_CUDA
    kernel::cuda_detail::ExecutionContextScope context_scope(Runtime().CudaContext());
#endif
#ifdef FEATHER_WITH_CUDA
    ResetCudaGraphs();
#endif
    Runtime().SetProfilingEnabled(enabled);
}

const model::ValueDesc* QwenExecutionPlan::FindValueDesc(const std::string& name) const {
    const auto& values = model_.graph.values;
    const auto it = std::find_if(values.begin(), values.end(), [&name](const model::ValueDesc& value) {
        return value.tensor.name == name;
    });
    return it == values.end() ? nullptr : &*it;
}

int32_t QwenExecutionPlan::ValidatePrefillCompanion(const QwenExecutionPlan& companion) {
    const auto fail = [this](const std::string& reason) {
        last_error_ = "Incompatible Qwen prefill companion: " + reason;
        return -1;
    };
    if (!optimized_execution_ || !companion.optimized_execution_ || execution_sequence_ != 1 ||
        companion.execution_sequence_ <= 1 || companion.execution_sequence_ > max_context_ ||
        weights_id_ != companion.weights_id_ || max_context_ != companion.max_context_ ||
        model_context_limit_ != companion.model_context_limit_ || backend_device_ != companion.backend_device_)
        return fail("execution version, source weights, backend or context limits differ");
    const std::unordered_set<std::string> names(state_input_names_.begin(), state_input_names_.end());
    const std::unordered_set<std::string> other_names(companion.state_input_names_.begin(), companion.state_input_names_.end());
    const std::unordered_set<std::string> caches(inplace_cache_names_.begin(), inplace_cache_names_.end());
    const std::unordered_set<std::string> other_caches(companion.inplace_cache_names_.begin(), companion.inplace_cache_names_.end());
    if (names != other_names || caches != other_caches || states_.size() != companion.states_.size())
        return fail("state input names or in-place caches differ");
    for (const auto& name : state_input_names_) {
        const auto* a = FindValueDesc(name);
        const auto* b = companion.FindValueDesc(name);
        if (!a || !b || a->tensor.data_type != b->tensor.data_type || a->tensor.dims != b->tensor.dims ||
            a->tensor.layout != b->tensor.layout || !HasSameQuantization(a->tensor.quantization, b->tensor.quantization))
            return fail("state descriptor differs for " + name);
    }
    for (const auto& binding : states_) {
        const auto other = std::find_if(companion.states_.begin(), companion.states_.end(), [&](const StateBinding& value) {
            return value.input_name == binding.input_name;
        });
        if (other == companion.states_.end() || other->output_name != binding.output_name ||
            other->cache_axis != binding.cache_axis) return fail("state update differs for " + binding.input_name);
    }
    const auto* logits = FindValueDesc(logits_output_name_);
    const auto* other_logits = companion.FindValueDesc(companion.logits_output_name_);
    if (!logits || !other_logits || logits->tensor.dims != other_logits->tensor.dims ||
        logits->tensor.data_type != other_logits->tensor.data_type) return fail("logits descriptors differ");
    return 0;
}

int32_t QwenExecutionPlan::PrepareExecutableGraph(const ExecutionSession* weight_source) {
    if (static_graph_.Build() != 0 || static_graph_.ApplyPasses() != 0) return -1;
    // Host shape inference represents view operators by aliasing Tensor
    // storage. External request state must remain a single stable allocation;
    // produced views therefore need their own destination before freezing.
    // Keep Tensor objects unchanged so the already-bound kernel parameters
    // remain valid. Their kernels copy into these destinations at execution.
    for (const auto& name : state_input_names_) {
        const auto state = static_graph_.GetTensor(name);
        for (const auto& item : static_graph_.tensors()) {
            const auto& view = item.second;
            if (!state || !view || view == state || view->storage() != state->storage() ||
                HasName(model_.graph.inputs, item.first)) continue;
            size_t bytes = 0;
            if (!view->IsInitialized() ||
                !CheckedTensorBytes(view->dims().data(), view->data_type(), &bytes) || !bytes ||
                view->memory_size() < bytes || view->offset() > view->storage()->size() ||
                bytes > view->storage()->size() - view->offset()) {
                last_error_ = "Invalid Qwen state view: " + item.first;
                return -1;
            }
            auto snapshot = std::make_shared<Tensor>(bytes);
            std::memcpy(snapshot->mutable_data(bytes), view->raw_data(), bytes);
            view->ResetStorage(snapshot->storage(), bytes);
        }
    }
    std::shared_ptr<const CompiledGraph> compiled;
    std::unordered_set<std::string> external_states;
    external_states.insert(state_input_names_.begin(), state_input_names_.end());
    auto status = CompiledGraph::Compile(static_graph_, &compiled, external_states);
    if (!status.ok()) { last_error_ = status.message(); return -1; }
    std::unique_ptr<ExecutionSession> session;
    SessionOptions options;
    options.immutable_weight_source = weight_source;
    options.fp8_weight_cache_limit = load_options_.fp8_weight_cache_bytes;
    {
        for (const auto& name : state_input_names_)
            options.mutable_state[name] = state_.tensors.at(name);
#ifdef FEATHER_WITH_CUDA
        if (backend_device_ == DeviceType::CUDA && cuda_context_) {
            auto context = cuda_context_;
            options.backend_factory = [context] { return CreateBackendContext(context); };
        }
#endif
    }
    status = compiled->CreateSession(&session, options);
    if (!status.ok()) { last_error_ = status.message(); return -1; }
    session->runtime().SetProfilingEnabled(Runtime().ProfilingEnabled());
    compiled_graph_ = std::move(compiled);
    session_ = std::move(session);
    return 0;
}

int32_t QwenExecutionPlan::Reset() {
#ifdef FEATHER_WITH_CUDA
    kernel::cuda_detail::ExecutionContextScope context_scope(Runtime().CudaContext());
#endif
#ifdef FEATHER_WITH_CUDA
    ResetCudaGraphs();
#endif
    if (Runtime().NodeSize() == 0) {
        last_error_ = "Qwen runner is not loaded";
        return -1;
    }
#ifdef FEATHER_WITH_CUDA
    if (backend_device_ == DeviceType::CUDA) {
        for (const auto& name : inplace_cache_names_) {
            auto cache = Runtime().GetTensor(name);
            if (cache) kernel::cuda_detail::MarkTensorDevicePersistent(cache.get(), true);
        }
        for (const auto& binding : states_) {
            auto input = Runtime().GetTensor(binding.input_name);
            auto output = Runtime().GetTensor(binding.output_name);
            if (input != nullptr) {
                kernel::cuda_detail::MarkTensorDevicePersistent(input.get(), true);
            }
            if (output != nullptr) {
                kernel::cuda_detail::MarkTensorDevicePersistent(output.get(), true);
            }
            const auto producer_name = static_graph_.GetProducer(binding.output_name);
            const auto* producer = Runtime().GetNode(producer_name);
            if (producer && producer->semantics) {
                auto source = Runtime().GetTensor(WholeTensorAliasSource(
                    *producer->semantics, producer->inputs, producer->outputs, binding.output_name));
                if (source) kernel::cuda_detail::MarkTensorDevicePersistent(source.get(), true);
            }
        }
    }
#endif
    for (const auto& input_name : model_.graph.inputs) {
        auto tensor = Runtime().GetTensor(input_name);
        if (tensor == nullptr || !tensor->IsInitialized()) {
            last_error_ = "Qwen runtime input is missing: " + input_name;
            return -1;
        }
        const size_t bytes = TensorBytes(*tensor);
        if (bytes == 0) {
            last_error_ = "Qwen runtime input has invalid size: " + input_name;
            return -1;
        }
        std::memset(tensor->mutable_data(bytes), 0, bytes);
#ifdef FEATHER_WITH_CUDA
        if (backend_device_ == DeviceType::CUDA) {
            kernel::cuda_detail::InvalidateTensorDevice(tensor.get());
        }
#endif
    }
    state_.tokens_processed = 0;
    steps_executed_ = 0;
    last_error_.clear();
    last_step_summary_.clear();
    state_.invalid = false;
    return 0;
}

int32_t QwenExecutionPlan::SetRuntimeInputs(const int64_t* values, int64_t count) {
#ifdef FEATHER_WITH_CUDA
    kernel::cuda_detail::ExecutionContextScope context_scope(Runtime().CudaContext());
#endif
    auto token_ids = Runtime().GetTensor(token_input_name_);
    auto position = Runtime().GetTensor(position_input_name_);
    auto mask = Runtime().GetTensor(attention_mask_input_name_);
    if (values == nullptr || count != execution_sequence_ || token_ids == nullptr || position == nullptr ||
        mask == nullptr || token_ids->numel() != count || position->numel() != 1 ||
        token_ids->data_type() != DataType::INT64 || position->data_type() != DataType::INT64 ||
        mask->data_type() != DataType::BF16 || mask->numel() != max_context_) {
        last_error_ = "Qwen runtime decode inputs have unexpected shape or type";
        return -1;
    }
    std::copy(values, values + count, token_ids->mutable_data<int64_t>());
    position->mutable_data<int64_t>()[0] = state_.tokens_processed;
    if (!optimized_execution_) {
        auto* mask_values = mask->mutable_data<BFloat16>();
        const uint16_t negative_infinity = FloatToBFloat16(-std::numeric_limits<float>::infinity());
        const uint16_t zero = FloatToBFloat16(0.0f);
        const int64_t valid_positions = std::min(state_.tokens_processed, max_context_ - 1) + 1;
        for (int64_t index = 0; index < max_context_; ++index) {
            mask_values[index].bits = index >= max_context_ - valid_positions ? zero : negative_infinity;
        }
    }
#ifdef FEATHER_WITH_CUDA
    if (backend_device_ == DeviceType::CUDA) {
        kernel::cuda_detail::InvalidateTensorDevice(token_ids.get());
        kernel::cuda_detail::InvalidateTensorDevice(position.get());
        if (!optimized_execution_) kernel::cuda_detail::InvalidateTensorDevice(mask.get());
    }
#endif
    return 0;
}

int32_t QwenExecutionPlan::CopyState(const StateBinding& binding) {
    const auto status = ApplyStateBinding(Runtime(), binding, backend_device_);
    if (!status.ok()) last_error_ = status.message();
    return status.ok() ? 0 : -1;
}

int32_t QwenExecutionPlan::UpdateStates() {
#ifdef FEATHER_WITH_CUDA
    kernel::cuda_detail::ExecutionContextScope context_scope(Runtime().CudaContext());
#endif
    for (const auto& binding : states_) {
        if (CopyState(binding) != 0) {
            return -1;
        }
    }
    return 0;
}

int64_t QwenExecutionPlan::SelectGreedyToken() const {
#ifdef FEATHER_WITH_CUDA
    kernel::cuda_detail::ExecutionContextScope context_scope(Runtime().CudaContext());
#endif
    const auto logits = Runtime().GetTensor(logits_output_name_);
    if (logits == nullptr || logits->numel() <= 0) {
        return -1;
    }
    switch (logits->data_type()) {
        case DataType::FP32:
            return SelectGreedyFp32(logits->data<float>(), logits->numel());
        case DataType::FP16:
            return SelectGreedyFp16(logits->data<uint16_t>(), logits->numel());
        case DataType::BF16:
            return SelectGreedyBf16(logits->data<BFloat16>(), logits->numel());
        case DataType::INT64:
            return logits->numel() == 1 ? logits->data<int64_t>()[0] : -1;
        default:
            return -1;
    }
}

bool QwenExecutionPlan::HasSequenceCapacity() const {
    return execution_sequence_ > 0 && max_context_ > 0 && state_.tokens_processed >= 0 &&
           state_.tokens_processed <= max_context_ && execution_sequence_ <= max_context_ - state_.tokens_processed;
}

int32_t QwenExecutionPlan::RunStep(const int64_t* token_ids, int64_t count, int64_t* next_token_id, bool read_logits) {
#ifdef FEATHER_WITH_CUDA
    kernel::cuda_detail::ExecutionContextScope context_scope(Runtime().CudaContext());
#endif
    if (state_.invalid) {
        last_error_ = "Qwen state is invalid after an execution failure; call Reset before reuse";
        return -1;
    }
    if (!HasSequenceCapacity()) {
        last_error_ = "Qwen context capacity is exhausted";
        return -1;
    }
    if (next_token_id == nullptr || SetRuntimeInputs(token_ids, count) != 0) {
        return -1;
    }
    const auto begin = std::chrono::steady_clock::now();
    int32_t run_status = 0;
    std::string runtime_error;
    const auto run_runtime = [&]() {
        if (!HasSequenceCapacity()) {
            runtime_error = "Qwen execution exceeds context capacity";
            return -1;
        }
        const auto status = Runtime().RunStatus();
        runtime_error = status.message();
        return status.ok() ? 0 : -1;
    };
#ifdef FEATHER_WITH_CUDA
    int cuda_graph_slot_used = -1;
    if (backend_device_ == DeviceType::CUDA) {
        {
            kernel::cuda_detail::DeferredHostSyncScope deferred_host_sync;
            // Replay bypasses per-kernel host-cache invalidation. Retained
            // diagnostic tensors must observe each token's fresh activations.
            const bool use_cuda_graph = !load_options_.retain_logits && load_options_.additional_outputs.empty() &&
                                        execution_sequence_ == 1 && !Runtime().ProfilingEnabled() &&
                                        !cuda_graph_capture_disabled_ && steps_executed_ > 1;
            if (!use_cuda_graph) {
                run_status = run_runtime();
            } else {
                const int slot = cuda_graph_slot_;
                if (cuda_graph_slots_[static_cast<size_t>(slot)].executable == nullptr) {
                    const int32_t capture_status = CaptureCudaGraph(slot);
                    if (capture_status != 0) {
                        cuda_graph_capture_disabled_ = true;
                        run_status = run_runtime();
                    }
                }
                if (run_status == 0 && !cuda_graph_capture_disabled_) {
                    run_status = LaunchCudaGraph(slot);
                    if (run_status == 0) {
                        cuda_graph_slot_used = slot;
                    }
                    if (run_status != 0) {
                        cuda_graph_capture_disabled_ = true;
                        run_status = run_runtime();
                    }
                }
            }
        }
        if (run_status == 0 && kernel::cuda_detail::SynchronizeInferenceStream() != 0) {
            run_status = -1;
        }
        if (run_status == 0 && read_logits) {
            run_status = cuda_graph_slot_used >= 0 ? SyncCudaGraphLogits(cuda_graph_slot_used)
                                                    : SyncTensorFromCuda(Runtime().GetTensor(logits_output_name_));
        }
    } else
#endif
    {
        run_status = run_runtime();
    }
    if (run_status != 0) {
        state_.invalid = true;
        last_error_ = "Qwen runtime graph execution failed";
        if (!runtime_error.empty()) last_error_ += ": " + runtime_error;
#ifdef FEATHER_WITH_CUDA
        if (backend_device_ == DeviceType::CUDA) {
            last_error_ += ": " + kernel::cuda_detail::CudaLastErrorMessage();
        }
#endif
        return -1;
    }
    if (UpdateStates() != 0) {
        state_.invalid = true;
        return -1;
    }
#ifdef FEATHER_WITH_CUDA
    if (backend_device_ == DeviceType::CUDA) {
        cuda_graph_slot_ = unified_execution_ ? 0 : 1 - cuda_graph_slot_;
    }
#endif
    if (read_logits) {
        *next_token_id = SelectGreedyToken();
        if (*next_token_id < 0) {
            state_.invalid = true;
            last_error_ = "Qwen logits are unavailable";
            return -1;
        }
    }
    state_.tokens_processed += count;
    ++steps_executed_;
    const auto end = std::chrono::steady_clock::now();
    last_step_summary_ = "token=" + std::to_string(token_ids[count - 1]) +
        (read_logits ? " next=" + std::to_string(*next_token_id) : std::string{}) +
        (count == 1 ? " decode_ms=" : " chunk_ms=") +
        std::to_string(std::chrono::duration<double, std::milli>(end - begin).count());
    return 0;
}

#ifdef FEATHER_WITH_CUDA
void QwenExecutionPlan::ResetCudaGraphs() {
#ifdef FEATHER_WITH_CUDA
    kernel::cuda_detail::ExecutionContextScope context_scope(Runtime().CudaContext());
#endif
    bool has_executable = false;
    for (const auto& slot : cuda_graph_slots_) {
        has_executable = has_executable || slot.executable != nullptr;
    }
    if (has_executable) {
        (void)kernel::cuda_detail::SynchronizeInferenceStream();
    }
    for (auto& slot : cuda_graph_slots_) {
        if (slot.executable != nullptr) {
            slot.executable.reset();
        }
        slot.logits_device = nullptr;
        slot.logits_bytes = 0;
    }
    cuda_graph_slot_ = 0;
    cuda_graph_capture_disabled_ = false;
    cuda_graph_launch_count_ = 0;
}

int32_t QwenExecutionPlan::CaptureCudaGraph(int slot) {
#ifdef FEATHER_WITH_CUDA
    kernel::cuda_detail::ExecutionContextScope context_scope(Runtime().CudaContext());
#endif
    if (!HasSequenceCapacity() || slot < 0 || slot >= static_cast<int>(cuda_graph_slots_.size()) ||
        cuda_graph_slots_[static_cast<size_t>(slot)].executable != nullptr) {
        return -1;
    }
    std::unique_ptr<ExecutionCapture> executable;
    const auto capture_status = Runtime().Backend().Capture([&] { return Runtime().RunStatus(); }, &executable);
    if (!capture_status.ok()) return 1;
    auto logits = Runtime().GetTensor(logits_output_name_);
    const size_t logits_bytes = logits == nullptr ? 0 : TensorBytes(*logits);
    void* logits_device = nullptr;
    if (logits_bytes == 0 ||
        kernel::cuda_detail::AcquireTensorDevice(logits.get(), logits_bytes, logits->raw_data(), &logits_device) != 0 ||
        logits_device == nullptr) {
        return 1;
    }
    auto& graph_slot = cuda_graph_slots_[static_cast<size_t>(slot)];
    graph_slot.executable = std::move(executable);
    graph_slot.logits_device = logits_device;
    graph_slot.logits_bytes = logits_bytes;
    return 0;
}

int32_t QwenExecutionPlan::LaunchCudaGraph(int slot) {
#ifdef FEATHER_WITH_CUDA
    kernel::cuda_detail::ExecutionContextScope context_scope(Runtime().CudaContext());
#endif
    if (!HasSequenceCapacity() || slot < 0 || slot >= static_cast<int>(cuda_graph_slots_.size())) {
        return -1;
    }
    const auto& executable = cuda_graph_slots_[static_cast<size_t>(slot)].executable;
    if (!executable || !executable->Launch().ok()) return -1;
    ++cuda_graph_launch_count_;
    return 0;
}

int32_t QwenExecutionPlan::SyncCudaGraphLogits(int slot) {
#ifdef FEATHER_WITH_CUDA
    kernel::cuda_detail::ExecutionContextScope context_scope(Runtime().CudaContext());
#endif
    if (slot < 0 || slot >= static_cast<int>(cuda_graph_slots_.size())) {
        return -1;
    }
    const auto& graph_slot = cuda_graph_slots_[static_cast<size_t>(slot)];
    auto logits = Runtime().GetTensor(logits_output_name_);
    const auto stream = kernel::cuda_detail::InferenceStream();
    if (graph_slot.logits_device == nullptr || graph_slot.logits_bytes == 0 || logits == nullptr ||
        graph_slot.logits_bytes != TensorBytes(*logits) || stream == nullptr ||
        cudaMemcpyAsync(logits->mutable_data(graph_slot.logits_bytes), graph_slot.logits_device, graph_slot.logits_bytes, cudaMemcpyDeviceToHost,
                        stream) != cudaSuccess) {
        (void)cudaGetLastError();
        return -1;
    }
    return kernel::cuda_detail::SynchronizeInferenceStream();
}

#endif

int32_t QwenExecutionPlan::Build(std::shared_ptr<const QwenModelResources> resources,
                                 const QwenModelSource& source, QwenBackend backend,
                                 const QwenLoadOptions& options, int64_t context_size,
                                 int64_t selected_sequence, const QwenExecutionPlan* weight_owner) {
    resources_ = std::move(resources);
    load_options_ = options;
    backend_ = backend;
    backend_device_ = ResolveBackendDevice(backend);
    const bool shared_plan = weight_owner && &weight_owner->state_ == &state_;
    // Specializers operate on a temporary view; the published source remains
    // immutable and no plan retains a loader or a second canonical model.
    auto loader = source.loader;
#ifdef FEATHER_WITH_CUDA
    if (shared_plan && backend_device_ == DeviceType::CUDA)
        cuda_context_ = weight_owner->Runtime().CudaContext();
#endif
    std::unordered_map<std::string, std::shared_ptr<Tensor>> constant_overrides;
    if (!SpecializeQwenContext(loader, context_size, &model_, &constant_overrides,
                               &model_context_limit_, &last_error_)) {
        return -1;
    }
    const auto sequence_version = model_.metadata.find("qwen.sequence.version");
    unified_execution_ = sequence_version != model_.metadata.end();
    if (unified_execution_) {
        if (!SpecializeQwenSequence(loader, selected_sequence, &model_, &constant_overrides, &last_error_)) return -1;
        const auto limit = model_.metadata.find("qwen.sequence.maximum");
        const auto* maximum = limit == model_.metadata.end() ? nullptr : std::get_if<int64_t>(&limit->second);
        if (!maximum || *maximum <= 0) { last_error_ = "Invalid Qwen sequence maximum"; return -1; }
        prefill_limit_ = *maximum;
    }
    model_.variants.clear();  // The loader owns optional plans; this compiler uses one.
    for (const auto& name : load_options_.additional_outputs) {
        if (!FindValueDesc(name)) {
            last_error_ = "Unknown Qwen diagnostic output: " + name;
            return -1;
        }
        if (!HasName(model_.graph.outputs, name)) model_.graph.outputs.push_back(name);
    }
    const auto& model = model_;
    token_input_name_ = "token_ids";
    position_input_name_ = "position_id";
    attention_mask_input_name_ = "attention_mask";
    logits_output_name_ = "logits";
    if (!HasName(model.graph.inputs, token_input_name_) || !HasName(model.graph.inputs, position_input_name_) ||
        !HasName(model.graph.inputs, attention_mask_input_name_) || !HasName(model.graph.outputs, logits_output_name_)) {
        last_error_ = "Qwen graph is missing token_ids, position_id, attention_mask, or logits";
        return -1;
    }
    const auto* mask_desc = FindValueDesc(attention_mask_input_name_);
    if (mask_desc == nullptr || mask_desc->tensor.data_type != DataType::BF16 || mask_desc->tensor.dims.size() != 4 ||
        mask_desc->tensor.dims[0] != 1 || mask_desc->tensor.dims[1] != 1 || mask_desc->tensor.dims[2] != 1 ||
        mask_desc->tensor.dims[3] < 2) {
        last_error_ = "Qwen attention_mask must be BF16 [1,1,1,max_context]";
        return -1;
    }
    max_context_ = mask_desc->tensor.dims[3];

    const auto execution_version = model.metadata.find("qwen.execution.version");
    const auto execution_sequence = model.metadata.find("qwen.execution.sequence");
    const auto cache_layout = model.metadata.find("qwen.cache.layout");
    const auto weights_id = model.metadata.find("qwen.weights.id");
    const auto prefill_file = model.metadata.find("qwen.prefill.file");
    const auto prefill_variant = model.metadata.find("qwen.prefill.variant");
    const bool has_execution_metadata = execution_version != model.metadata.end() ||
        execution_sequence != model.metadata.end() || cache_layout != model.metadata.end() ||
        weights_id != model.metadata.end() || prefill_file != model.metadata.end() ||
        prefill_variant != model.metadata.end();
    if (has_execution_metadata) {
        const auto* version = execution_version == model.metadata.end() ? nullptr :
            std::get_if<int64_t>(&execution_version->second);
        const auto* sequence = execution_sequence == model.metadata.end() ? nullptr :
            std::get_if<int64_t>(&execution_sequence->second);
        const auto* layout = cache_layout == model.metadata.end() ? nullptr :
            std::get_if<std::string>(&cache_layout->second);
        const auto* identity = weights_id == model.metadata.end() ? nullptr :
            std::get_if<std::string>(&weights_id->second);
        if (!version || (*version != 1 && *version != 2) || (*version == 2 && !unified_execution_) || !sequence || *sequence <= 0 || *sequence > max_context_ ||
            (!shared_plan && *sequence != 1) || !layout || *layout != "position" || !identity || identity->empty()) {
            last_error_ = "Invalid or unsupported Qwen execution metadata; load a supported decode graph";
            return -1;
        }
        optimized_execution_ = true;
        execution_sequence_ = *sequence;
        weights_id_ = *identity;
        const auto* token_desc = FindValueDesc(token_input_name_);
        const auto* position_desc = FindValueDesc(position_input_name_);
        if (!token_desc || token_desc->tensor.data_type != DataType::INT64 ||
            token_desc->tensor.dims != std::vector<int64_t>({1, execution_sequence_}) || !position_desc ||
            position_desc->tensor.data_type != DataType::INT64 || position_desc->tensor.dims != std::vector<int64_t>({1})) {
            last_error_ = "Qwen execution metadata does not match token_ids or position_id";
            return -1;
        }
        std::unordered_set<std::string> caches;
        for (const auto& node : model.graph.nodes) {
            if (node.op_type != "CachedAttention") continue;
            if (node.inputs.size() != 6 || node.inputs[5] != position_input_name_) {
                last_error_ = "Qwen CachedAttention has an invalid position/cache contract";
                return -1;
            }
            for (size_t slot : {size_t{3}, size_t{4}}) {
                const auto* desc = FindValueDesc(node.inputs[slot]);
                if (!HasName(model.graph.inputs, node.inputs[slot]) || !desc ||
                    desc->tensor.data_type != DataType::BF16 || desc->tensor.dims.size() != 4 ||
                    desc->tensor.dims[0] != 1 || desc->tensor.dims[1] <= 0 ||
                    desc->tensor.dims[2] != max_context_ || desc->tensor.dims[3] <= 0 ||
                    !caches.insert(node.inputs[slot]).second ||
                    HasName(model.graph.outputs, "next_" + node.inputs[slot])) {
                    last_error_ = "Qwen position cache contract is invalid for " + node.inputs[slot];
                    return -1;
                }
                inplace_cache_names_.push_back(node.inputs[slot]);
            }
        }
    }

    for (const auto& input_name : model.graph.inputs) {
        if (input_name == token_input_name_ || input_name == position_input_name_ || input_name == attention_mask_input_name_) {
            continue;
        }
        state_input_names_.push_back(input_name);
        if (HasName(inplace_cache_names_, input_name)) continue;
        const std::string output_name = "next_" + input_name;
        const auto* input_desc = FindValueDesc(input_name);
        const auto* output_desc = FindValueDesc(output_name);
        if (input_desc == nullptr || output_desc == nullptr || !HasName(model.graph.outputs, output_name) ||
            input_desc->tensor.data_type != output_desc->tensor.data_type ||
            input_desc->tensor.dims.size() != output_desc->tensor.dims.size()) {
            last_error_ = "Qwen graph state contract is invalid for " + input_name;
            return -1;
        }
        int cache_axis = -1;
        for (size_t axis = 0; axis < input_desc->tensor.dims.size(); ++axis) {
            if (input_desc->tensor.dims[axis] == output_desc->tensor.dims[axis]) {
                continue;
            }
            if (optimized_execution_ || cache_axis >= 0 || input_desc->tensor.dims[axis] <= 1 ||
                output_desc->tensor.dims[axis] != 1) {
                last_error_ = "Qwen state dimensions are incompatible for " + input_name;
                return -1;
            }
            cache_axis = static_cast<int>(axis);
        }
        states_.push_back({input_name, output_name, cache_axis, true});
    }
    if (state_input_names_.empty()) {
        last_error_ = "Qwen graph does not declare explicit decode state";
        return -1;
    }

    static_graph_ = StaticGraph();
    static_graph_.SetKernelDevice(backend_device_);
    static_graph_.SetPassManager(CreateQwenPassManager(!load_options_.retain_logits));
    session_.reset();
    compiled_graph_.reset();
    Runtime().Clear();
    Runtime().SetThreadMode(RuntimeThreadMode::kSerialGraph);
    if (static_graph_.SetModel(model) != 0) {
        last_error_ = "failed to set Qwen model on static graph";
        return -1;
    }
    for (const auto& value : model.graph.values) {
        if (!value.constant) {
            continue;
        }
        const auto constant_override = constant_overrides.find(value.tensor.name);
        auto tensor = constant_override == constant_overrides.end() ? loader.CreateWeightTensor(value.tensor.name)
                                                                    : constant_override->second;
        if (tensor == nullptr || static_graph_.SetTensor(value.tensor.name, tensor) != 0) {
            last_error_ = "failed to attach Qwen weight: " + value.tensor.name;
            return -1;
        }
    }
    for (const auto& input_name : model.graph.inputs) {
        const auto* input_desc = FindValueDesc(input_name);
        const auto state = state_.tensors.find(input_name);
        auto tensor = shared_plan && HasName(state_input_names_, input_name)
            ? state == state_.tensors.end() ? nullptr : state->second
            : input_desc == nullptr ? nullptr : CreateZeroTensor(input_desc->tensor);
        if (tensor == nullptr || static_graph_.SetTensor(input_name, tensor) != 0) {
            last_error_ = "failed to allocate Qwen input: " + input_name;
            return -1;
        }
        if (!shared_plan && HasName(state_input_names_, input_name)) state_.tensors[input_name] = tensor;
    }
    if (PrepareExecutableGraph(weight_owner ? weight_owner->session_.get() : nullptr) != 0) {
        const auto detail = last_error_;
        last_error_ = "failed to build Qwen graph for backend " + std::string(QwenBackendName(backend));
        if (!detail.empty()) last_error_ += ": " + detail;
        return -1;
    }
    if (shared_plan) {
        auto scope = Runtime().Backend().Enter();
        for (const auto& name : state_input_names_) Runtime().Backend().MarkPersistent(*Runtime().GetTensor(name), true);
        for (const auto& binding : states_) {
            auto output = Runtime().GetTensor(binding.output_name);
            if (output) Runtime().Backend().MarkPersistent(*output, true);
            const auto* producer = Runtime().GetProducerNode(binding.output_name);
            if (producer && producer->semantics) {
                auto source = Runtime().GetTensor(WholeTensorAliasSource(
                    *producer->semantics, producer->inputs, producer->outputs, binding.output_name));
                if (source) Runtime().Backend().MarkPersistent(*source, true);
            }
        }
    } else if (Reset() != 0) {
        return -1;
    }
    return 0;
}

bool ParseQwenBackend(const std::string& value, QwenBackend* backend) {
    if (backend == nullptr) {
        return false;
    }
    std::string normalized;
    normalized.reserve(value.size());
    for (const auto character : value) {
        normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(character))));
    }
    if (normalized == "host") {
        *backend = QwenBackend::kHost;
    } else if (normalized == "common") {
        *backend = QwenBackend::kCommon;
    } else if (normalized == "x86") {
        *backend = QwenBackend::kX86;
    } else if (normalized == "cuda") {
        *backend = QwenBackend::kCuda;
    } else {
        return false;
    }
    return true;
}

const char* QwenBackendName(QwenBackend backend) {
    switch (backend) {
        case QwenBackend::kHost:
            return "host";
        case QwenBackend::kX86:
            return "x86";
        case QwenBackend::kCuda:
            return "cuda";
        case QwenBackend::kCommon:
        default:
            return "common";
    }
}

int32_t QwenRunner::ReadOutput(const std::string& name, std::vector<float>* values) {
    OperationGuard operation(*this);
    if (!operation) return -1;
    return ReadOutputInternal(name, values);
}

int32_t QwenRunner::ReadOutputInternal(const std::string& name, std::vector<float>* values) {
    if (!values || !decode_plan_ || !decode_plan_->session_ || !diagnostic_outputs_ready_ || decode_->invalid ||
        !HasName(decode_plan_->model_.graph.outputs, name)) {
        last_error_ = "Qwen output is not readable: " + name;
        return -1;
    }
    const auto tensor = Runtime().GetTensor(name);
    if (!tensor || tensor->numel() <= 0 ||
        (tensor->data_type() != DataType::FP32 && tensor->data_type() != DataType::BF16 &&
         tensor->data_type() != DataType::FP16)) {
        last_error_ = "Qwen diagnostic output must be floating point: " + name;
        return -1;
    }
    const auto status = Runtime().Backend().ReadHost(*tensor);
    if (!status.ok()) { last_error_ = status.message(); return -1; }
    std::vector<float> result(static_cast<size_t>(tensor->numel()));
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = tensor->data_type() == DataType::FP32 ? tensor->data<float>()[i] :
            tensor->data_type() == DataType::BF16 ? BFloat16ToFloat(tensor->data<BFloat16>()[i].bits) :
            HalfToFloat(tensor->data<uint16_t>()[i]);
        if (!std::isfinite(result[i])) {
            last_error_ = "Qwen diagnostic output contains non-finite values: " + name;
            return -1;
        }
    }
    *values = std::move(result);
    return 0;
}

int32_t QwenRunner::EvaluateToken(int64_t token_id, std::vector<float>* logits) {
    OperationGuard operation(*this);
    if (!operation) return -1;
    ++decode_->revision;
    if (!load_options_.retain_logits || !logits) {
        last_error_ = "Qwen scoring requires retain_logits at load and a non-null output";
        return -1;
    }
    int64_t next = -1;
    if (EnsureContext(decode_->tokens_processed + 1) != 0 || RunToken(token_id, &next) != 0) return -1;
    diagnostic_outputs_ready_ = true;
    return ReadOutputInternal(decode_plan_->logits_output_name_, logits);
}

int32_t QwenRunner::CheckCancellation(const std::function<bool()>& callback) {
    try {
        return callback && callback() ? 1 : 0;
    } catch (const std::exception& error) {
        last_error_ = "Qwen cancellation callback failed: " + std::string(error.what());
    } catch (...) {
        last_error_ = "Qwen cancellation callback failed with an unknown exception";
    }
    return -1;
}

int32_t QwenRunner::InvokeTokenCallback(const std::function<void(int64_t)>& callback, int64_t token) {
    try {
        callback(token);
        return 0;
    } catch (const std::exception& error) {
        last_error_ = "Qwen token callback failed: " + std::string(error.what());
    } catch (...) {
        last_error_ = "Qwen token callback failed with an unknown exception";
    }
    return -1;
}

int32_t QwenRunner::RunPrompt(const std::vector<int64_t>& token_ids, const std::function<bool()>& should_cancel,
                              int64_t* next_token_id) {
    diagnostic_outputs_ready_ = false;
    if (decode_->invalid) {
        last_error_ = "Qwen state is invalid after an execution failure; call Reset before reuse";
        return -1;
    }
    const auto begin = std::chrono::steady_clock::now();
    const int64_t initial_tokens = decode_->tokens_processed;
    last_prompt_tokens_consumed_ = 0;
    last_prefill_chunks_ = 0;
    last_prefill_ms_ = 0.0;
    const auto cancelled = [&] { return CheckCancellation(should_cancel); };
    const auto finish = [&](int32_t status) {
        last_prompt_tokens_consumed_ = decode_->tokens_processed - initial_tokens;
        last_prefill_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        RefreshRunSummary();
        return status;
    };
    size_t offset = 0;
    if (decode_plan_->unified_execution_) {
        while (offset < token_ids.size()) {
            if (const auto cancellation = cancelled(); cancellation != 0) return finish(cancellation);
            const auto remaining = static_cast<int64_t>(token_ids.size() - offset);
            int64_t chunk = 1;
            while (chunk <= std::min(remaining, decode_plan_->prefill_limit_) / 2) chunk *= 2;
            auto* plan = GetSequencePlan(chunk);
            if (!plan) return finish(-1);
            if (plan->RunStep(token_ids.data() + offset, chunk, next_token_id,
                              remaining == chunk) != 0) {
                last_error_ = plan->last_error_;
                decode_->invalid = true;
                return finish(-1);
            }
            last_step_summary_ = plan->last_step_summary_;
            offset += static_cast<size_t>(chunk);
            if (chunk > 1) ++last_prefill_chunks_;
        }
        return finish(0);
    }
    if (prefill_plan_ && token_ids.size() >= static_cast<uint64_t>(prefill_plan_->execution_sequence_)) {
        if (const auto cancellation = cancelled(); cancellation != 0) return finish(cancellation);
        const size_t chunk = static_cast<size_t>(prefill_plan_->execution_sequence_);
        int32_t status = 0;
        bool first_chunk = true;
        while (token_ids.size() - offset >= chunk) {
            if (!first_chunk) {
                const auto cancellation = cancelled();
                if (cancellation != 0) { status = cancellation; break; }
            }
            first_chunk = false;
            const bool last_chunk_is_final = token_ids.size() - offset == chunk;
            if (prefill_plan_->RunStep(token_ids.data() + offset, static_cast<int64_t>(chunk), next_token_id,
                                         last_chunk_is_final) != 0) {
                last_error_ = "Qwen prefill execution failed: " + prefill_plan_->last_error_;
                // A failed chunk may already have written some KV layers.
                // Invalidate the shared request before either executable can
                // observe a partially updated cache; Reset restores all plans.
                decode_->invalid = true;
                return finish(-1);
            }
            offset += chunk;
            ++last_prefill_chunks_;
            last_step_summary_ = prefill_plan_->last_step_summary_;
        }
        if (status != 0) return finish(status);
    }
    for (; offset < token_ids.size(); ++offset) {
        if (const auto cancellation = cancelled(); cancellation != 0) return finish(cancellation);
        if (RunToken(token_ids[offset], next_token_id) != 0) return finish(-1);
    }
    return finish(0);
}

int32_t QwenRunner::GenerateImpl(const std::vector<int64_t>& prompt_tokens,
                                 const QwenGenerationOptions& options,
                                 const std::vector<int64_t>& stop_token_ids,
                                 const std::function<void(int64_t)>& on_token,
                                 std::vector<int64_t>* generated_tokens) {
    ++decode_->revision;
#ifdef FEATHER_WITH_CUDA
    kernel::cuda_detail::ExecutionContextScope context_scope(Runtime().CudaContext());
#endif
    last_error_.clear();
    last_finish_reason_ = QwenFinishReason::kNone;
    if (prompt_tokens.empty() || (options.max_new_tokens <= 0 && options.max_new_tokens != -1) ||
        (!on_token && generated_tokens == nullptr)) {
        last_error_ = "Qwen prompt must be non-empty and max_new_tokens must be positive or -1";
        return -1;
    }
    if (!decode_plan_ || decode_plan_->max_context_ <= 0 || Runtime().NodeSize() == 0) {
        last_error_ = "Qwen runner is not loaded";
        return -1;
    }
    if (options.context_reserve_tokens < 0 || options.context_reserve_tokens > (auto_context_growth_ ? ModelContextLimit() : MaxContext())) {
        last_error_ = "Qwen context reserve must be between zero and context capacity";
        return -1;
    }
    if (prompt_tokens.size() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - decode_->tokens_processed) ||
        options.context_reserve_tokens > std::numeric_limits<int64_t>::max() - decode_->tokens_processed -
            static_cast<int64_t>(prompt_tokens.size()) ||
        EnsureContext(decode_->tokens_processed + static_cast<int64_t>(prompt_tokens.size()) + options.context_reserve_tokens) != 0)
        return -1;
    int64_t generation_capacity = decode_plan_->max_context_ - options.context_reserve_tokens;
    if (decode_->tokens_processed > generation_capacity ||
        prompt_tokens.size() > static_cast<uint64_t>(generation_capacity - decode_->tokens_processed)) {
        last_error_ = "Qwen prompt exceeds available context capacity after reserve";
        return -1;
    }
    // Inputs may be the caller's output vector. Retain the prompt only when it
    // aliases, and freeze stop IDs before clearing or appending any output.
    const std::vector<int64_t> saved_prompt = generated_tokens == &prompt_tokens ? prompt_tokens : std::vector<int64_t>{};
    const auto& prompt = generated_tokens == &prompt_tokens ? saved_prompt : prompt_tokens;
    const std::unordered_set<int64_t> stop_ids(stop_token_ids.begin(), stop_token_ids.end());
    if (generated_tokens != nullptr) generated_tokens->clear();
    const auto cancelled = [&] { return CheckCancellation(options.should_cancel); };
    int64_t next_token = -1;
    const int32_t prompt_status = RunPrompt(prompt, options.should_cancel, &next_token);
    if (prompt_status < 0) return -1;
    if (prompt_status > 0) { last_finish_reason_ = QwenFinishReason::kCancelled; return 0; }
    int64_t generated_count = 0;
    while (true) {
        const auto cancellation = cancelled();
        if (cancellation < 0) return -1;
        if (cancellation > 0) {
            last_finish_reason_ = QwenFinishReason::kCancelled;
            return 0;
        }
        if (options.max_new_tokens != -1 && generated_count >= options.max_new_tokens) {
            last_finish_reason_ = QwenFinishReason::kLength;
            return 0;
        }
        if (decode_->tokens_processed >= generation_capacity && auto_context_growth_ && decode_plan_->max_context_ < decode_plan_->model_context_limit_) {
            if (EnsureContext(decode_->tokens_processed + options.context_reserve_tokens + 1) != 0) return -1;
            generation_capacity = decode_plan_->max_context_ - options.context_reserve_tokens;
        }
        if (decode_->tokens_processed >= generation_capacity) {
            last_finish_reason_ = QwenFinishReason::kContextLimit;
            return 0;
        }
        const int64_t generated_token = next_token;
        if (on_token && InvokeTokenCallback(on_token, generated_token) != 0) return -1;
        if (RunToken(next_token, &next_token) != 0) {
            return -1;
        }
        if (generated_tokens != nullptr) {
            generated_tokens->push_back(generated_token);
        }
        ++generated_count;
        if (stop_ids.count(generated_token) != 0) {
            last_finish_reason_ = QwenFinishReason::kEos;
            return 0;
        }
    }
}

int32_t QwenRunner::Consume(const std::vector<int64_t>& token_ids) {
    OperationGuard operation(*this);
    if (!operation) return -1;
    ++decode_->revision;
#ifdef FEATHER_WITH_CUDA
    kernel::cuda_detail::ExecutionContextScope context_scope(Runtime().CudaContext());
#endif
    last_error_.clear();
    if (token_ids.empty()) {
        last_error_ = "Qwen tokens to consume must be non-empty";
        return -1;
    }
    if (token_ids.size() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - decode_->tokens_processed) ||
        EnsureContext(decode_->tokens_processed + static_cast<int64_t>(token_ids.size())) != 0) return -1;
    if (!decode_plan_ || decode_plan_->max_context_ <= 0 || decode_->tokens_processed > decode_plan_->max_context_ ||
        token_ids.size() > static_cast<uint64_t>(decode_plan_->max_context_ - decode_->tokens_processed)) {
        last_error_ = "Qwen tokens to consume exceed available context capacity";
        return -1;
    }
    int64_t ignored_next_token = -1;
    return RunPrompt(token_ids, std::function<bool()>(), &ignored_next_token);
}

int32_t QwenRunner::EnsureContext(int64_t required) {
    if (!decode_plan_ || decode_plan_->max_context_ <= 0 || !decode_plan_->session_ || Runtime().NodeSize() == 0) {
        last_error_ = "Qwen runner is not loaded";
        return -1;
    }
    if (required < 0) { last_error_ = "Qwen context size overflow"; return -1; }
    if (required <= decode_plan_->max_context_) return 0;
    if (!auto_context_growth_ || required > decode_plan_->model_context_limit_) {
        last_error_ = "Qwen tokens exceed available context capacity";
        return -1;
    }
    int64_t capacity = decode_plan_->max_context_;
    while (capacity < required) capacity = capacity > decode_plan_->model_context_limit_ / 2 ? decode_plan_->model_context_limit_ : capacity * 2;
    return ReserveContextInternal(capacity);
}

void QwenRunner::RefreshRunSummary() {
    last_run_summary_ = last_step_summary_;
    if (decode_plan_->optimized_execution_) {
        last_run_summary_ += " prefill_chunk=" + std::to_string(PrefillChunkSize()) +
            " prompt_consumed=" + std::to_string(last_prompt_tokens_consumed_) +
            " prefill_chunks=" + std::to_string(last_prefill_chunks_) +
            " prefill_ms=" + std::to_string(last_prefill_ms_) +
            " cached_plans=" + std::to_string(CachedPlanCount()) +
            " context_growths=" + std::to_string(context_growth_count_);
    }
}

int32_t QwenRunner::GenerateStream(const std::vector<int64_t>& prompt_tokens, int max_new_tokens,
                                   const std::vector<int64_t>& stop_token_ids,
                                   const std::function<void(int64_t)>& on_token) {
    QwenGenerationOptions options;
    options.max_new_tokens = max_new_tokens;
    return GenerateStream(prompt_tokens, options, stop_token_ids, on_token);
}

int32_t QwenRunner::Generate(const std::vector<int64_t>& prompt_tokens, int max_new_tokens,
                              const std::vector<int64_t>& stop_token_ids, std::vector<int64_t>* generated_tokens) {
    QwenGenerationOptions options;
    options.max_new_tokens = max_new_tokens;
    return Generate(prompt_tokens, options, stop_token_ids, generated_tokens);
}

const char* QwenFinishReasonName(QwenFinishReason reason) {
    switch (reason) {
        case QwenFinishReason::kEos: return "eos";
        case QwenFinishReason::kLength: return "length";
        case QwenFinishReason::kContextLimit: return "context_limit";
        case QwenFinishReason::kCancelled: return "cancelled";
        case QwenFinishReason::kNone: default: return "none";
    }
}

int32_t QwenRunner::GenerateStream(const std::vector<int64_t>& prompt_tokens,
                                   const QwenGenerationOptions& options,
                                   const std::vector<int64_t>& stop_token_ids,
                                   const std::function<void(int64_t)>& on_token) {
    OperationGuard operation(*this);
    if (!operation) return -1;
    if (!on_token) {
        last_error_ = "Qwen streaming callback must not be empty";
        last_finish_reason_ = QwenFinishReason::kNone;
        return -1;
    }
    return GenerateImpl(prompt_tokens, options, stop_token_ids, on_token, nullptr);
}

int32_t QwenRunner::Generate(const std::vector<int64_t>& prompt_tokens,
                             const QwenGenerationOptions& options,
                             const std::vector<int64_t>& stop_token_ids,
                             std::vector<int64_t>* generated_tokens) {
    OperationGuard operation(*this);
    if (!operation) return -1;
    if (generated_tokens == nullptr) {
        last_error_ = "Qwen generated token output must not be null";
        last_finish_reason_ = QwenFinishReason::kNone;
        return -1;
    }
    return GenerateImpl(prompt_tokens, options, stop_token_ids, std::function<void(int64_t)>(), generated_tokens);
}


QwenRunner::QwenRunner() : decode_(std::make_unique<QwenDecodeState>()) {}
QwenRunner::~QwenRunner() = default;

RuntimeGraph& QwenRunner::Runtime() { return decode_plan_ ? decode_plan_->Runtime() : bootstrap_graph_; }
const RuntimeGraph& QwenRunner::Runtime() const { return decode_plan_ ? decode_plan_->Runtime() : bootstrap_graph_; }
size_t QwenRunner::CachedPlanCount() const { return (decode_plan_ ? 1 : 0) + sequence_plans_.size(); }
int64_t QwenRunner::TokensProcessed() const { return decode_->tokens_processed; }
uint64_t QwenRunner::StateRevision() const { return decode_->revision; }
int64_t QwenRunner::MaxContext() const { return decode_plan_ ? decode_plan_->max_context_ : 0; }
int64_t QwenRunner::ModelContextLimit() const { return decode_plan_ ? decode_plan_->model_context_limit_ : 0; }
int64_t QwenRunner::PrefillChunkSize() const {
    if (!decode_plan_) return 1;
    return decode_plan_->unified_execution_ ? decode_plan_->prefill_limit_ :
        prefill_plan_ ? prefill_plan_->execution_sequence_ : decode_plan_->execution_sequence_;
}
int64_t QwenRunner::CudaGraphLaunchCount() const {
    return decode_plan_ ? decode_plan_->cuda_graph_launch_count_ : 0;
}
const std::vector<RuntimeProfileSummary>& QwenRunner::RuntimeProfileSummaries() const {
    return Runtime().ProfileSummaries();
}
void QwenRunner::SetRuntimeProfilingEnabled(bool enabled) {
    OperationGuard operation(*this);
    if (!operation) return;
    if (decode_plan_) decode_plan_->SetRuntimeProfilingEnabled(enabled);
    else bootstrap_graph_.SetProfilingEnabled(enabled);
    if (prefill_plan_) prefill_plan_->SetRuntimeProfilingEnabled(enabled);
    for (auto& plan : sequence_plans_) plan.second->SetRuntimeProfilingEnabled(enabled);
}
bool QwenRunner::SharesModelResourcesWith(const QwenRunner& other) const {
    return resources_ && resources_ == other.resources_;
}
int32_t QwenRunner::CreateRequest(std::unique_ptr<QwenRunner>* request) const {
    OperationGuard operation(*this);
    if (!operation) return -1;
    if (!request || request->get() == this || !decode_plan_ || !resources_) return -1;
    auto candidate = std::make_unique<QwenRunner>();
    candidate->load_options_ = load_options_;
    if (candidate->LoadInternal(model_path_, backend_, MaxContext(), resources_, decode_plan_.get()) != 0) return -1;
    candidate->auto_context_growth_ = auto_context_growth_;
    candidate->SetRuntimeProfilingEnabled(Runtime().ProfilingEnabled());
    *request = std::move(candidate);
    return 0;
}

void QwenRunner::UnloadExecution() {
    // Plans retire captures and unregister their runtime handles before state
    // Tensor objects and the shared model mappings are released.
    sequence_plans_.clear();
    prefill_plan_.reset();
    decode_plan_.reset();
    decode_->tensors.clear();
    resources_.reset();
    bootstrap_graph_.Clear();
    diagnostic_outputs_ready_ = false;
}
int32_t QwenRunner::Load(const std::string& path, QwenBackend backend, int64_t context_size) {
    OperationGuard operation(*this);
    if (!operation) return -1;
    load_options_ = QwenLoadOptions{};
    load_options_.context_size = context_size;
    const auto status = LoadInternal(path, backend, context_size);
    if (status != 0) UnloadExecution();
    return status;
}
int32_t QwenRunner::Load(const std::string& path, QwenBackend backend, const QwenLoadOptions& options) {
    OperationGuard operation(*this);
    if (!operation) return -1;
    load_options_ = options;
    const auto status = LoadInternal(path, backend, options.context_size);
    if (status != 0) UnloadExecution();
    return status;
}

int32_t QwenRunner::LoadInternal(const std::string& path, QwenBackend backend, int64_t context_size,
                                std::shared_ptr<const QwenModelResources> resources,
                                const QwenExecutionPlan* weight_owner) {
    ++decode_->revision;
    UnloadExecution();
    decode_->tokens_processed = 0;
    decode_->invalid = false;
    last_error_.clear();
    last_build_summary_.clear();
    last_run_summary_.clear();
    last_step_summary_.clear();
    last_prompt_tokens_consumed_ = last_prefill_chunks_ = 0;
    last_prefill_ms_ = 0;
    last_finish_reason_ = QwenFinishReason::kNone;
    auto_context_growth_ = false;
    context_growth_count_ = 0;
    model_path_ = path;
    backend_ = backend;
    const auto fail = [this](const std::string& error) { last_error_ = error; return -1; };
#ifndef FEATHER_WITH_CUDA
    if (backend == QwenBackend::kCuda)
        return fail("CUDA backend requested but this build was compiled without CUDA support");
#else
    if (backend == QwenBackend::kCuda && kernel::cuda_detail::WarmupCudaRuntime() != 0)
        return fail("CUDA backend initialization failed: " + kernel::cuda_detail::CudaLastErrorMessage());
#endif
    if (!resources) {
        auto loaded = std::make_shared<QwenModelResources>();
        loaded->decode.path = path;
        if (!loaded->decode.loader.Load(path)) return fail("failed to load Feather model: " + path);
        // Resolve mappings before publishing the source. Temporary loader views
        // can then specialize/build without mutating the shared source.
        const auto map_weights = [&](QwenModelSource& source) {
            for (const auto& value : source.loader.model().graph.values)
                if (value.constant && !source.loader.CreateWeightTensor(value.tensor.name)) return false;
            return true;
        };
        if (!map_weights(loaded->decode)) return fail("failed to map Qwen model weights: " + path);
        const auto& metadata = loaded->decode.loader.model().metadata;
        const auto variant = metadata.find("qwen.prefill.variant");
        const auto file = metadata.find("qwen.prefill.file");
        if (variant != metadata.end() || file != metadata.end()) {
            auto companion = std::make_unique<QwenModelSource>();
            if (variant != metadata.end()) {
                const auto* name = std::get_if<std::string>(&variant->second);
                companion->path = path;
                if (!name || name->empty() || !loaded->decode.loader.SelectVariant(*name, &companion->loader))
                    return fail("Invalid or missing embedded Qwen prefill plan");
            } else {
                const auto* filename = std::get_if<std::string>(&file->second);
                if (!filename || filename->empty() || std::filesystem::path(*filename).is_absolute())
                    return fail("Qwen prefill companion must be a non-empty relative filename");
                companion->path = (std::filesystem::path(path).parent_path() / *filename).string();
                if (!companion->loader.Load(companion->path))
                    return fail("Failed to load Qwen prefill companion: " + companion->path);
            }
            const auto& other_metadata = companion->loader.model().metadata;
            const auto sequence_it = other_metadata.find("qwen.execution.sequence");
            const auto* sequence = sequence_it == other_metadata.end() ? nullptr : std::get_if<int64_t>(&sequence_it->second);
            if (!sequence || *sequence <= 1) return fail("Invalid Qwen prefill companion sequence");
            if (!map_weights(*companion)) return fail("failed to map Qwen prefill weights");
            loaded->prefill = std::move(companion);
        }
        resources = std::move(loaded);
    }
    resources_ = std::move(resources);
    auto plan = std::make_unique<QwenExecutionPlan>(*decode_);
    if (plan->Build(resources_, resources_->decode, backend, load_options_, context_size, 1, weight_owner) != 0)
        return fail(plan->last_error_);
    auto_context_growth_ = context_size == 0 && plan->model_.metadata.count("qwen.context.version") != 0;
    decode_plan_ = std::move(plan);
    if (resources_->prefill) {
        const auto& source = *resources_->prefill;
        const auto sequence = std::get<int64_t>(source.loader.model().metadata.at("qwen.execution.sequence"));
        if (sequence <= MaxContext()) {
            auto companion = std::make_unique<QwenExecutionPlan>(*decode_);
            if (companion->Build(resources_, source, backend, load_options_, MaxContext(), 1, decode_plan_.get()) != 0)
                return fail("Failed to load Qwen prefill companion: " + companion->last_error_);
            if (decode_plan_->ValidatePrefillCompanion(*companion) != 0) return fail(decode_plan_->last_error_);
            companion->SetRuntimeProfilingEnabled(Runtime().ProfilingEnabled());
            prefill_plan_ = std::move(companion);
        }
    }
    std::ostringstream summary;
    summary << "model=" << decode_plan_->model_.name << " backend=" << QwenBackendName(backend)
            << " max_context=" << MaxContext() << " static_nodes=" << decode_plan_->static_graph_.NodeSize()
            << " runtime_nodes=" << Runtime().NodeSize();
    if (decode_plan_->optimized_execution_) {
        summary << " execution_version=" << (decode_plan_->unified_execution_ ? 2 : 1)
                << " cache_layout=position prefill_chunk=" << PrefillChunkSize();
        if (prefill_plan_) summary << " prefill_runtime_nodes=" << prefill_plan_->Runtime().NodeSize()
            << " prefill_weights=shared reused_weight_bytes=" << prefill_plan_->session_->memory_usage().reused_weight_bytes;
    }
    last_build_summary_ = summary.str();
    return 0;
}

int32_t QwenRunner::Reset() {
    OperationGuard operation(*this);
    if (!operation) return -1;
    ++decode_->revision;
    diagnostic_outputs_ready_ = false;
    if (!decode_plan_) { last_error_ = "Qwen runner is not loaded"; return -1; }
    const auto reset = [this](QwenExecutionPlan& plan) {
        if (plan.Reset() == 0) return true;
        last_error_ = plan.last_error_;
        return false;
    };
    if (!reset(*decode_plan_)) return -1;
    if (prefill_plan_ && !reset(*prefill_plan_)) return -1;
    for (auto& item : sequence_plans_) if (!reset(*item.second)) return -1;
    last_error_.clear();
    last_finish_reason_ = QwenFinishReason::kNone;
    last_run_summary_.clear();
    last_step_summary_.clear();
    last_prompt_tokens_consumed_ = last_prefill_chunks_ = 0;
    last_prefill_ms_ = 0;
    return 0;
}

int32_t QwenRunner::RunToken(int64_t token_id, int64_t* next_token_id) {
    if (!decode_plan_) { last_error_ = "Qwen runner is not loaded"; return -1; }
    const auto status = decode_plan_->RunStep(&token_id, 1, next_token_id);
    if (status != 0) last_error_ = decode_plan_->last_error_;
    else { last_step_summary_ = decode_plan_->last_step_summary_; RefreshRunSummary(); }
    return status;
}
QwenExecutionPlan* QwenRunner::GetSequencePlan(int64_t sequence) {
    if (sequence == 1) return decode_plan_.get();
    const auto found = sequence_plans_.find(sequence);
    if (found != sequence_plans_.end()) return found->second.get();
    auto plan = std::make_unique<QwenExecutionPlan>(*decode_);
    if (plan->Build(resources_, resources_->decode, backend_, load_options_, MaxContext(), sequence, decode_plan_.get()) != 0) {
        last_error_ = "Failed to prepare Qwen sequence plan: " + plan->last_error_;
        return nullptr;
    }
    plan->SetRuntimeProfilingEnabled(Runtime().ProfilingEnabled());
    auto* result = plan.get();
    sequence_plans_.emplace(sequence, std::move(plan));
    return result;
}

}  // namespace demo
}  // namespace feather
