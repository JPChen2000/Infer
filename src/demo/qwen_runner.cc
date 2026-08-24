#include "demo/qwen_runner.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
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
    std::memset(tensor->raw_data(), 0, bytes);
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

const model::ValueDesc* QwenRunner::FindValueDesc(const std::string& name) const {
    const auto& values = loader_.model().graph.values;
    const auto it = std::find_if(values.begin(), values.end(), [&name](const model::ValueDesc& value) {
        return value.tensor.name == name;
    });
    return it == values.end() ? nullptr : &*it;
}

int32_t QwenRunner::Load(const std::string& model_path, QwenBackend backend) {
    last_error_.clear();
    states_.clear();
    tokens_processed_ = 0;
    backend_ = backend;
    backend_device_ = ResolveBackendDevice(backend);
#ifndef FEATHER_WITH_CUDA
    if (backend_device_ == DeviceType::CUDA) {
        last_error_ = "CUDA backend requested but this build was compiled without CUDA support";
        return -1;
    }
#endif
#ifdef FEATHER_WITH_CUDA
    if (backend_device_ == DeviceType::CUDA) {
        if (kernel::cuda_detail::WarmupCudaRuntime() != 0) {
            last_error_ = "CUDA backend initialization failed: " + kernel::cuda_detail::CudaLastErrorMessage();
            return -1;
        }
        kernel::cuda_detail::ClearTensorCache();
    }
#endif
    if (!loader_.Load(model_path)) {
        last_error_ = "failed to load Feather model: " + model_path;
        return -1;
    }
    const auto& model = loader_.model();
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

    for (const auto& input_name : model.graph.inputs) {
        if (input_name == token_input_name_ || input_name == position_input_name_ || input_name == attention_mask_input_name_) {
            continue;
        }
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
            if (cache_axis >= 0 || input_desc->tensor.dims[axis] <= 1 || output_desc->tensor.dims[axis] != 1) {
                last_error_ = "Qwen state dimensions are incompatible for " + input_name;
                return -1;
            }
            cache_axis = static_cast<int>(axis);
        }
        states_.push_back({input_name, output_name, cache_axis});
    }
    if (states_.empty()) {
        last_error_ = "Qwen graph does not declare explicit decode state";
        return -1;
    }

    static_graph_ = StaticGraph();
    static_graph_.SetKernelDevice(backend_device_);
    static_graph_.SetPassManager(CreateDefaultPassManager());
    runtime_graph_.Clear();
    runtime_graph_.SetThreadMode(RuntimeThreadMode::kSerialGraph);
    if (static_graph_.SetModel(model) != 0) {
        last_error_ = "failed to set Qwen model on static graph";
        return -1;
    }
    for (const auto& value : model.graph.values) {
        if (!value.constant) {
            continue;
        }
        auto tensor = loader_.CreateWeightTensor(value.tensor.name);
        if (tensor == nullptr || static_graph_.SetTensor(value.tensor.name, tensor) != 0) {
            last_error_ = "failed to attach Qwen weight: " + value.tensor.name;
            return -1;
        }
    }
    for (const auto& input_name : model.graph.inputs) {
        const auto* input_desc = FindValueDesc(input_name);
        auto tensor = input_desc == nullptr ? nullptr : CreateZeroTensor(input_desc->tensor);
        if (tensor == nullptr || static_graph_.SetTensor(input_name, tensor) != 0) {
            last_error_ = "failed to allocate Qwen input: " + input_name;
            return -1;
        }
    }
    if (PrepareExecutableGraph() != 0) {
        last_error_ = "failed to build Qwen graph for backend " + std::string(QwenBackendName(backend));
        return -1;
    }
    if (Reset() != 0) {
        return -1;
    }
    std::ostringstream summary;
    summary << "model=" << model.name << " backend=" << QwenBackendName(backend) << " max_context=" << max_context_
            << " static_nodes=" << static_graph_.NodeSize() << " runtime_nodes=" << runtime_graph_.NodeSize();
    last_build_summary_ = summary.str();
    return 0;
}

int32_t QwenRunner::PrepareExecutableGraph() {
    if (static_graph_.Build() != 0 || static_graph_.ApplyPasses() != 0) {
        return -1;
    }
    runtime_graph_.Clear();
    runtime_graph_.SetThreadMode(RuntimeThreadMode::kSerialGraph);
    return lowering_.Lower(static_graph_, &runtime_graph_);
}

int32_t QwenRunner::Reset() {
    if (runtime_graph_.NodeSize() == 0) {
        last_error_ = "Qwen runner is not loaded";
        return -1;
    }
#ifdef FEATHER_WITH_CUDA
    if (backend_device_ == DeviceType::CUDA) {
        for (const auto& binding : states_) {
            auto input = runtime_graph_.GetTensor(binding.input_name);
            auto output = runtime_graph_.GetTensor(binding.output_name);
            if (input != nullptr) {
                kernel::cuda_detail::MarkTensorDevicePersistent(input.get(), true);
            }
            if (output != nullptr) {
                kernel::cuda_detail::MarkTensorDevicePersistent(output.get(), true);
            }
            const auto producer_name = static_graph_.GetProducer(binding.output_name);
            const auto* producer = runtime_graph_.GetNode(producer_name);
            if (producer != nullptr && producer->op_type == "Identity" && producer->inputs.size() == 1) {
                auto identity_input = runtime_graph_.GetTensor(producer->inputs[0]);
                if (identity_input != nullptr) {
                    kernel::cuda_detail::MarkTensorDevicePersistent(identity_input.get(), true);
                }
            }
        }
    }
#endif
    for (const auto& input_name : loader_.model().graph.inputs) {
        auto tensor = runtime_graph_.GetTensor(input_name);
        if (tensor == nullptr || !tensor->IsInitialized()) {
            last_error_ = "Qwen runtime input is missing: " + input_name;
            return -1;
        }
        const size_t bytes = TensorBytes(*tensor);
        if (bytes == 0) {
            last_error_ = "Qwen runtime input has invalid size: " + input_name;
            return -1;
        }
        std::memset(tensor->raw_data(), 0, bytes);
#ifdef FEATHER_WITH_CUDA
        if (backend_device_ == DeviceType::CUDA) {
            kernel::cuda_detail::InvalidateTensorDevice(tensor.get());
        }
#endif
    }
    tokens_processed_ = 0;
    last_run_summary_.clear();
    return 0;
}

int32_t QwenRunner::SetDecodeInputs(int64_t token_id) {
    auto token_ids = runtime_graph_.GetTensor(token_input_name_);
    auto position = runtime_graph_.GetTensor(position_input_name_);
    auto mask = runtime_graph_.GetTensor(attention_mask_input_name_);
    if (token_ids == nullptr || position == nullptr || mask == nullptr || token_ids->numel() != 1 || position->numel() != 1 ||
        token_ids->data_type() != DataType::INT64 || position->data_type() != DataType::INT64 ||
        mask->data_type() != DataType::BF16 || mask->numel() != max_context_) {
        last_error_ = "Qwen runtime decode inputs have unexpected shape or type";
        return -1;
    }
    token_ids->mutable_data<int64_t>()[0] = token_id;
    position->mutable_data<int64_t>()[0] = tokens_processed_;
    auto* mask_values = mask->mutable_data<BFloat16>();
    const uint16_t negative_infinity = FloatToBFloat16(-std::numeric_limits<float>::infinity());
    const uint16_t zero = FloatToBFloat16(0.0f);
    const int64_t valid_positions = std::min(tokens_processed_, max_context_ - 1) + 1;
    for (int64_t index = 0; index < max_context_; ++index) {
        mask_values[index].bits = index >= max_context_ - valid_positions ? zero : negative_infinity;
    }
#ifdef FEATHER_WITH_CUDA
    if (backend_device_ == DeviceType::CUDA) {
        kernel::cuda_detail::InvalidateTensorDevice(token_ids.get());
        kernel::cuda_detail::InvalidateTensorDevice(position.get());
        kernel::cuda_detail::InvalidateTensorDevice(mask.get());
    }
#endif
    return 0;
}

int32_t QwenRunner::CopyState(const StateBinding& binding) {
    auto input = runtime_graph_.GetTensor(binding.input_name);
    auto output = runtime_graph_.GetTensor(binding.output_name);
    if (input == nullptr || output == nullptr || input->data_type() != output->data_type() || !input->IsInitialized() ||
        !output->IsInitialized()) {
        last_error_ = "Qwen state tensor is unavailable: " + binding.input_name;
        return -1;
    }
    if (binding.cache_axis < 0) {
        if (input->dims() != output->dims() || TensorBytes(*input) != TensorBytes(*output)) {
            last_error_ = "Qwen state shape changed unexpectedly: " + binding.input_name;
            return -1;
        }
#ifdef FEATHER_WITH_CUDA
        if (backend_device_ != DeviceType::CUDA) {
#else
        {
#endif
            // Host Identity nodes are storage views. When a next-state graph
            // output is such a view, swapping the output itself would leave
            // the Identity input aliasing the following token's state input.
            // Swap its source buffer instead so producer and consumer remain
            // on opposite sides of the recurrent ping-pong pair.
            const auto producer_name = static_graph_.GetProducer(binding.output_name);
            const auto* producer = runtime_graph_.GetNode(producer_name);
            if (producer != nullptr && producer->op_type == "Identity" && producer->inputs.size() == 1) {
                auto identity_input = runtime_graph_.GetTensor(producer->inputs[0]);
                if (identity_input != nullptr && identity_input->IsInitialized() &&
                    identity_input->data_type() == input->data_type() && identity_input->dims() == input->dims() &&
                    TensorBytes(*identity_input) == TensorBytes(*input)) {
                    // An Identity of the state itself is already persistent;
                    // self-swapping it is unnecessary.
                    if (identity_input.get() == input.get() || identity_input->raw_data() == input->raw_data()) {
                        return 0;
                    }
                    input->SwapStorage(*identity_input);
                    return 0;
                }
            }
            // Recurrent and convolution states have a same-shaped next-state
            // tensor. Ping-pong the two buffers so the following token reads
            // the new state without copying it back into the graph input.
            input->SwapStorage(*output);
            return 0;
        }
#ifdef FEATHER_WITH_CUDA
        const auto producer_name = static_graph_.GetProducer(binding.output_name);
        const auto* producer = runtime_graph_.GetNode(producer_name);
        if (backend_device_ == DeviceType::CUDA) {
            std::shared_ptr<Tensor> swap_target = output;
            if (producer != nullptr && producer->op_type == "Identity" && producer->inputs.size() == 1) {
                auto identity_input = runtime_graph_.GetTensor(producer->inputs[0]);
                if (identity_input != nullptr && identity_input->IsInitialized() &&
                    identity_input->data_type() == input->data_type() && identity_input->dims() == input->dims() &&
                    TensorBytes(*identity_input) == TensorBytes(*input)) {
                    if (identity_input.get() == input.get()) {
                        return 0;
                    }
                    swap_target = identity_input;
                }
            }
            if (kernel::cuda_detail::SwapTensorDeviceStorage(input.get(), swap_target.get()) != 0) {
                last_error_ = "Qwen CUDA state device swap failed: " + binding.input_name;
                return -1;
            }
            return 0;
        }
#endif
        std::memcpy(input->raw_data(), output->raw_data(), TensorBytes(*input));
#ifdef FEATHER_WITH_CUDA
        if (backend_device_ == DeviceType::CUDA) kernel::cuda_detail::InvalidateTensorDevice(input.get());
#endif
        return 0;
    }
    const auto& input_dims = input->dims().data();
    const auto& output_dims = output->dims().data();
    const size_t axis = static_cast<size_t>(binding.cache_axis);
    if (axis >= input_dims.size() || input_dims.size() != output_dims.size() || input_dims[axis] <= 1 ||
        output_dims[axis] != 1) {
        last_error_ = "Qwen cache state shape is invalid: " + binding.input_name;
        return -1;
    }
#ifdef FEATHER_WITH_CUDA
    if (backend_device_ == DeviceType::CUDA) {
        if (kernel::cuda_detail::AppendTensorStateOnDevice(input.get(), output.get(), binding.cache_axis) != 0) {
            last_error_ = "Qwen CUDA cache state append failed: " + binding.input_name;
            return -1;
        }
        return 0;
    }
#endif
    int64_t outer = 1;
    int64_t inner = 1;
    for (size_t index = 0; index < axis; ++index) outer *= input_dims[index];
    for (size_t index = axis + 1; index < input_dims.size(); ++index) inner *= input_dims[index];
    const size_t element_bytes = DataTypeBytes(input->data_type());
    const size_t block_bytes = static_cast<size_t>(inner) * element_bytes;
    const size_t input_slots = static_cast<size_t>(input_dims[axis]);
    auto* destination = static_cast<uint8_t*>(input->raw_data());
    const auto* source = static_cast<const uint8_t*>(output->raw_data());
    for (int64_t index = 0; index < outer; ++index) {
        auto* cache = destination + static_cast<size_t>(index) * input_slots * block_bytes;
        std::memmove(cache, cache + block_bytes, (input_slots - 1) * block_bytes);
        std::memcpy(cache + (input_slots - 1) * block_bytes, source + static_cast<size_t>(index) * block_bytes,
                    block_bytes);
    }
#ifdef FEATHER_WITH_CUDA
    if (backend_device_ == DeviceType::CUDA) {
        kernel::cuda_detail::InvalidateTensorDevice(input.get());
    }
#endif
    return 0;
}

int32_t QwenRunner::UpdateStates() {
    for (const auto& binding : states_) {
        if (CopyState(binding) != 0) {
            return -1;
        }
    }
    return 0;
}

int64_t QwenRunner::SelectGreedyToken() const {
    const auto logits = runtime_graph_.GetTensor(logits_output_name_);
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

int32_t QwenRunner::RunToken(int64_t token_id, int64_t* next_token_id) {
    if (next_token_id == nullptr || SetDecodeInputs(token_id) != 0) {
        return -1;
    }
    const auto begin = std::chrono::steady_clock::now();
    int32_t run_status = 0;
#ifdef FEATHER_WITH_CUDA
    if (backend_device_ == DeviceType::CUDA) {
        {
            kernel::cuda_detail::DeferredHostSyncScope deferred_host_sync;
            run_status = runtime_graph_.Run();
        }
        if (run_status == 0 && kernel::cuda_detail::SynchronizeInferenceStream() != 0) {
            run_status = -1;
        }
        if (run_status == 0) {
            run_status = SyncTensorFromCuda(runtime_graph_.GetTensor(logits_output_name_));
        }
    } else
#endif
    {
        run_status = runtime_graph_.Run();
    }
    if (run_status != 0) {
        last_error_ = "Qwen runtime graph execution failed";
#ifdef FEATHER_WITH_CUDA
        if (backend_device_ == DeviceType::CUDA) {
            last_error_ += ": " + kernel::cuda_detail::CudaLastErrorMessage();
        }
#endif
        return -1;
    }
    if (UpdateStates() != 0) {
        return -1;
    }
    *next_token_id = SelectGreedyToken();
    if (*next_token_id < 0) {
        last_error_ = "Qwen logits are unavailable";
        return -1;
    }
    ++tokens_processed_;
    const auto end = std::chrono::steady_clock::now();
    last_run_summary_ = "token=" + std::to_string(token_id) + " next=" + std::to_string(*next_token_id) +
                        " decode_ms=" + std::to_string(std::chrono::duration<double, std::milli>(end - begin).count());
    return 0;
}

int32_t QwenRunner::GenerateImpl(const std::vector<int64_t>& prompt_tokens, int max_new_tokens,
                                 const std::vector<int64_t>& stop_token_ids,
                                 const std::function<void(int64_t)>& on_token,
                                 std::vector<int64_t>* generated_tokens) {
    last_error_.clear();
    if (prompt_tokens.empty() || max_new_tokens <= 0 || (!on_token && generated_tokens == nullptr)) {
        last_error_ = "Qwen prompt and max_new_tokens must be non-empty and positive";
        return -1;
    }
    int64_t next_token = -1;
    for (const auto token : prompt_tokens) {
        if (RunToken(token, &next_token) != 0) {
            return -1;
        }
    }
    if (generated_tokens != nullptr) {
        generated_tokens->clear();
    }
    const std::unordered_set<int64_t> stop_ids(stop_token_ids.begin(), stop_token_ids.end());
    for (int index = 0; index < max_new_tokens; ++index) {
        const int64_t generated_token = next_token;
        if (generated_tokens != nullptr) {
            generated_tokens->push_back(generated_token);
        }
        if (on_token) {
            on_token(generated_token);
        }
        if (RunToken(next_token, &next_token) != 0) {
            return -1;
        }
        if (stop_ids.count(generated_token) != 0) {
            break;
        }
    }
    return 0;
}

int32_t QwenRunner::GenerateStream(const std::vector<int64_t>& prompt_tokens, int max_new_tokens,
                                   const std::vector<int64_t>& stop_token_ids,
                                   const std::function<void(int64_t)>& on_token) {
    if (!on_token) {
        last_error_ = "Qwen streaming callback must not be empty";
        return -1;
    }
    return GenerateImpl(prompt_tokens, max_new_tokens, stop_token_ids, on_token, nullptr);
}

int32_t QwenRunner::Generate(const std::vector<int64_t>& prompt_tokens, int max_new_tokens,
                              const std::vector<int64_t>& stop_token_ids, std::vector<int64_t>* generated_tokens) {
    if (generated_tokens == nullptr) {
        last_error_ = "Qwen prompt and max_new_tokens must be non-empty and positive";
        return -1;
    }
    return GenerateImpl(prompt_tokens, max_new_tokens, stop_token_ids, std::function<void(int64_t)>(),
                        generated_tokens);
}

int32_t QwenRunner::Consume(const std::vector<int64_t>& token_ids) {
    last_error_.clear();
    if (token_ids.empty()) {
        last_error_ = "Qwen tokens to consume must be non-empty";
        return -1;
    }
    int64_t ignored_next_token = -1;
    for (const auto token : token_ids) {
        if (RunToken(token, &ignored_next_token) != 0) {
            return -1;
        }
    }
    return 0;
}

}  // namespace demo
}  // namespace feather
