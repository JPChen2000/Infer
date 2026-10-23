// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0

#include "demo/qwen_runner.h"
#include "demo/internal/qwen_execution_plan.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <utility>

#ifdef FEATHER_WITH_CUDA
#include "src/kernel/cuda/runtime.h"
#endif

namespace feather {
namespace demo {
namespace {
struct GrowthCopy {
    std::shared_ptr<Tensor> source, destination;
    size_t source_bytes{}, destination_bytes{};
    size_t rows{1}, width{}, source_pitch{}, destination_pitch{}, source_offset{}, destination_offset{};
#ifdef FEATHER_WITH_CUDA
    void* source_device{};
#endif
};

bool StorageReady(const Tensor& tensor, size_t* bytes) {
    for (const auto dim : tensor.dims().data()) if (dim <= 0) return false;
    return tensor.IsInitialized() && CheckedTensorBytes(tensor.dims().data(), tensor.data_type(), bytes) &&
           *bytes > 0 && tensor.memory_size() >= *bytes && tensor.storage() &&
           tensor.offset() <= tensor.storage()->size() && *bytes <= tensor.storage()->size() - tensor.offset();
}

bool PrepareCopy(GrowthCopy* copy, int axis, bool position_cache, int64_t tokens) {
    if (!copy->source || !copy->destination || copy->source->data_type() != copy->destination->data_type() ||
        copy->source->layout() != copy->destination->layout() ||
        !StorageReady(*copy->source, &copy->source_bytes) ||
        !StorageReady(*copy->destination, &copy->destination_bytes)) return false;
    const auto& source = copy->source->dims().data();
    const auto& destination = copy->destination->dims().data();
    if (axis < 0) {
        if (source != destination) return false;
        copy->width = copy->source_bytes;
        copy->source_pitch = copy->source_bytes;
        copy->destination_pitch = copy->destination_bytes;
        return true;
    }
    if (source.size() != destination.size() || static_cast<size_t>(axis) >= source.size() ||
        destination[axis] <= source[axis] || tokens < 0) return false;
    for (size_t i = 0; i < source.size(); ++i) {
        if (i != static_cast<size_t>(axis) && source[i] != destination[i]) return false;
    }
    if (!CheckedElementCount(std::vector<int64_t>(source.begin(), source.begin() + axis), &copy->rows)) return false;
    const size_t element_bytes = DataTypeBytes(copy->source->data_type());
    if (!element_bytes) return false;
    size_t inner_bytes = 0;
    if (!CheckedByteCount(std::vector<int64_t>(source.begin() + axis + 1, source.end()),
                          element_bytes, &inner_bytes)) return false;
    const size_t present = static_cast<size_t>(std::min(tokens, source[axis]));
    copy->width = present * inner_bytes;
    copy->source_pitch = static_cast<size_t>(source[axis]) * inner_bytes;
    copy->destination_pitch = static_cast<size_t>(destination[axis]) * inner_bytes;
    if (!position_cache) {
        copy->source_offset = copy->source_pitch - copy->width;
        copy->destination_offset = copy->destination_pitch - copy->width;
    }
    return true;
}
} // namespace

int32_t QwenRunner::ReserveContext(int64_t capacity) {
    OperationGuard operation(*this);
    if (!operation) return -1;
    return ReserveContextInternal(capacity);
}

int32_t QwenRunner::ReserveContextInternal(int64_t capacity) {
    const auto fail = [this](const std::string& reason) {
        last_error_ = "Qwen context growth failed: " + reason;
        return -1;
    };
    if (!decode_plan_ || !decode_plan_->session_ || Runtime().NodeSize() == 0) return fail("runner is not loaded");
    if (capacity <= 0 || capacity > decode_plan_->model_context_limit_) return fail("requested capacity is outside model limits");
    if (decode_->invalid) return fail("state is invalid after execution failure; call Reset before reuse");
    if (capacity <= decode_plan_->max_context_) { last_error_.clear(); return 0; }
    try {
        auto replacement = std::make_unique<QwenRunner>();
        replacement->load_options_ = load_options_;
        if (replacement->LoadInternal(model_path_, backend_, capacity, resources_, decode_plan_.get()) != 0)
            return fail(replacement->LastError());
        if (replacement->decode_plan_->max_context_ != capacity || replacement->decode_plan_->state_input_names_ != decode_plan_->state_input_names_)
            return fail("replacement has an incompatible state contract");
        replacement->SetRuntimeProfilingEnabled(Runtime().ProfilingEnabled());
        std::vector<GrowthCopy> copies;
        copies.reserve(decode_plan_->state_input_names_.size());
        for (const auto& name : decode_plan_->state_input_names_) {
            GrowthCopy copy;
            copy.source = Runtime().GetTensor(name);
            copy.destination = replacement->Runtime().GetTensor(name);
            const bool position_cache = std::find(decode_plan_->inplace_cache_names_.begin(), decode_plan_->inplace_cache_names_.end(), name) !=
                                        decode_plan_->inplace_cache_names_.end();
            int axis = position_cache ? 2 : -1;
            if (!position_cache) {
                const auto binding = std::find_if(decode_plan_->states_.begin(), decode_plan_->states_.end(), [&](const StateBinding& state) {
                    return state.input_name == name;
                });
                if (binding == decode_plan_->states_.end()) return fail("missing state binding for " + name);
                axis = binding->cache_axis;
            }
            if (!PrepareCopy(&copy, axis, position_cache, decode_->tokens_processed))
                return fail("incompatible state storage for " + name);
            copies.push_back(std::move(copy));
        }
#ifdef FEATHER_WITH_CUDA
        if (decode_plan_->backend_device_ == DeviceType::CUDA) {
            {
                kernel::cuda_detail::ExecutionContextScope source_scope(Runtime().CudaContext());
                kernel::cuda_detail::DeferredHostSyncScope deferred_sync;
                for (auto& copy : copies) {
                    if (kernel::cuda_detail::AcquireTensorDevice(copy.source.get(), copy.source_bytes,
                            copy.source->raw_data(), &copy.source_device) != 0)
                        return fail("source state acquisition failed");
                }
                if (kernel::cuda_detail::SynchronizeInferenceStream() != 0)
                    return fail("source state synchronization failed");
            }
            {
                kernel::cuda_detail::ExecutionContextScope destination_scope(replacement->Runtime().CudaContext());
                kernel::cuda_detail::DeferredHostSyncScope deferred_sync;
                for (auto& copy : copies) {
                    void* target = nullptr;
                    if (kernel::cuda_detail::AcquireOutputTensorDevice(copy.destination.get(), copy.destination_bytes,
                            &target) != 0 ||
                        cudaMemsetAsync(target, 0, copy.destination_bytes,
                                        kernel::cuda_detail::InferenceStream()) != cudaSuccess ||
                        (copy.width && cudaMemcpy2DAsync(static_cast<char*>(target) + copy.destination_offset,
                            copy.destination_pitch, static_cast<char*>(copy.source_device) + copy.source_offset,
                            copy.source_pitch, copy.width, copy.rows, cudaMemcpyDeviceToDevice,
                            kernel::cuda_detail::InferenceStream()) != cudaSuccess)) {
                        (void)kernel::cuda_detail::SynchronizeInferenceStream();
                        return fail("destination state copy failed");
                    }
                }
                if (kernel::cuda_detail::SynchronizeInferenceStream() != 0)
                    return fail("destination state synchronization failed");
            }
        } else
#endif
        {
            for (const auto& copy : copies) {
                auto* destination = static_cast<char*>(copy.destination->mutable_data(copy.destination_bytes));
                const auto* source = static_cast<const char*>(copy.source->raw_data());
                std::memset(destination, 0, copy.destination_bytes);
                if (copy.width) {
                    for (size_t row = 0; row < copy.rows; ++row)
                        std::memcpy(destination + copy.destination_offset + row * copy.destination_pitch,
                                    source + copy.source_offset + row * copy.source_pitch, copy.width);
                }
                replacement->Runtime().Backend().HostWritten(*copy.destination);
            }
        }
        // Commit the complete request allocation and every plan together. Each
        // plan's borrowed state reference follows its heap allocation across
        // this swap. Old captures retire before the old state is destroyed.
        replacement->decode_->tokens_processed = decode_->tokens_processed;
        replacement->decode_->revision = decode_->revision;
        replacement->decode_->invalid = decode_->invalid;
        using std::swap;
        swap(resources_, replacement->resources_);
        swap(decode_, replacement->decode_);
        swap(decode_plan_, replacement->decode_plan_);
        swap(prefill_plan_, replacement->prefill_plan_);
        swap(sequence_plans_, replacement->sequence_plans_);
        swap(last_build_summary_, replacement->last_build_summary_);
        diagnostic_outputs_ready_ = false;
        ++context_growth_count_;
        last_error_.clear();
        return 0;
    } catch (const std::exception& error) {
        return fail(error.what());
    }
}
} // namespace demo
} // namespace feather
