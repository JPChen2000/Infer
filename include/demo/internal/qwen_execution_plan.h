// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#ifndef FEATHER_DEMO_INTERNAL_QWEN_EXECUTION_PLAN_H
#define FEATHER_DEMO_INTERNAL_QWEN_EXECUTION_PLAN_H
#include "demo/qwen_runner.h"
#include "demo/internal/qwen_model_resources.h"
#include "demo/internal/qwen_decode_state.h"
namespace feather { namespace demo { namespace internal {
// A shape-specific executable, with no generation, history, request creation,
// revision ownership, or child plan cache. State outlives every plan that binds it.
class QwenExecutionPlan {
 public:
    explicit QwenExecutionPlan(QwenDecodeState& state) : state_(state) {}
    ~QwenExecutionPlan();
    int32_t Build(std::shared_ptr<const QwenModelResources> resources,
                  const QwenModelSource& source, QwenBackend backend,
                  const QwenLoadOptions& options, int64_t context_size,
                  int64_t selected_sequence = 1, const QwenExecutionPlan* weight_owner = nullptr);
    int32_t Reset();
    void SetRuntimeProfilingEnabled(bool enabled);
    int32_t RunStep(const int64_t* token_ids, int64_t count, int64_t* next_token_id, bool read_logits = true);
    RuntimeGraph& Runtime() { return session_ ? session_->runtime() : bootstrap_graph_; }
    const RuntimeGraph& Runtime() const { return session_ ? session_->runtime() : bootstrap_graph_; }
    const model::ValueDesc* FindValueDesc(const std::string& name) const;
    int32_t ValidatePrefillCompanion(const QwenExecutionPlan& companion);
    int32_t PrepareExecutableGraph(const ExecutionSession* weight_source = nullptr);
    int32_t SetRuntimeInputs(const int64_t* token_ids, int64_t count);
    int32_t CopyState(const StateBinding& binding);
    int32_t UpdateStates();
    int64_t SelectGreedyToken() const;
    bool HasSequenceCapacity() const;
#ifdef FEATHER_WITH_CUDA
    struct CudaGraphSlot {
        std::unique_ptr<ExecutionCapture> executable;
        const void* logits_device{nullptr};
        size_t logits_bytes{};
    };
    int32_t CaptureCudaGraph(int slot);
    int32_t LaunchCudaGraph(int slot);
    int32_t SyncCudaGraphLogits(int slot);
    void ResetCudaGraphs();
#endif
#ifdef FEATHER_WITH_CUDA
    std::shared_ptr<kernel::cuda_detail::CudaExecutionContext> cuda_context_;
#endif
    QwenDecodeState& state_;
    // Declared first so mappings outlive sessions and their captures.
    std::shared_ptr<const QwenModelResources> resources_;
    model::ModelDesc model_;
    StaticGraph static_graph_;
    RuntimeGraph bootstrap_graph_;
    std::shared_ptr<const CompiledGraph> compiled_graph_;
    std::unique_ptr<ExecutionSession> session_;
    QwenLoadOptions load_options_;
    bool unified_execution_{false};
    int64_t prefill_limit_{1};
    std::vector<StateBinding> states_;
    std::vector<std::string> state_input_names_;
    std::vector<std::string> inplace_cache_names_;
    std::string token_input_name_, position_input_name_, attention_mask_input_name_, logits_output_name_;
    QwenBackend backend_{QwenBackend::kCommon};
    DeviceType backend_device_{DeviceType::COMMON};
    int64_t max_context_{0}, model_context_limit_{0};
    bool optimized_execution_{false};
    int64_t execution_sequence_{1};
    std::string weights_id_;
    int64_t steps_executed_{0}, cuda_graph_launch_count_{0};
    std::string last_error_, last_step_summary_;
#ifdef FEATHER_WITH_CUDA
    std::array<CudaGraphSlot, 2> cuda_graph_slots_{};
    int cuda_graph_slot_{0};
    bool cuda_graph_capture_disabled_{false};
#endif
};
} } }
#endif
