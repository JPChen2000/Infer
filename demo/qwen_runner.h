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

#ifndef FEATHER_DEMO_QWEN_RUNNER_H
#define FEATHER_DEMO_QWEN_RUNNER_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <map>
#include <string>
#include <vector>

#include "core/graph.h"
#include "core/execution_plan.h"
#include "core/static_graph.h"
#include "core/stateful_session.h"
#include "model/model_io.h"

#ifdef FEATHER_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

namespace feather {
namespace demo {

enum class QwenBackend {
    kHost,
    kCommon,
    kX86,
    kCuda,
};

bool ParseQwenBackend(const std::string& value, QwenBackend* backend);
const char* QwenBackendName(QwenBackend backend);

enum class QwenFinishReason { kNone, kEos, kLength, kContextLimit, kCancelled };
const char* QwenFinishReasonName(QwenFinishReason reason);

struct QwenGenerationOptions {
    // -1 removes the token budget; context capacity still applies.
    int max_new_tokens{512};
    // Leave these slots available for the caller to Consume turn terminators.
    int64_t context_reserve_tokens{0};
    // Checked before each prompt chunk or generated token. Already emitted tokens
    // are consumed even if the streaming callback requests cancellation.
    std::function<bool()> should_cancel;
};

struct QwenLoadOptions {
    int64_t context_size{0};
    // Resident BF16 copies used only by CUDA FP8 fallback. Zero disables them.
    size_t fp8_weight_cache_bytes{256 * 1024 * 1024};
    // Offline scoring needs the complete floating-point distribution.
    bool retain_logits{false};
    // Explicit diagnostic outputs retained through graph optimization/lowering.
    std::vector<std::string> additional_outputs;
};

namespace internal {
struct QwenModelResources;
struct QwenDecodeState;
class QwenExecutionPlan;
}

// Calls on one runner must be serialized. Callbacks may inspect the runner;
// nested operations that change it return an error without changing the active
// operation's state or diagnostics (the void profiling setter is ignored).
class QwenRunner {
   public:
    QwenRunner();
    ~QwenRunner();
    // Start a fresh request from the same immutable model/mappings and policy.
    // The source request and model files may be released after this returns.
    int32_t CreateRequest(std::unique_ptr<QwenRunner>* request) const;
    bool SharesModelResourcesWith(const QwenRunner& other) const;

    int32_t Load(const std::string& model_path, QwenBackend backend = QwenBackend::kCommon,
                 int64_t context_size = 0);
    int32_t Load(const std::string& model_path, QwenBackend backend, const QwenLoadOptions& options);
    int32_t EvaluateToken(int64_t token_id, std::vector<float>* logits);
    // Only declared outputs of the single-token plan are readable. Call after
    // EvaluateToken, before another execution; this is an offline diagnostic API.
    int32_t ReadOutput(const std::string& name, std::vector<float>* values);
    int32_t Reset();
    int32_t ReserveContext(int64_t capacity);
    bool AutoContextGrowthEnabled() const { return auto_context_growth_; }
    size_t CachedPlanCount() const;
    int64_t ContextGrowthCount() const { return context_growth_count_; }
    int32_t Consume(const std::vector<int64_t>& token_ids);
    // Calls the callback before feeding each selected token back into the decoder.
    // Stop tokens are included so the existing state-update semantics are preserved.
    // Callback exceptions become errors; a throwing token callback's token is
    // not consumed. Previously consumed prompt/tokens remain until Reset.
    int32_t GenerateStream(const std::vector<int64_t>& prompt_tokens, int max_new_tokens,
                           const std::vector<int64_t>& stop_token_ids,
                           const std::function<void(int64_t)>& on_token);
    int32_t Generate(const std::vector<int64_t>& prompt_tokens, int max_new_tokens,
                     const std::vector<int64_t>& stop_token_ids, std::vector<int64_t>* generated_tokens);
    int32_t GenerateStream(const std::vector<int64_t>& prompt_tokens, const QwenGenerationOptions& options,
                           const std::vector<int64_t>& stop_token_ids,
                           const std::function<void(int64_t)>& on_token);
    int32_t Generate(const std::vector<int64_t>& prompt_tokens, const QwenGenerationOptions& options,
                     const std::vector<int64_t>& stop_token_ids, std::vector<int64_t>* generated_tokens);

    int64_t TokensProcessed() const;
    // Changes whenever a public operation may reset or mutate decoder state,
    // including failed executions. Capacity-only growth preserves this revision.
    uint64_t StateRevision() const;
    int64_t MaxContext() const;
    int64_t ModelContextLimit() const;
    int64_t PrefillChunkSize() const;
    int64_t LastPromptTokensConsumed() const { return last_prompt_tokens_consumed_; }
    double LastPrefillMilliseconds() const { return last_prefill_ms_; }
    QwenFinishReason LastFinishReason() const { return last_finish_reason_; }
    const std::string& LastError() const { return last_error_; }
    const std::string& DescribeLastBuild() const { return last_build_summary_; }
    const std::string& DescribeLastRun() const { return last_run_summary_; }
    int64_t CudaGraphLaunchCount() const;
    void SetRuntimeProfilingEnabled(bool enabled);
    const std::vector<RuntimeProfileSummary>& RuntimeProfileSummaries() const;

   private:
    class OperationGuard {
       public:
        explicit OperationGuard(const QwenRunner& runner)
            : runner_(runner), acquired_(!runner.operation_active_) {
            if (acquired_) runner_.operation_active_ = true;
        }
        ~OperationGuard() { if (acquired_) runner_.operation_active_ = false; }
        explicit operator bool() const { return acquired_; }
        OperationGuard(const OperationGuard&) = delete;
        OperationGuard& operator=(const OperationGuard&) = delete;
       private:
        const QwenRunner& runner_;
        bool acquired_;
    };
    int32_t ReserveContextInternal(int64_t capacity);
    int32_t ReadOutputInternal(const std::string& name, std::vector<float>* values);
    int32_t CheckCancellation(const std::function<bool()>& callback);
    int32_t InvokeTokenCallback(const std::function<void(int64_t)>& callback, int64_t token);
    void UnloadExecution();
    int32_t LoadInternal(const std::string& model_path, QwenBackend backend, int64_t context_size,
                         std::shared_ptr<const internal::QwenModelResources> resources = nullptr,
                         const internal::QwenExecutionPlan* weight_owner = nullptr);
    int32_t RunPrompt(const std::vector<int64_t>& token_ids, const std::function<bool()>& should_cancel,
                      int64_t* next_token_id);
    void RefreshRunSummary();
    internal::QwenExecutionPlan* GetSequencePlan(int64_t sequence);
    int32_t EnsureContext(int64_t required);
    int32_t GenerateImpl(const std::vector<int64_t>& prompt_tokens, const QwenGenerationOptions& options,
                         const std::vector<int64_t>& stop_token_ids,
                         const std::function<void(int64_t)>& on_token,
                         std::vector<int64_t>* generated_tokens);
    int32_t RunToken(int64_t token_id, int64_t* next_token_id);
    RuntimeGraph& Runtime();
    const RuntimeGraph& Runtime() const;
    // Destruction order retires captures/sessions before request state/mappings.
    std::shared_ptr<const internal::QwenModelResources> resources_;
    std::unique_ptr<internal::QwenDecodeState> decode_;
    RuntimeGraph bootstrap_graph_;
    std::unique_ptr<internal::QwenExecutionPlan> decode_plan_;
    std::unique_ptr<internal::QwenExecutionPlan> prefill_plan_;
    std::map<int64_t, std::unique_ptr<internal::QwenExecutionPlan>> sequence_plans_;
    std::string model_path_;
    QwenLoadOptions load_options_;
    mutable bool operation_active_{false};
    bool diagnostic_outputs_ready_{false};
    bool auto_context_growth_{false};
    int64_t context_growth_count_{0};
    QwenBackend backend_{QwenBackend::kCommon};
    QwenFinishReason last_finish_reason_{QwenFinishReason::kNone};
    std::string last_error_, last_build_summary_, last_run_summary_, last_step_summary_;
    int64_t last_prompt_tokens_consumed_{0}, last_prefill_chunks_{0};
    double last_prefill_ms_{0.0};
};

}  // namespace demo
}  // namespace feather

#endif  // FEATHER_DEMO_QWEN_RUNNER_H
