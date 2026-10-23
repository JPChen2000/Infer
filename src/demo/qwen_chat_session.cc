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

#include "demo/qwen_chat_session.h"

#include <algorithm>
#include <utility>

#include "demo/qwen_runner.h"

namespace feather {
namespace demo {

QwenChatHistory::QwenChatHistory(std::vector<int64_t> system_tokens)
    : system_tokens_(std::move(system_tokens)) {}

QwenChatPlan QwenChatHistory::PlanTurn(const std::vector<int64_t>& prompt_tokens,
                                      int64_t context_size, int max_new_tokens,
                                      int64_t closure_tokens) const {
    QwenChatPlan plan;
    if (prompt_tokens.empty() || context_size <= 0 || closure_tokens < 0 ||
        (max_new_tokens <= 0 && max_new_tokens != -1)) {
        plan.error = "Qwen chat requires a non-empty prompt, positive context, and a positive generation budget or -1";
        return plan;
    }
    // Subtract each required component before computing a generation reserve.
    // Reject the new prompt before selecting or dropping any stored turn.
    int64_t available = context_size;
    for (const auto count : {static_cast<int64_t>(system_tokens_.size()),
                             static_cast<int64_t>(prompt_tokens.size()), closure_tokens}) {
        if (count > available) {
            plan.error = "Qwen system and current user prompt exceed context capacity including turn closure";
            return plan;
        }
        available -= count;
    }
    if (available == 0) {
        plan.error = "Qwen system and current user prompt leave no context for generation after turn closure";
        return plan;
    }
    plan.reserved_generation_tokens = max_new_tokens == -1 ? 1 : std::min<int64_t>(max_new_tokens, available);
    int64_t retained_tokens = TokenCount() - static_cast<int64_t>(system_tokens_.size());
    const int64_t history_capacity = available - plan.reserved_generation_tokens;
    while (retained_tokens > history_capacity && plan.dropped_turns < turns_.size()) {
        retained_tokens -= static_cast<int64_t>(turns_[plan.dropped_turns].size());
        ++plan.dropped_turns;
    }
    plan.replay_tokens = system_tokens_;
    for (size_t index = plan.dropped_turns; index < turns_.size(); ++index) {
        plan.replay_tokens.insert(plan.replay_tokens.end(), turns_[index].begin(), turns_[index].end());
    }
    plan.valid = true;
    return plan;
}

void QwenChatHistory::ApplyPlan(const QwenChatPlan& plan) {
    if (plan.valid && plan.dropped_turns <= turns_.size()) {
        turns_.erase(turns_.begin(), turns_.begin() + plan.dropped_turns);
    }
}

void QwenChatHistory::AppendTurn(std::vector<int64_t> consumed_tokens) {
    turns_.push_back(std::move(consumed_tokens));
}

std::vector<int64_t> QwenChatHistory::ReplayTokens() const {
    std::vector<int64_t> replay = system_tokens_;
    for (const auto& turn : turns_) {
        replay.insert(replay.end(), turn.begin(), turn.end());
    }
    return replay;
}

int64_t QwenChatHistory::TokenCount() const {
    int64_t count = static_cast<int64_t>(system_tokens_.size());
    for (const auto& turn : turns_) {
        count += static_cast<int64_t>(turn.size());
    }
    return count;
}

QwenChatSession::QwenChatSession(QwenRunner* runner, std::vector<int64_t> system_tokens,
                               int64_t end_of_turn_token, std::vector<int64_t> newline_tokens,
                               std::vector<int64_t> stop_tokens)
    : runner_(runner), history_(std::move(system_tokens)), end_of_turn_token_(end_of_turn_token),
      newline_tokens_(std::move(newline_tokens)), stop_tokens_(std::move(stop_tokens)) {}

QwenFinishReason QwenChatSession::LastFinishReason() const { return last_finish_reason_; }

int32_t QwenChatSession::GenerateTurn(const std::vector<int64_t>& prompt_tokens,
                                     const QwenGenerationOptions& options,
                                     const std::function<void(int64_t)>& on_token) {
    last_error_.clear();
    last_dropped_turns_ = 0;
    last_finish_reason_ = QwenFinishReason::kNone;
    if (runner_ == nullptr || end_of_turn_token_ < 0 || newline_tokens_.empty() ||
        options.context_reserve_tokens < 0) {
        last_error_ = "Qwen chat requires a runner and valid turn closure tokens and reserve";
        return -1;
    }
    const int64_t closure_tokens = 1 + static_cast<int64_t>(newline_tokens_.size());
    const int64_t reserve_tokens = std::max(closure_tokens, options.context_reserve_tokens);
    // Validate the new turn before replacing another chat's live state. An
    // automatic session can grow up to the model limit before trimming history.
    const auto preliminary_plan = history_.PlanTurn(prompt_tokens,
        runner_->AutoContextGrowthEnabled() ? runner_->ModelContextLimit() : runner_->MaxContext(),
        options.max_new_tokens, reserve_tokens);
    if (!preliminary_plan.valid) {
        last_error_ = preliminary_plan.error;
        return -1;
    }
    needs_rebuild_ = needs_rebuild_ || runner_state_revision_ != runner_->StateRevision();
    bool reset_for_rebuild = false;
    if (needs_rebuild_) {
        // External operations may have failed after partially writing state.
        // Reset before reserving capacity, which requires valid source state.
        if (runner_->Reset() != 0) {
            last_error_ = runner_->LastError();
            return -1;
        }
        reset_for_rebuild = true;
    }
    if (runner_->AutoContextGrowthEnabled() && !prompt_tokens.empty() &&
        (options.max_new_tokens > 0 || options.max_new_tokens == -1)) {
        // Reserve for the complete live history before deciding whether any
        // whole turns must be replayed. Saturation also bounds large budgets.
        const int64_t limit = runner_->ModelContextLimit();
        int64_t required = 0;
        for (const int64_t count : {history_.TokenCount(), static_cast<int64_t>(prompt_tokens.size()),
                                     reserve_tokens, options.max_new_tokens == -1 ? int64_t{1} :
                                         static_cast<int64_t>(options.max_new_tokens)}) {
            required += std::min(count, limit - required);
        }
        if (runner_->ReserveContext(required) != 0) {
            last_error_ = runner_->LastError();
            return -1;
        }
    }
    const auto plan = history_.PlanTurn(prompt_tokens, runner_->MaxContext(), options.max_new_tokens, reserve_tokens);
    if (!plan.valid) {
        last_error_ = plan.error;
        return -1;
    }
    const auto restore_history = [&]() {
        needs_rebuild_ = true;
        if (runner_->Reset() != 0) {
            return false;
        }
        // ApplyPlan runs only after a successful turn, so this still includes
        // every complete turn provisionally dropped for the failed attempt.
        const auto original = history_.ReplayTokens();
        if (!original.empty() && runner_->Consume(original) != 0) {
            return false;
        }
        needs_rebuild_ = false;
        runner_state_revision_ = runner_->StateRevision();
        return true;
    };
    const auto runner_failed = [&]() {
        const std::string original_error = runner_->LastError();
        if (!restore_history()) {
            last_error_ = original_error + "; history recovery failed: " + runner_->LastError();
        } else {
            last_error_ = original_error;
        }
        return -1;
    };
    if (plan.dropped_turns != 0 || needs_rebuild_) {
        // Reset clears KV, convolution and recurrent states as one operation.
        needs_rebuild_ = true;
        if (!reset_for_rebuild && runner_->Reset() != 0) {
            return runner_failed();
        }
        if (!plan.replay_tokens.empty() && runner_->Consume(plan.replay_tokens) != 0) {
            return runner_failed();
        }
        needs_rebuild_ = false;
        runner_state_revision_ = runner_->StateRevision();
    } else if (runner_->TokensProcessed() == 0 && !plan.replay_tokens.empty()) {
        // The system prefix is consumed once; later turns reuse the live state.
        needs_rebuild_ = true;
        if (runner_->Consume(plan.replay_tokens) != 0) {
            return runner_failed();
        }
        needs_rebuild_ = false;
        runner_state_revision_ = runner_->StateRevision();
    }
    const int64_t before_prompt = runner_->TokensProcessed();
    QwenGenerationOptions generation_options = options;
    generation_options.context_reserve_tokens = reserve_tokens;
    std::vector<int64_t> generated;
    if (runner_->GenerateStream(prompt_tokens, generation_options, stop_tokens_, [&](int64_t token) {
            generated.push_back(token);
            if (on_token) {
                on_token(token);
            }
        }) != 0) {
        return runner_failed();
    }
    last_finish_reason_ = runner_->LastFinishReason();
    if (runner_->TokensProcessed() - before_prompt < static_cast<int64_t>(prompt_tokens.size())) {
        // Cancellation during prompt ingestion leaves no complete new turn.
        // Restore the original history, including any turns provisionally dropped.
        if (!restore_history()) {
            last_error_ = "Qwen history recovery failed after prompt cancellation: " + runner_->LastError();
            return -1;
        }
        return 0;
    }
    const bool ended_with_stop = !generated.empty() &&
        std::find(stop_tokens_.begin(), stop_tokens_.end(), generated.back()) != stop_tokens_.end();
    std::vector<int64_t> closure;
    if (!ended_with_stop) {
        closure.push_back(end_of_turn_token_);
    }
    closure.insert(closure.end(), newline_tokens_.begin(), newline_tokens_.end());
    if (runner_->Consume(closure) != 0) {
        return runner_failed();
    }
    std::vector<int64_t> complete_turn = prompt_tokens;
    complete_turn.insert(complete_turn.end(), generated.begin(), generated.end());
    complete_turn.insert(complete_turn.end(), closure.begin(), closure.end());
    history_.ApplyPlan(plan);
    history_.AppendTurn(std::move(complete_turn));
    runner_state_revision_ = runner_->StateRevision();
    last_dropped_turns_ = plan.dropped_turns;
    return 0;
}

}  // namespace demo
}  // namespace feather
