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

#ifndef FEATHER_DEMO_QWEN_CHAT_SESSION_H
#define FEATHER_DEMO_QWEN_CHAT_SESSION_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace feather {
namespace demo {

class QwenRunner;
struct QwenGenerationOptions;
enum class QwenFinishReason;

struct QwenChatPlan {
    bool valid{false};
    size_t dropped_turns{0};
    int64_t reserved_generation_tokens{0};
    std::vector<int64_t> replay_tokens;
    std::string error;
};

// Stores the exact consumed tokens of each complete user/assistant turn.
// Planning is read-only so an oversized prompt cannot discard live history.
class QwenChatHistory {
   public:
    explicit QwenChatHistory(std::vector<int64_t> system_tokens = {});
    QwenChatPlan PlanTurn(const std::vector<int64_t>& prompt_tokens, int64_t context_size,
                          int max_new_tokens, int64_t closure_tokens) const;
    void ApplyPlan(const QwenChatPlan& plan);
    void AppendTurn(std::vector<int64_t> consumed_tokens);
    std::vector<int64_t> ReplayTokens() const;
    size_t TurnCount() const { return turns_.size(); }
    int64_t TokenCount() const;

   private:
    std::vector<int64_t> system_tokens_;
    std::vector<std::vector<int64_t>> turns_;
};

// Couples exact token history to the runner's live state. Generation reserves
// one explicit EOT and the actual encoded newline, including multi-token ones.
// Sequential users of the same runner replay their own history after external
// mutations; calls sharing a runner must still be serialized by the caller.
class QwenChatSession {
   public:
    QwenChatSession(QwenRunner* runner, std::vector<int64_t> system_tokens,
                    int64_t end_of_turn_token, std::vector<int64_t> newline_tokens,
                    std::vector<int64_t> stop_tokens);
    int32_t GenerateTurn(const std::vector<int64_t>& prompt_tokens,
                         const QwenGenerationOptions& options,
                         const std::function<void(int64_t)>& on_token);
    QwenFinishReason LastFinishReason() const;
    const QwenChatHistory& History() const { return history_; }
    const std::string& LastError() const { return last_error_; }
    size_t LastDroppedTurns() const { return last_dropped_turns_; }
    bool LastRebuiltHistory() const { return last_dropped_turns_ != 0; }

   private:
    QwenRunner* runner_;
    QwenChatHistory history_;
    int64_t end_of_turn_token_;
    std::vector<int64_t> newline_tokens_;
    std::vector<int64_t> stop_tokens_;
    std::string last_error_;
    bool needs_rebuild_{true};
    uint64_t runner_state_revision_{0};
    size_t last_dropped_turns_{0};
    QwenFinishReason last_finish_reason_{};
};

}  // namespace demo
}  // namespace feather

#endif  // FEATHER_DEMO_QWEN_CHAT_SESSION_H
