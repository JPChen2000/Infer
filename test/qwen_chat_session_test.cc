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

#include <gtest/gtest.h>

#include <filesystem>
#include <limits>
#include <memory>
#include <unordered_map>

#include "demo/qwen_runner.h"
#include "demo/qwen_demo.h"
#include "model/model_io.h"

#include "demo/qwen_chat_session.h"

namespace {

using feather::demo::QwenChatHistory;

TEST(qwen_chat_session_test, ReusesLiveHistoryWhenWholeTurnFits) {
    QwenChatHistory history({90, 91});
    history.AppendTurn({10, 11, 12, 13});
    const auto plan = history.PlanTurn({20, 21}, 16, 3, 2);
    ASSERT_TRUE(plan.valid) << plan.error;
    EXPECT_EQ(plan.dropped_turns, 0U);
    EXPECT_EQ(plan.reserved_generation_tokens, 3);
    EXPECT_EQ(history.TokenCount(), 6);
}

TEST(qwen_chat_session_test, DropsOnlyOldestCompleteTurnsAndKeepsSystem) {
    QwenChatHistory history({90, 91});
    history.AppendTurn({10, 11, 12, 13});
    history.AppendTurn({20, 21, 22, 23});
    history.AppendTurn({30, 31, 32, 33});
    const auto plan = history.PlanTurn({40, 41}, 15, 3, 2);
    ASSERT_TRUE(plan.valid) << plan.error;
    EXPECT_EQ(plan.dropped_turns, 2U);
    EXPECT_EQ(plan.replay_tokens, (std::vector<int64_t>{90, 91, 30, 31, 32, 33}));
    // The plan must not mutate history before the runner can accept it.
    EXPECT_EQ(history.TurnCount(), 3U);
    history.ApplyPlan(plan);
    EXPECT_EQ(history.TurnCount(), 1U);
    EXPECT_EQ(history.ReplayTokens(), (std::vector<int64_t>{90, 91, 30, 31, 32, 33}));
}

TEST(qwen_chat_session_test, ReplayPreservesExactAssistantPrefixAndClosureTokens) {
    QwenChatHistory history({90});
    // 11/12 represent the assistant think prefix; 99/13 are EOT/newline.
    history.AppendTurn({10, 11, 12, 80, 99, 13});
    history.AppendTurn({20, 11, 12, 81, 82, 99, 13});
    EXPECT_EQ(history.ReplayTokens(),
              (std::vector<int64_t>{90, 10, 11, 12, 80, 99, 13, 20, 11, 12, 81, 82, 99, 13}));
}

TEST(qwen_chat_session_test, OverlargePromptLeavesSystemAndHistoryUntouched) {
    QwenChatHistory history({90, 91, 92});
    history.AppendTurn({10, 11, 12});
    const auto plan = history.PlanTurn({20, 21, 22, 23, 24}, 10, 5, 2);
    EXPECT_FALSE(plan.valid);
    EXPECT_FALSE(plan.error.empty());
    EXPECT_EQ(history.TurnCount(), 1U);
    EXPECT_EQ(history.ReplayTokens(), (std::vector<int64_t>{90, 91, 92, 10, 11, 12}));
}

TEST(qwen_chat_session_test, CapsRequestedBudgetToSmallContextAvailableSpace) {
    QwenChatHistory history({90, 91});
    const auto plan = history.PlanTurn({10, 11, 12}, 12, 512, 2);
    ASSERT_TRUE(plan.valid) << plan.error;
    EXPECT_EQ(plan.reserved_generation_tokens, 5);
}

TEST(qwen_chat_session_test, UnlimitedGenerationRetainsNewestTurnsWithRoomForClosure) {
    QwenChatHistory history({90});
    history.AppendTurn({10, 11, 12, 13});
    history.AppendTurn({20, 21, 22, 23});
    const auto plan = history.PlanTurn({30, 31}, 10, -1, 2);
    ASSERT_TRUE(plan.valid) << plan.error;
    EXPECT_EQ(plan.reserved_generation_tokens, 1);
    EXPECT_EQ(plan.dropped_turns, 1U);
    EXPECT_EQ(plan.replay_tokens, (std::vector<int64_t>{90, 20, 21, 22, 23}));
}

TEST(qwen_chat_session_test, ReservesExactMultiTokenClosureSize) {
    QwenChatHistory history({90});
    const auto plan = history.PlanTurn({10, 11}, 9, 512, 4);
    ASSERT_TRUE(plan.valid) << plan.error;
    EXPECT_EQ(plan.reserved_generation_tokens, 2);
    EXPECT_FALSE(history.PlanTurn({10, 11}, 7, 512, 4).valid);
}

TEST(qwen_chat_session_test, RejectsInvalidBudgetsAndEmptyPrompts) {
    QwenChatHistory history({90});
    EXPECT_FALSE(history.PlanTurn({10}, 10, 0, 2).valid);
    EXPECT_FALSE(history.PlanTurn({10}, 10, -2, 2).valid);
    EXPECT_FALSE(history.PlanTurn({}, 10, 1, 2).valid);
    EXPECT_FALSE(history.PlanTurn({10}, 10, 1, -1).valid);
}

std::filesystem::path WriteChatSessionFixture(int64_t context_size, int64_t model_limit = 0) {
    feather::model::ModelDesc model;
    model.name = "qwen_chat_session_fixture";
    model.graph.inputs = {"token_ids", "position_id", "attention_mask", "recurrent_state_0"};
    model.graph.outputs = {"next_recurrent_state_0", "logits"};
    const auto value = [](const std::string& name, std::vector<int64_t> dims,
                          feather::DataType type, bool constant = false) {
        feather::model::ValueDesc result;
        result.tensor.name = name;
        result.tensor.dims = std::move(dims);
        result.tensor.data_type = type;
        result.constant = constant;
        return result;
    };
    model.graph.values = {
        value("token_ids", {1, 1}, feather::DataType::INT64),
        value("position_id", {1}, feather::DataType::INT64),
        value("attention_mask", {1, 1, 1, context_size}, feather::DataType::BF16),
        value("recurrent_state_0", {1}, feather::DataType::FP32),
        value("lookup", {4, 4}, feather::DataType::FP32, true),
        value("logits", {1, 1, 4}, feather::DataType::FP32),
        value("next_recurrent_state_0", {1}, feather::DataType::FP32),
    };
    feather::model::NodeDesc gather;
    gather.name = "lookup_logits";
    gather.op_type = "Gather";
    gather.inputs = {"lookup", "token_ids"};
    gather.outputs = {"logits"};
    gather.attributes["axis"] = int64_t{0};
    feather::model::NodeDesc state;
    state.name = "next_state";
    state.op_type = "Identity";
    state.inputs = {"recurrent_state_0"};
    state.outputs = {"next_recurrent_state_0"};
    model.graph.nodes = {gather, state};
    if (model_limit != 0) {
        model.metadata = {
            {"qwen.context.version", int64_t{1}},
            {"qwen.context.default", context_size},
            {"qwen.context.maximum", model_limit},
            {"qwen.context.value.attention_mask", std::vector<int64_t>{3, 0}},
        };
    }
    auto lookup = std::make_shared<feather::Tensor>();
    // 0 -> 1 -> 3 (EOS); closure tokens 2 and 0 remain valid inputs.
    lookup->Assign<float>({0, 5, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0, 0, 0, 0, 5}, {4, 4});
    const auto path = std::filesystem::temp_directory_path() /
                      ("qwen_chat_session_fixture_" + std::to_string(context_size) + "_" +
                       std::to_string(model_limit) + ".fth");
    feather::model::ModelWriter writer;
    EXPECT_TRUE(writer.Save(path.string(), model, {{"lookup", lookup}}));
    return path;
}

std::filesystem::path WriteStatefulChatFixture(int64_t context_size, int64_t model_limit = 0) {
    feather::model::ModelLoader loader;
    EXPECT_TRUE(loader.Load(WriteChatSessionFixture(context_size, model_limit).string()));
    auto model = loader.model();
    for (auto& value : model.graph.values) {
        if (value.tensor.name == "recurrent_state_0" || value.tensor.name == "next_recurrent_state_0")
            value.tensor.dims = {1, 1, 4};
    }
    auto raw_logits = model.graph.values[5];
    raw_logits.tensor.name = "raw_logits";
    model.graph.values.push_back(raw_logits);
    model.graph.nodes[0].outputs = {"raw_logits"};
    model.graph.nodes[1].op_type = "Add";
    model.graph.nodes[1].inputs = {"recurrent_state_0", "raw_logits"};
    feather::model::NodeDesc logits;
    logits.name = "state_logits";
    logits.op_type = "Identity";
    logits.inputs = {"next_recurrent_state_0"};
    logits.outputs = {"logits"};
    model.graph.nodes.push_back(logits);
    const auto path = std::filesystem::temp_directory_path() /
        ("qwen_stateful_chat_fixture_" + std::to_string(context_size) + "_" + std::to_string(model_limit) + ".fth");
    feather::model::ModelWriter writer;
    EXPECT_TRUE(writer.Save(path.string(), model, {{"lookup", loader.CreateWeightTensor("lookup")}}));
    return path;
}

TEST(qwen_chat_session_test, NewChatReplaysItsSystemOnPreviouslyUsedRunner) {
    const auto path = WriteStatefulChatFixture(32);
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(path.string()), 0);
    ASSERT_EQ(runner.Consume({1, 1}), 0);
    feather::demo::QwenChatSession session(&runner, {2}, 3, {0}, {3});
    feather::demo::QwenGenerationOptions options;
    options.max_new_tokens = 1;
    std::vector<int64_t> generated;
    ASSERT_EQ(session.GenerateTurn({0}, options, [&](int64_t t) { generated.push_back(t); }), 0);
    EXPECT_EQ(generated, (std::vector<int64_t>{1}));
    EXPECT_EQ(runner.TokensProcessed(), 5);
    EXPECT_EQ(session.History().ReplayTokens(), (std::vector<int64_t>{2, 0, 1, 3, 0}));
}

TEST(qwen_chat_session_test, AlternatingChatsReplayTheirOwnHistoriesAcrossGrowth) {
    const auto path = WriteStatefulChatFixture(8, 64);
    feather::demo::QwenRunner shared, isolated_a, isolated_b;
    ASSERT_EQ(shared.Load(path.string()), 0);
    ASSERT_EQ(isolated_a.Load(path.string()), 0);
    ASSERT_EQ(isolated_b.Load(path.string()), 0);
    feather::demo::QwenChatSession a(&shared, {2}, 3, {0}, {3});
    feather::demo::QwenChatSession b(&shared, {1}, 3, {0}, {3});
    feather::demo::QwenChatSession reference_a(&isolated_a, {2}, 3, {0}, {3});
    feather::demo::QwenChatSession reference_b(&isolated_b, {1}, 3, {0}, {3});
    feather::demo::QwenGenerationOptions options;
    options.max_new_tokens = 1;
    for (int turn = 0; turn != 3; ++turn) {
        for (auto pair : {std::make_pair(&a, &reference_a), std::make_pair(&b, &reference_b)}) {
            std::vector<int64_t> actual, expected;
            ASSERT_EQ(pair.first->GenerateTurn({0}, options, [&](int64_t t) { actual.push_back(t); }), 0)
                << pair.first->LastError();
            ASSERT_EQ(pair.second->GenerateTurn({0}, options, [&](int64_t t) { expected.push_back(t); }), 0);
            EXPECT_EQ(actual, expected);
            EXPECT_EQ(shared.TokensProcessed(), pair.first->History().TokenCount());
            EXPECT_EQ(pair.first->History().ReplayTokens(), pair.second->History().ReplayTokens());
        }
    }
    EXPECT_GT(shared.ContextGrowthCount(), 0);
}

TEST(qwen_chat_session_test, ExternalMutationAtSamePositionReplaysExactOwnedHistory) {
    const auto path = WriteStatefulChatFixture(32);
    feather::demo::QwenRunner runner, isolated;
    ASSERT_EQ(runner.Load(path.string()), 0);
    ASSERT_EQ(isolated.Load(path.string()), 0);
    feather::demo::QwenChatSession session(&runner, {2}, 3, {0}, {3});
    feather::demo::QwenChatSession reference(&isolated, {2}, 3, {0}, {3});
    feather::demo::QwenGenerationOptions options;
    options.max_new_tokens = 1;
    ASSERT_EQ(session.GenerateTurn({0}, options, {}), 0);
    ASSERT_EQ(reference.GenerateTurn({0}, options, {}), 0);
    ASSERT_EQ(runner.Reset(), 0);
    ASSERT_EQ(runner.Consume({3, 3, 3, 3, 3}), 0);
    ASSERT_EQ(runner.TokensProcessed(), session.History().TokenCount());
    std::vector<int64_t> actual, expected;
    ASSERT_EQ(session.GenerateTurn({0}, options, [&](int64_t t) { actual.push_back(t); }), 0);
    ASSERT_EQ(reference.GenerateTurn({0}, options, [&](int64_t t) { expected.push_back(t); }), 0);
    EXPECT_EQ(actual, expected);
    EXPECT_EQ(runner.TokensProcessed(), session.History().TokenCount());
}

TEST(qwen_chat_session_test, ExternalExecutionFailureRecoversHistoryBeforeAutomaticGrowth) {
    const auto path = WriteStatefulChatFixture(8, 32);
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(path.string()), 0);
    feather::demo::QwenChatSession session(&runner, {2}, 3, {0}, {3});
    feather::demo::QwenGenerationOptions options;
    options.max_new_tokens = 1;
    ASSERT_EQ(session.GenerateTurn({0}, options, {}), 0);
    ASSERT_NE(runner.Consume({99}), 0);
    ASSERT_EQ(session.GenerateTurn({0}, options, {}), 0) << session.LastError();
    EXPECT_EQ(runner.TokensProcessed(), session.History().TokenCount());
    EXPECT_EQ(session.History().TurnCount(), 2U);
    EXPECT_GT(runner.ContextGrowthCount(), 0);
}

TEST(qwen_chat_session_test, AutomaticGrowthKeepsCompleteHistoryBeforePlanningReplay) {
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(WriteChatSessionFixture(4, 32).string()), 0) << runner.LastError();
    ASSERT_TRUE(runner.AutoContextGrowthEnabled());
    feather::demo::QwenChatSession session(&runner, {2}, 3, {0}, {3});
    feather::demo::QwenGenerationOptions options;
    options.max_new_tokens = 1;
    ASSERT_EQ(session.GenerateTurn({0}, options, {}), 0) << session.LastError();
    EXPECT_GE(runner.MaxContext(), 5);
    ASSERT_EQ(session.GenerateTurn({0}, options, {}), 0) << session.LastError();
    EXPECT_GE(runner.MaxContext(), 9);
    EXPECT_GE(runner.ContextGrowthCount(), 1);
    EXPECT_FALSE(session.LastRebuiltHistory());
    EXPECT_EQ(session.LastDroppedTurns(), 0U);
    EXPECT_EQ(session.History().TurnCount(), 2U);
    EXPECT_EQ(session.History().ReplayTokens(), (std::vector<int64_t>{2, 0, 1, 3, 0, 0, 1, 3, 0}));
    EXPECT_EQ(runner.TokensProcessed(), 9);
}

TEST(qwen_chat_session_test, AutomaticGrowthReservesRequestedBudgetAndEncodedClosure) {
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(WriteChatSessionFixture(4, 32).string()), 0) << runner.LastError();
    feather::demo::QwenChatSession session(&runner, {2}, 3, {0, 2}, {3});
    feather::demo::QwenGenerationOptions options;
    options.max_new_tokens = 8;
    options.context_reserve_tokens = 5;
    ASSERT_EQ(session.GenerateTurn({0}, options, {}), 0) << session.LastError();
    EXPECT_GE(runner.MaxContext(), 15);  // system + prompt + reserve + generation
    EXPECT_EQ(session.LastFinishReason(), feather::demo::QwenFinishReason::kEos);
    EXPECT_EQ(session.History().ReplayTokens(), (std::vector<int64_t>{2, 0, 1, 3, 0, 2}));
}

TEST(qwen_chat_session_test, AutomaticGrowthWithUnlimitedBudgetReservesAtLeastOneToken) {
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(WriteChatSessionFixture(4, 32).string()), 0) << runner.LastError();
    feather::demo::QwenChatSession session(&runner, {2}, 3, {0}, {3});
    feather::demo::QwenGenerationOptions options;
    options.max_new_tokens = -1;
    ASSERT_EQ(session.GenerateTurn({0}, options, {}), 0) << session.LastError();
    EXPECT_GE(runner.MaxContext(), 5);
    EXPECT_EQ(session.History().TurnCount(), 1U);
    EXPECT_EQ(session.LastDroppedTurns(), 0U);
}

TEST(qwen_chat_session_test, AutomaticGrowthAtModelLimitKeepsWholeTurnReplayPolicy) {
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(WriteChatSessionFixture(4, 8).string()), 0) << runner.LastError();
    feather::demo::QwenChatSession session(&runner, {2}, 3, {0}, {3});
    feather::demo::QwenGenerationOptions options;
    options.max_new_tokens = 1;
    ASSERT_EQ(session.GenerateTurn({0}, options, {}), 0) << session.LastError();
    ASSERT_EQ(session.GenerateTurn({0}, options, {}), 0) << session.LastError();
    EXPECT_EQ(runner.MaxContext(), 8);
    EXPECT_EQ(session.LastDroppedTurns(), 1U);
    EXPECT_EQ(session.History().TurnCount(), 1U);
    EXPECT_EQ(session.History().ReplayTokens(), (std::vector<int64_t>{2, 0, 1, 3, 0}));
    EXPECT_EQ(runner.TokensProcessed(), 5);
}

TEST(qwen_chat_session_test, ExplicitCapacityRemainsHardBoundEvenWithGrowableModel) {
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(WriteChatSessionFixture(4, 32).string(),
                          feather::demo::QwenBackend::kCommon, 8), 0) << runner.LastError();
    EXPECT_FALSE(runner.AutoContextGrowthEnabled());
    feather::demo::QwenChatSession session(&runner, {2}, 3, {0}, {3});
    feather::demo::QwenGenerationOptions options;
    options.max_new_tokens = 1;
    ASSERT_EQ(session.GenerateTurn({0}, options, {}), 0) << session.LastError();
    ASSERT_EQ(session.GenerateTurn({0}, options, {}), 0) << session.LastError();
    EXPECT_EQ(runner.MaxContext(), 8);
    EXPECT_EQ(runner.ContextGrowthCount(), 0);
    EXPECT_EQ(session.LastDroppedTurns(), 1U);
}

TEST(qwen_chat_session_test, AutomaticGrowthClampsLargeRequestedBudgetToModelMaximum) {
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(WriteChatSessionFixture(4, 8).string()), 0) << runner.LastError();
    feather::demo::QwenChatSession session(&runner, {2}, 3, {0}, {3});
    feather::demo::QwenGenerationOptions options;
    options.max_new_tokens = std::numeric_limits<int>::max();
    ASSERT_EQ(session.GenerateTurn({0}, options, {}), 0) << session.LastError();
    EXPECT_EQ(runner.MaxContext(), 8);
    EXPECT_EQ(session.LastFinishReason(), feather::demo::QwenFinishReason::kEos);
    EXPECT_EQ(session.History().TurnCount(), 1U);
}

TEST(qwen_chat_session_test, SessionClosesLengthLimitedTurnsAndReusesLiveRunner) {
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(WriteChatSessionFixture(20).string()), 0) << runner.LastError();
    feather::demo::QwenChatSession session(&runner, {2}, 3, {0}, {3});
    feather::demo::QwenGenerationOptions options;
    options.max_new_tokens = 1;
    std::vector<int64_t> generated;
    ASSERT_EQ(session.GenerateTurn({0}, options, [&](int64_t t) { generated.push_back(t); }), 0)
        << session.LastError();
    EXPECT_EQ(generated, (std::vector<int64_t>{1}));
    EXPECT_EQ(runner.LastFinishReason(), feather::demo::QwenFinishReason::kLength);
    EXPECT_EQ(session.History().ReplayTokens(), (std::vector<int64_t>{2, 0, 1, 3, 0}));
    EXPECT_EQ(runner.TokensProcessed(), 5);
    ASSERT_EQ(session.GenerateTurn({0}, options, {}), 0) << session.LastError();
    EXPECT_FALSE(session.LastRebuiltHistory());
    EXPECT_EQ(runner.TokensProcessed(), 9);
    EXPECT_EQ(session.History().TurnCount(), 2U);
}

TEST(qwen_chat_session_test, SessionDoesNotDuplicateEosAndStoresEncodedNewlineExactly) {
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(WriteChatSessionFixture(20).string()), 0) << runner.LastError();
    feather::demo::QwenChatSession session(&runner, {2}, 3, {0, 2}, {3});
    feather::demo::QwenGenerationOptions options;
    options.max_new_tokens = -1;
    ASSERT_EQ(session.GenerateTurn({0}, options, {}), 0) << session.LastError();
    EXPECT_EQ(runner.LastFinishReason(), feather::demo::QwenFinishReason::kEos);
    EXPECT_EQ(session.History().ReplayTokens(), (std::vector<int64_t>{2, 0, 1, 3, 0, 2}));
    EXPECT_EQ(runner.TokensProcessed(), 6);
}

TEST(qwen_chat_session_test, SessionRebuildDropsWholeTurnAndReplaysExactRetainedTokens) {
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(WriteChatSessionFixture(10).string()), 0) << runner.LastError();
    feather::demo::QwenChatSession session(&runner, {2}, 3, {0}, {3});
    feather::demo::QwenGenerationOptions options;
    options.max_new_tokens = 1;
    ASSERT_EQ(session.GenerateTurn({0}, options, {}), 0) << session.LastError();
    ASSERT_EQ(session.GenerateTurn({1}, options, {}), 0) << session.LastError();
    ASSERT_EQ(session.GenerateTurn({0}, options, {}), 0) << session.LastError();
    EXPECT_TRUE(session.LastRebuiltHistory());
    EXPECT_EQ(session.LastDroppedTurns(), 1U);
    EXPECT_EQ(session.History().ReplayTokens(), (std::vector<int64_t>{2, 1, 3, 0, 0, 1, 3, 0}));
    EXPECT_EQ(runner.TokensProcessed(), 8);
}

TEST(qwen_chat_session_test, SessionRejectsOverlargePromptBeforeRunnerOrHistoryMutation) {
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(WriteChatSessionFixture(10).string()), 0) << runner.LastError();
    feather::demo::QwenChatSession session(&runner, {2}, 3, {0}, {3});
    feather::demo::QwenGenerationOptions options;
    options.max_new_tokens = 1;
    ASSERT_EQ(session.GenerateTurn({0}, options, {}), 0) << session.LastError();
    EXPECT_NE(session.GenerateTurn(std::vector<int64_t>(7, 0), options, {}), 0);
    EXPECT_EQ(runner.TokensProcessed(), 5);
    EXPECT_EQ(session.History().ReplayTokens(), (std::vector<int64_t>{2, 0, 1, 3, 0}));
    EXPECT_EQ(session.History().TurnCount(), 1U);
}

TEST(qwen_chat_session_test, SessionRestoresCompleteHistoryWhenPromptIsCancelled) {
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(WriteChatSessionFixture(20).string()), 0) << runner.LastError();
    feather::demo::QwenChatSession session(&runner, {2}, 3, {0}, {3});
    feather::demo::QwenGenerationOptions options;
    options.max_new_tokens = 1;
    ASSERT_EQ(session.GenerateTurn({0}, options, {}), 0) << session.LastError();
    int cancellation_checks = 0;
    options.should_cancel = [&]() { return ++cancellation_checks >= 2; };
    ASSERT_EQ(session.GenerateTurn({0, 1, 0}, options, {}), 0) << session.LastError();
    EXPECT_EQ(session.LastFinishReason(), feather::demo::QwenFinishReason::kCancelled);
    EXPECT_EQ(session.History().ReplayTokens(), (std::vector<int64_t>{2, 0, 1, 3, 0}));
    EXPECT_EQ(runner.TokensProcessed(), 5);
}

TEST(qwen_chat_session_test, SessionRestoresHistoryAfterPartiallyConsumedInvalidPrompt) {
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(WriteChatSessionFixture(20).string()), 0) << runner.LastError();
    feather::demo::QwenChatSession session(&runner, {2}, 3, {0}, {3});
    feather::demo::QwenGenerationOptions options;
    options.max_new_tokens = 1;
    ASSERT_EQ(session.GenerateTurn({0}, options, {}), 0) << session.LastError();

    // The valid prefix executes before Gather rejects the out-of-range token.
    EXPECT_NE(session.GenerateTurn({0, 99}, options, {}), 0);
    EXPECT_FALSE(session.LastError().empty());
    EXPECT_EQ(runner.TokensProcessed(), 5);
    EXPECT_EQ(session.History().ReplayTokens(), (std::vector<int64_t>{2, 0, 1, 3, 0}));

    ASSERT_EQ(session.GenerateTurn({0}, options, {}), 0) << session.LastError();
    EXPECT_FALSE(session.LastRebuiltHistory());
    EXPECT_EQ(runner.TokensProcessed(), 9);
    EXPECT_EQ(session.History().ReplayTokens(), (std::vector<int64_t>{2, 0, 1, 3, 0, 0, 1, 3, 0}));
}

TEST(qwen_chat_session_test, SessionRestoresProvisionallyDroppedTurnsAfterGenerationFails) {
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(WriteChatSessionFixture(10).string()), 0) << runner.LastError();
    feather::demo::QwenChatSession session(&runner, {2}, 3, {0}, {3});
    feather::demo::QwenGenerationOptions options;
    options.max_new_tokens = 1;
    ASSERT_EQ(session.GenerateTurn({0}, options, {}), 0) << session.LastError();
    ASSERT_EQ(session.GenerateTurn({1}, options, {}), 0) << session.LastError();

    EXPECT_NE(session.GenerateTurn({0, 99}, options, {}), 0);
    EXPECT_FALSE(session.LastError().empty());
    EXPECT_EQ(runner.TokensProcessed(), 8);
    EXPECT_EQ(session.History().TurnCount(), 2U);
    EXPECT_EQ(session.History().ReplayTokens(), (std::vector<int64_t>{2, 0, 1, 3, 0, 1, 3, 0}));
    EXPECT_FALSE(session.LastRebuiltHistory());

    ASSERT_EQ(session.GenerateTurn({0}, options, {}), 0) << session.LastError();
    EXPECT_EQ(runner.TokensProcessed(), 8);
    EXPECT_EQ(session.History().ReplayTokens(), (std::vector<int64_t>{2, 1, 3, 0, 0, 1, 3, 0}));
}

TEST(qwen_chat_session_test, SessionRestoresHistoryAfterClosureFails) {
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(WriteChatSessionFixture(20).string()), 0) << runner.LastError();
    feather::demo::QwenChatSession session(&runner, {2}, 3, {99}, {3});
    feather::demo::QwenGenerationOptions options;
    options.max_new_tokens = 1;
    EXPECT_NE(session.GenerateTurn({0}, options, {}), 0);
    EXPECT_FALSE(session.LastError().empty());
    EXPECT_EQ(runner.TokensProcessed(), 1);
    EXPECT_EQ(session.History().ReplayTokens(), (std::vector<int64_t>{2}));
    EXPECT_EQ(session.History().TurnCount(), 0U);
}

TEST(qwen_chat_session_test, SessionKeepsOriginalFailureWhenHistoryRecoveryAlsoFails) {
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(WriteChatSessionFixture(20).string()), 0) << runner.LastError();
    feather::demo::QwenChatSession session(&runner, {2, 99}, 3, {0}, {3});
    feather::demo::QwenGenerationOptions options;
    options.max_new_tokens = 1;
    EXPECT_NE(session.GenerateTurn({0}, options, {}), 0);
    EXPECT_NE(session.LastError().find("Qwen runtime graph execution failed"), std::string::npos);
    EXPECT_NE(session.LastError().find("history recovery failed"), std::string::npos);
    EXPECT_EQ(session.History().ReplayTokens(), (std::vector<int64_t>{2, 99}));
    EXPECT_EQ(session.History().TurnCount(), 0U);
    // Recovery consumed only the valid system prefix. The next call must
    // rebuild and fail again instead of using that incomplete live state.
    EXPECT_NE(session.GenerateTurn({0}, options, {}), 0);
    EXPECT_NE(session.LastError().find("history recovery failed"), std::string::npos);
    EXPECT_EQ(session.History().ReplayTokens(), (std::vector<int64_t>{2, 99}));
    EXPECT_EQ(session.History().TurnCount(), 0U);
}

TEST(qwen_chat_session_test, SmallContextDefaultBudgetReportsContextLimitAndClosesTurn) {
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(WriteChatSessionFixture(8).string()), 0) << runner.LastError();
    feather::demo::QwenChatSession session(&runner, {2}, 2, {0}, {2});
    feather::demo::QwenGenerationOptions options;
    ASSERT_EQ(session.GenerateTurn({0}, options, {}), 0) << session.LastError();
    EXPECT_EQ(session.LastFinishReason(), feather::demo::QwenFinishReason::kContextLimit);
    EXPECT_EQ(session.History().ReplayTokens(), (std::vector<int64_t>{2, 0, 1, 3, 3, 3, 2, 0}));
    EXPECT_EQ(runner.TokensProcessed(), 8);
}

int RunDemoWithArguments(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    for (auto& argument : arguments) {
        argv.push_back(argument.data());
    }
    return feather::demo::RunQwenDemo(static_cast<int>(argv.size()), argv.data());
}

TEST(qwen_chat_session_test, CliRejectsMalformedNumbersWithoutThrowing) {
    EXPECT_EQ(RunDemoWithArguments({"qwen_demo", "--max-new-tokens", "12oops", "--help"}), 1);
    EXPECT_NO_THROW(EXPECT_EQ(RunDemoWithArguments({"qwen_demo", "--max-new-tokens", "abc"}), 1));
    EXPECT_NO_THROW(EXPECT_EQ(RunDemoWithArguments({"qwen_demo", "--max-context", "999999999999999999999"}), 1));
}

TEST(qwen_chat_session_test, CliAcceptsContextSizeAndLegacyAlias) {
    EXPECT_EQ(RunDemoWithArguments({"qwen_demo", "--context-size", "128", "--help"}), 0);
    EXPECT_EQ(RunDemoWithArguments({"qwen_demo", "--max-context", "128", "--help"}), 0);
    EXPECT_EQ(RunDemoWithArguments({"qwen_demo", "--max-new-tokens", "-1", "--help"}), 0);
}

}  // namespace
