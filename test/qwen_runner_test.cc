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
#include <string>
#include <unordered_map>
#include <vector>

#ifdef FEATHER_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include "demo/qwen_demo.h"
#include "demo/qwen_runner.h"
#include "model/model_io.h"
#include "util/bf16.h"

namespace {

std::filesystem::path QwenModelDirectory() {
    return std::filesystem::path(__FILE__).parent_path().parent_path() / "models" / "llm" / "qwen3.5-0.8b";
}

feather::model::ValueDesc MakeValue(const std::string& name, const std::vector<int64_t>& dims,
                                    feather::DataType data_type, bool constant = false) {
    feather::model::ValueDesc value;
    value.tensor.name = name;
    value.tensor.dims = dims;
    value.tensor.data_type = data_type;
    value.constant = constant;
    return value;
}

std::shared_ptr<feather::Tensor> MakeFloatTensor(const std::vector<float>& values, const std::vector<int64_t>& dims) {
    auto tensor = std::make_shared<feather::Tensor>();
    tensor->Assign<float>(values, dims);
    return tensor;
}

std::shared_ptr<feather::Tensor> MakeBf16Tensor(const std::vector<float>& values,
                                                const std::vector<int64_t>& dims) {
    std::vector<feather::BFloat16> storage;
    storage.reserve(values.size());
    for (const float value : values) {
        storage.push_back(feather::BFloat16{feather::FloatToBFloat16(value)});
    }
    auto tensor = std::make_shared<feather::Tensor>();
    tensor->Assign<feather::BFloat16>(storage, dims);
    return tensor;
}

std::filesystem::path WriteAutoregressiveFixture(bool flexible_context = false) {
    feather::model::ModelDesc model;
    model.name = "qwen_runner_fixture";
    model.version = 1;
    if (flexible_context) {
        model.metadata["qwen.context.version"] = int64_t{1};
        model.metadata["qwen.context.default"] = int64_t{4};
        model.metadata["qwen.context.maximum"] = int64_t{8};
        model.metadata["qwen.context.value.attention_mask"] = std::vector<int64_t>{3, 0};
    }
    model.graph.name = "decode";
    model.graph.inputs = {"token_ids", "position_id", "attention_mask", "recurrent_state_0"};
    model.graph.outputs = {"next_recurrent_state_0", "logits"};
    model.graph.values = {
        MakeValue("token_ids", {1, 1}, feather::DataType::INT64),
        MakeValue("position_id", {1}, feather::DataType::INT64),
        MakeValue("attention_mask", {1, 1, 1, 4}, feather::DataType::BF16),
        MakeValue("recurrent_state_0", {1}, feather::DataType::FP32),
        MakeValue("lookup", {4, 4}, feather::DataType::FP32, true),
        MakeValue("logits", {1, 1, 4}, feather::DataType::FP32),
        MakeValue("next_recurrent_state_0", {1}, feather::DataType::FP32),
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

    std::unordered_map<std::string, std::shared_ptr<feather::Tensor>> weights;
    weights["lookup"] = MakeFloatTensor({0.0f, 5.0f, 0.0f, 0.0f,
                                          0.0f, 0.0f, 0.0f, 5.0f,
                                          0.0f, 0.0f, 5.0f, 0.0f,
                                          0.0f, 0.0f, 5.0f, 0.0f},
                                         {4, 4});
    const auto path = std::filesystem::temp_directory_path() /
                      (flexible_context ? "qwen_runner_flexible_fixture.fth" : "qwen_runner_fixture.fth");
    feather::model::ModelWriter writer;
    EXPECT_TRUE(writer.Save(path.string(), model, weights));
    return path;
}

TEST(qwen_runner_test, EvaluationReadsFullDistributionAndRejectsUnrequestedScoringBeforeMutation) {
    const auto path = WriteAutoregressiveFixture();
    feather::demo::QwenRunner normal;
    ASSERT_EQ(normal.Load(path.string()), 0) << normal.LastError();
    std::vector<float> logits;
    EXPECT_NE(normal.EvaluateToken(0, &logits), 0);
    EXPECT_EQ(normal.TokensProcessed(), 0);
    feather::demo::QwenLoadOptions options;
    options.retain_logits = true;
    options.additional_outputs = {"next_recurrent_state_0"};
    feather::demo::QwenRunner evaluation;
    ASSERT_EQ(evaluation.Load(path.string(), feather::demo::QwenBackend::kCommon, options), 0)
        << evaluation.LastError();
    ASSERT_EQ(evaluation.EvaluateToken(0, &logits), 0) << evaluation.LastError();
    EXPECT_EQ(logits, std::vector<float>({0, 5, 0, 0}));
    EXPECT_EQ(evaluation.TokensProcessed(), 1);
    std::vector<float> state;
    ASSERT_EQ(evaluation.ReadOutput("next_recurrent_state_0", &state), 0);
    EXPECT_EQ(state, std::vector<float>({0}));
    EXPECT_NE(evaluation.ReadOutput("lookup", &state), 0);
    ASSERT_EQ(evaluation.Reset(), 0);
    ASSERT_EQ(evaluation.EvaluateToken(1, &logits), 0);
    EXPECT_EQ(logits, std::vector<float>({0, 0, 0, 5}));
    ASSERT_EQ(evaluation.Consume({2}), 0);
    EXPECT_NE(evaluation.ReadOutput("logits", &logits), 0);
}

TEST(qwen_runner_test, EvaluationRetainsOptionsAfterAutomaticContextGrowth) {
    const auto path = WriteAutoregressiveFixture(true);
    feather::demo::QwenLoadOptions options;
    options.retain_logits = true;
    options.additional_outputs = {"next_recurrent_state_0"};
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(path.string(), feather::demo::QwenBackend::kCommon, options), 0);
    std::vector<float> logits;
    for (int i = 0; i < 7; ++i) {
        ASSERT_EQ(runner.EvaluateToken(i % 4, &logits), 0) << runner.LastError();
        ASSERT_EQ(logits.size(), 4U);
    }
    EXPECT_EQ(runner.MaxContext(), 8);
    EXPECT_EQ(runner.ContextGrowthCount(), 1);
    std::vector<float> state;
    EXPECT_EQ(runner.ReadOutput("next_recurrent_state_0", &state), 0);
}

std::filesystem::path WriteStateTransitionFixture() {
    feather::model::ModelDesc model;
    model.name = "qwen_runner_state_transition_fixture";
    model.version = 1;
    model.graph.name = "decode";
    model.graph.inputs = {"token_ids", "position_id", "attention_mask", "recurrent_state_0"};
    model.graph.outputs = {"next_recurrent_state_0", "logits"};
    model.graph.values = {
        MakeValue("token_ids", {1, 1}, feather::DataType::INT64),
        MakeValue("position_id", {1}, feather::DataType::INT64),
        MakeValue("attention_mask", {1, 1, 1, 8}, feather::DataType::BF16),
        MakeValue("recurrent_state_0", {1, 4}, feather::DataType::FP32),
        MakeValue("transition", {4, 4}, feather::DataType::FP32, true),
        MakeValue("bias", {4}, feather::DataType::FP32, true),
        MakeValue("next_state_raw", {1, 4}, feather::DataType::FP32),
        MakeValue("next_recurrent_state_0", {1, 4}, feather::DataType::FP32),
        MakeValue("logits", {1, 1, 4}, feather::DataType::FP32),
    };

    feather::model::NodeDesc transition;
    transition.name = "state_transition";
    transition.op_type = "Gemm";
    transition.inputs = {"recurrent_state_0", "transition", "bias"};
    transition.outputs = {"next_state_raw"};

    feather::model::NodeDesc state_output;
    state_output.name = "next_state";
    state_output.op_type = "Identity";
    state_output.inputs = {"next_state_raw"};
    state_output.outputs = {"next_recurrent_state_0"};

    feather::model::NodeDesc logits;
    logits.name = "state_logits";
    logits.op_type = "Reshape";
    logits.inputs = {"next_state_raw"};
    logits.outputs = {"logits"};
    logits.attributes["shape"] = std::vector<int64_t>{1, 1, 4};
    model.graph.nodes = {transition, state_output, logits};

    std::unordered_map<std::string, std::shared_ptr<feather::Tensor>> weights;
    weights["transition"] = MakeFloatTensor({0.0f, 0.0f, 0.0f, 0.0f,
                                               0.0f, -2.0f, 2.0f, 0.0f,
                                               0.0f, 0.0f, 0.0f, 0.0f,
                                               0.0f, 0.0f, 0.0f, 0.0f},
                                              {4, 4});
    weights["bias"] = MakeFloatTensor({0.0f, 10.0f, 0.0f, 0.0f}, {4});
    const auto path = std::filesystem::temp_directory_path() / "qwen_runner_state_transition_fixture.fth";
    feather::model::ModelWriter writer;
    EXPECT_TRUE(writer.Save(path.string(), model, weights));
    return path;
}

TEST(qwen_runner_test, FailedReloadCannotExecuteOldGraphWithNewStateContract) {
    const auto path = WriteStateTransitionFixture();
    feather::model::ModelLoader loader;
    ASSERT_TRUE(loader.Load(path.string()));
    auto broken = loader.model();
    broken.graph.outputs = {"logits"};
    std::unordered_map<std::string, std::shared_ptr<feather::Tensor>> weights;
    for (const auto& value : broken.graph.values)
        if (value.constant) weights[value.tensor.name] = loader.CreateWeightTensor(value.tensor.name);
    feather::model::ModelWriter writer;
    const auto broken_path = std::filesystem::temp_directory_path() / "qwen_failed_reload_fixture.fth";
    ASSERT_TRUE(writer.Save(broken_path.string(), broken, weights));
    feather::demo::QwenRunner runner;
    feather::demo::QwenLoadOptions options;
    options.retain_logits = true;
    ASSERT_EQ(runner.Load(path.string(), feather::demo::QwenBackend::kCommon, options), 0);
    std::vector<float> logits;
    ASSERT_EQ(runner.EvaluateToken(0, &logits), 0);
    ASSERT_NE(runner.Load(broken_path.string(), feather::demo::QwenBackend::kCommon, options), 0);
    EXPECT_NE(runner.EvaluateToken(0, &logits), 0);
    EXPECT_NE(runner.Consume({0}), 0);
    EXPECT_NE(runner.Reset(), 0);
    EXPECT_EQ(runner.CachedPlanCount(), 0U);
    ASSERT_EQ(runner.Load(path.string(), feather::demo::QwenBackend::kCommon, options), 0);
    ASSERT_EQ(runner.EvaluateToken(0, &logits), 0);
    EXPECT_EQ(logits, (std::vector<float>{0, 10, 0, 0}));
}

TEST(qwen_runner_test, FailedReloadAfterCompilationLeavesRunnerUnloaded) {
    const auto path = WriteStateTransitionFixture();
    feather::model::ModelLoader loader;
    ASSERT_TRUE(loader.Load(path.string()));
    auto broken = loader.model();
    broken.metadata["qwen.prefill.file"] = std::string("missing_qwen_companion.fth");
    broken.metadata["qwen.execution.version"] = int64_t{1};
    broken.metadata["qwen.execution.sequence"] = int64_t{1};
    broken.metadata["qwen.cache.layout"] = std::string("position");
    broken.metadata["qwen.weights.id"] = std::string("test");
    std::unordered_map<std::string, std::shared_ptr<feather::Tensor>> weights;
    for (const auto& value : broken.graph.values)
        if (value.constant) weights[value.tensor.name] = loader.CreateWeightTensor(value.tensor.name);
    const auto broken_path = std::filesystem::temp_directory_path() / "qwen_failed_companion_fixture.fth";
    feather::model::ModelWriter writer;
    ASSERT_TRUE(writer.Save(broken_path.string(), broken, weights));
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(path.string()), 0);
    ASSERT_NE(runner.Load(broken_path.string()), 0);
    EXPECT_NE(runner.Consume({0}), 0);
    EXPECT_EQ(runner.CachedPlanCount(), 0U);
}

std::filesystem::path WriteBf16GreedyFixture() {
    feather::model::ModelDesc model;
    model.name = "qwen_runner_bf16_greedy_fixture";
    model.version = 1;
    model.graph.name = "decode";
    model.graph.inputs = {"token_ids", "position_id", "attention_mask", "recurrent_state_0"};
    model.graph.outputs = {"next_recurrent_state_0", "logits"};
    model.graph.values = {
        MakeValue("token_ids", {1, 1}, feather::DataType::INT64),
        MakeValue("position_id", {1}, feather::DataType::INT64),
        MakeValue("attention_mask", {1, 1, 1, 4}, feather::DataType::BF16),
        MakeValue("recurrent_state_0", {1}, feather::DataType::FP32),
        MakeValue("lookup", {2, 4}, feather::DataType::BF16, true),
        MakeValue("logits", {1, 1, 4}, feather::DataType::BF16),
        MakeValue("next_recurrent_state_0", {1}, feather::DataType::FP32),
    };

    feather::model::NodeDesc gather;
    gather.name = "lookup_bf16_logits";
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

    std::unordered_map<std::string, std::shared_ptr<feather::Tensor>> weights;
    weights["lookup"] = MakeBf16Tensor({std::numeric_limits<float>::quiet_NaN(), 5.0f, 5.0f, 4.0f,
                                         -1.0f, -2.0f, -3.0f, -4.0f},
                                        {2, 4});
    const auto path = std::filesystem::temp_directory_path() / "qwen_runner_bf16_greedy_fixture.fth";
    feather::model::ModelWriter writer;
    EXPECT_TRUE(writer.Save(path.string(), model, weights));
    return path;
}

#ifdef FEATHER_WITH_CUDA
bool HasCudaDevice() {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}
#endif

}  // namespace

TEST(qwen_demo_test, BuffersIncompleteUtf8UntilTheNextChunk) {
    feather::demo::QwenUtf8Stream stream;

    EXPECT_EQ(stream.Push("hello"), "hello");
    EXPECT_TRUE(stream.Push("\xE4").empty());
    EXPECT_EQ(stream.Push("\xBD\xA0"), "\xE4\xBD\xA0");
    EXPECT_TRUE(stream.Finish().empty());

    feather::demo::QwenUtf8Stream emoji;
    EXPECT_TRUE(emoji.Push("\xF0\x9F").empty());
    EXPECT_EQ(emoji.Push("\x98\x80"), "\xF0\x9F\x98\x80");

    feather::demo::QwenUtf8Stream malformed;
    EXPECT_EQ(malformed.Push("\xC0ok"), "\xEF\xBF\xBDok");

    feather::demo::QwenUtf8Stream truncated;
    EXPECT_TRUE(truncated.Push("\xE4").empty());
    EXPECT_EQ(truncated.Finish(), "\xEF\xBF\xBD");
}

TEST(qwen_runner_test, StreamsGeneratedTokensThroughCallback) {
    const auto model_path = WriteAutoregressiveFixture();
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(model_path.string(), feather::demo::QwenBackend::kCommon), 0) << runner.LastError();

    std::vector<int64_t> streamed;
    std::vector<int64_t> processed_at_callback;
    ASSERT_EQ(runner.GenerateStream({0}, 3, {2}, [&runner, &streamed, &processed_at_callback](int64_t token_id) {
                  streamed.push_back(token_id);
                  processed_at_callback.push_back(runner.TokensProcessed());
              }),
              0)
        << runner.LastError();
    EXPECT_EQ(streamed, (std::vector<int64_t>{1, 3, 2}));
    EXPECT_EQ(processed_at_callback, (std::vector<int64_t>{1, 2, 3}));
    EXPECT_EQ(runner.TokensProcessed(), 4);
}

TEST(qwen_runner_test, ExposesRuntimeProfileSummariesWhenEnabled) {
    const auto model_path = WriteAutoregressiveFixture();
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(model_path.string(), feather::demo::QwenBackend::kCommon), 0) << runner.LastError();

    runner.SetRuntimeProfilingEnabled(true);
    std::vector<int64_t> generated;
    ASSERT_EQ(runner.Generate({0}, 1, {}, &generated), 0) << runner.LastError();

    const auto& summaries = runner.RuntimeProfileSummaries();
    ASSERT_FALSE(summaries.empty());
    EXPECT_EQ(summaries[0].call_count, 2);
    EXPECT_EQ(summaries[0].op_type, "Gather");
}

TEST(qwen_runner_test, PreservesOutputWhenPromptValidationFails) {
    const auto model_path = WriteAutoregressiveFixture();
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(model_path.string(), feather::demo::QwenBackend::kCommon), 0) << runner.LastError();

    std::vector<int64_t> generated = {42};
    EXPECT_NE(runner.Generate({}, 1, {}, &generated), 0);
    EXPECT_EQ(generated, (std::vector<int64_t>{42}));
}

TEST(qwen_runner_test, ReportsLengthAndResetClearsFinishReason) {
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(WriteAutoregressiveFixture().string()), 0) << runner.LastError();
    std::vector<int64_t> generated;
    ASSERT_EQ(runner.Generate({0}, 1, {}, &generated), 0) << runner.LastError();
    EXPECT_EQ(generated, (std::vector<int64_t>{1}));
    EXPECT_EQ(runner.LastFinishReason(), feather::demo::QwenFinishReason::kLength);
    EXPECT_STREQ(feather::demo::QwenFinishReasonName(runner.LastFinishReason()), "length");
    ASSERT_EQ(runner.Reset(), 0) << runner.LastError();
    EXPECT_EQ(runner.TokensProcessed(), 0);
    EXPECT_EQ(runner.LastFinishReason(), feather::demo::QwenFinishReason::kNone);
}

TEST(qwen_runner_test, UnlimitedBudgetEndsAtEosAndConsumesStopToken) {
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(WriteAutoregressiveFixture().string()), 0) << runner.LastError();
    std::vector<int64_t> generated;
    ASSERT_EQ(runner.Generate({0}, -1, {2}, &generated), 0) << runner.LastError();
    EXPECT_EQ(generated, (std::vector<int64_t>{1, 3, 2}));
    EXPECT_EQ(runner.TokensProcessed(), 4);
    EXPECT_EQ(runner.LastFinishReason(), feather::demo::QwenFinishReason::kEos);
    EXPECT_STREQ(feather::demo::QwenFinishReasonName(runner.LastFinishReason()), "eos");
}

TEST(qwen_runner_test, StopsAtContextBeforeEmittingAnUnconsumableToken) {
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(WriteAutoregressiveFixture().string()), 0) << runner.LastError();
    std::vector<int64_t> streamed;
    ASSERT_EQ(runner.GenerateStream({0}, 10, {}, [&](int64_t token) { streamed.push_back(token); }), 0)
        << runner.LastError();
    EXPECT_EQ(streamed, (std::vector<int64_t>{1, 3, 2}));
    EXPECT_EQ(runner.TokensProcessed(), 4);
    EXPECT_EQ(runner.LastFinishReason(), feather::demo::QwenFinishReason::kContextLimit);
    EXPECT_STREQ(feather::demo::QwenFinishReasonName(runner.LastFinishReason()), "context_limit");
}

TEST(qwen_runner_test, FullPromptProducesNoTokensAndReportsContextLimit) {
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(WriteAutoregressiveFixture().string()), 0) << runner.LastError();
    std::vector<int64_t> generated{42};
    ASSERT_EQ(runner.Generate({0, 1, 3, 2}, -1, {}, &generated), 0) << runner.LastError();
    EXPECT_TRUE(generated.empty());
    EXPECT_EQ(runner.TokensProcessed(), 4);
    EXPECT_EQ(runner.LastFinishReason(), feather::demo::QwenFinishReason::kContextLimit);
}

TEST(qwen_runner_test, RejectsPromptOverflowBeforeChangingStateOrOutput) {
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(WriteStateTransitionFixture().string()), 0) << runner.LastError();
    ASSERT_EQ(runner.Consume({0}), 0) << runner.LastError();
    std::vector<int64_t> generated{42};
    EXPECT_NE(runner.Generate(std::vector<int64_t>(8, 0), 1, {}, &generated), 0);
    EXPECT_EQ(runner.TokensProcessed(), 1);
    EXPECT_EQ(generated, (std::vector<int64_t>{42}));
    ASSERT_EQ(runner.Generate({0}, 1, {}, &generated), 0) << runner.LastError();
    EXPECT_EQ(generated, (std::vector<int64_t>{2}));
}

TEST(qwen_runner_test, RejectsConsumeOverflowBeforeChangingState) {
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(WriteStateTransitionFixture().string()), 0) << runner.LastError();
    ASSERT_EQ(runner.Consume({0}), 0) << runner.LastError();
    EXPECT_NE(runner.Consume(std::vector<int64_t>(8, 0)), 0);
    EXPECT_EQ(runner.TokensProcessed(), 1);
    std::vector<int64_t> generated;
    ASSERT_EQ(runner.Generate({0}, 1, {}, &generated), 0) << runner.LastError();
    EXPECT_EQ(generated, (std::vector<int64_t>{2}));
}

TEST(qwen_runner_test, ReservesContextSlotsForTurnClosure) {
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(WriteAutoregressiveFixture().string()), 0) << runner.LastError();
    feather::demo::QwenGenerationOptions options;
    options.max_new_tokens = -1;
    options.context_reserve_tokens = 2;
    std::vector<int64_t> generated;
    ASSERT_EQ(runner.Generate({0}, options, {}, &generated), 0) << runner.LastError();
    EXPECT_EQ(generated, (std::vector<int64_t>{1}));
    EXPECT_EQ(runner.TokensProcessed(), 2);
    EXPECT_EQ(runner.LastFinishReason(), feather::demo::QwenFinishReason::kContextLimit);
    ASSERT_EQ(runner.Consume({3, 2}), 0) << runner.LastError();
    EXPECT_EQ(runner.TokensProcessed(), 4);
    EXPECT_NE(runner.Consume({0}), 0);
    EXPECT_EQ(runner.TokensProcessed(), 4);
}

TEST(qwen_runner_test, RejectsInvalidBudgetsAndReserveWithoutMutation) {
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(WriteAutoregressiveFixture().string()), 0) << runner.LastError();
    for (const int budget : {0, -2}) {
        feather::demo::QwenGenerationOptions options;
        options.max_new_tokens = budget;
        std::vector<int64_t> generated{42};
        EXPECT_NE(runner.Generate({0}, options, {}, &generated), 0);
        EXPECT_EQ(generated, (std::vector<int64_t>{42}));
        EXPECT_EQ(runner.TokensProcessed(), 0);
    }
    for (const int64_t reserve : {-1, 4, 5}) {
        feather::demo::QwenGenerationOptions options;
        options.context_reserve_tokens = reserve;
        std::vector<int64_t> generated{42};
        EXPECT_NE(runner.Generate({0}, options, {}, &generated), 0);
        EXPECT_EQ(generated, (std::vector<int64_t>{42}));
        EXPECT_EQ(runner.TokensProcessed(), 0);
    }
}

TEST(qwen_runner_test, CancellationDuringPromptKeepsConsumedPrefixAndClearsGeneratedOutput) {
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(WriteAutoregressiveFixture().string()), 0) << runner.LastError();
    feather::demo::QwenGenerationOptions options;
    options.should_cancel = [&] { return runner.TokensProcessed() >= 1; };
    std::vector<int64_t> generated{42};
    ASSERT_EQ(runner.Generate({0, 1}, options, {}, &generated), 0) << runner.LastError();
    EXPECT_TRUE(generated.empty());
    EXPECT_EQ(runner.TokensProcessed(), 1);
    EXPECT_EQ(runner.LastFinishReason(), feather::demo::QwenFinishReason::kCancelled);
    EXPECT_STREQ(feather::demo::QwenFinishReasonName(runner.LastFinishReason()), "cancelled");
}

TEST(qwen_runner_test, StreamingCancellationPreservesEmittedAndConsumedTokenCounts) {
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(WriteAutoregressiveFixture().string()), 0) << runner.LastError();
    feather::demo::QwenGenerationOptions options;
    bool cancelled = false;
    options.should_cancel = [&] { return cancelled; };
    std::vector<int64_t> streamed;
    ASSERT_EQ(runner.GenerateStream({0}, options, {}, [&](int64_t token) {
                  streamed.push_back(token);
                  cancelled = true;
              }), 0) << runner.LastError();
    EXPECT_EQ(streamed, (std::vector<int64_t>{1}));
    EXPECT_EQ(runner.TokensProcessed(), 2);
    EXPECT_EQ(runner.LastFinishReason(), feather::demo::QwenFinishReason::kCancelled);
}

TEST(qwen_runner_test, LegacyModelRejectsDifferentCapacityAndReportsItsLimit) {
    feather::demo::QwenRunner runner;
    const auto path = WriteAutoregressiveFixture();
    ASSERT_EQ(runner.Load(path.string(), feather::demo::QwenBackend::kCommon, 4), 0) << runner.LastError();
    EXPECT_EQ(runner.MaxContext(), 4);
    EXPECT_EQ(runner.ModelContextLimit(), 4);
    EXPECT_NE(runner.Load(path.string(), feather::demo::QwenBackend::kCommon, 8), 0);
}

TEST(qwen_runner_test, UsesSpecializedContextForRuntimeInputsAndGenerationCapacity) {
    feather::demo::QwenRunner runner;
    const auto path = WriteAutoregressiveFixture(true);
    ASSERT_EQ(runner.Load(path.string(), feather::demo::QwenBackend::kCommon, 8), 0) << runner.LastError();
    EXPECT_EQ(runner.MaxContext(), 8);
    EXPECT_EQ(runner.ModelContextLimit(), 8);
    std::vector<int64_t> generated;
    ASSERT_EQ(runner.Generate({0}, -1, {}, &generated), 0) << runner.LastError();
    EXPECT_EQ(generated, (std::vector<int64_t>{1, 3, 2, 2, 2, 2, 2}));
    EXPECT_EQ(runner.TokensProcessed(), 8);
    EXPECT_EQ(runner.LastFinishReason(), feather::demo::QwenFinishReason::kContextLimit);
    ASSERT_EQ(runner.Load(path.string(), feather::demo::QwenBackend::kCommon, 2), 0) << runner.LastError();
    EXPECT_EQ(runner.MaxContext(), 2);
    EXPECT_EQ(runner.ModelContextLimit(), 8);
    ASSERT_EQ(runner.Generate({0}, -1, {}, &generated), 0) << runner.LastError();
    EXPECT_EQ(generated, (std::vector<int64_t>{1}));
    EXPECT_EQ(runner.TokensProcessed(), 2);
}

TEST(qwen_runner_test, GreedilyGeneratesTokensAndPersistsExplicitState) {
    const auto model_path = WriteAutoregressiveFixture();
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(model_path.string(), feather::demo::QwenBackend::kCommon), 0) << runner.LastError();

    std::vector<int64_t> generated;
    ASSERT_EQ(runner.Generate({0}, 3, {2}, &generated), 0) << runner.LastError();
    EXPECT_EQ(generated, (std::vector<int64_t>{1, 3, 2}));
    EXPECT_EQ(runner.TokensProcessed(), 4);

    ASSERT_EQ(runner.Reset(), 0) << runner.LastError();
    ASSERT_EQ(runner.Generate({0}, 1, {}, &generated), 0) << runner.LastError();
    EXPECT_EQ(generated, (std::vector<int64_t>{1}));
    EXPECT_EQ(runner.TokensProcessed(), 2);
    ASSERT_EQ(runner.Consume({2}), 0) << runner.LastError();
    EXPECT_EQ(runner.TokensProcessed(), 3);

    ASSERT_EQ(runner.Reset(), 0) << runner.LastError();
    EXPECT_EQ(runner.TokensProcessed(), 0);
    ASSERT_EQ(runner.Generate({0}, 3, {2}, &generated), 0) << runner.LastError();
    EXPECT_EQ(generated, (std::vector<int64_t>{1, 3, 2}));
}

TEST(qwen_runner_test, PreservesPingPongStateWhenNextStateUsesIdentityView) {
    const auto model_path = WriteStateTransitionFixture();
    for (const auto backend : {feather::demo::QwenBackend::kCommon, feather::demo::QwenBackend::kX86}) {
        feather::demo::QwenRunner runner;
        ASSERT_EQ(runner.Load(model_path.string(), backend), 0) << runner.LastError();

        std::vector<int64_t> generated;
        ASSERT_EQ(runner.Generate({0}, 2, {}, &generated), 0) << runner.LastError();
        // The second decode depends on the first state transition. With the
        // Identity output buffer swapped directly, Gemm becomes in-place and
        // picks token 0 instead of token 2.
        EXPECT_EQ(generated, (std::vector<int64_t>{1, 2}));
    }
}

#ifdef FEATHER_WITH_CUDA
TEST(qwen_runner_test, CudaPreservesPingPongStateAndResetWhenNextStateUsesIdentityRelay) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    const auto model_path = WriteStateTransitionFixture();
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(model_path.string(), feather::demo::QwenBackend::kCuda), 0) << runner.LastError();

    std::vector<int64_t> generated;
    ASSERT_EQ(runner.Generate({0}, 2, {}, &generated), 0) << runner.LastError();
    EXPECT_EQ(generated, (std::vector<int64_t>{1, 2}));

    ASSERT_EQ(runner.Reset(), 0) << runner.LastError();
    ASSERT_EQ(runner.Generate({0}, 2, {}, &generated), 0) << runner.LastError();
    EXPECT_EQ(generated, (std::vector<int64_t>{1, 2}));
}

TEST(qwen_runner_test, CudaGraphPreservesPingPongStateAcrossAlternatingExecutions) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    const auto model_path = WriteStateTransitionFixture();
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(model_path.string(), feather::demo::QwenBackend::kCuda), 0) << runner.LastError();

    std::vector<int64_t> generated;
    ASSERT_EQ(runner.Generate({0}, 4, {}, &generated), 0) << runner.LastError();
    SCOPED_TRACE("cuda graph launches=" + std::to_string(runner.CudaGraphLaunchCount()));
    EXPECT_EQ(generated, (std::vector<int64_t>{1, 2, 1, 2}));
    EXPECT_GE(runner.CudaGraphLaunchCount(), 2);

    runner.SetRuntimeProfilingEnabled(true);
    EXPECT_EQ(runner.CudaGraphLaunchCount(), 0);
}
#endif

TEST(qwen_runner_test, SelectsFirstFiniteMaximumFromBf16Logits) {
    const auto model_path = WriteBf16GreedyFixture();
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(model_path.string(), feather::demo::QwenBackend::kCommon), 0) << runner.LastError();

    std::vector<int64_t> generated;
    ASSERT_EQ(runner.Generate({0}, 1, {}, &generated), 0) << runner.LastError();
    EXPECT_EQ(generated, (std::vector<int64_t>{1}));
}

TEST(qwen_runner_test, LoadsRealDirectSafetensorsExportWithExplicitStateBindings) {
    const auto model_path = QwenModelDirectory() / "qwen3.5-0.8b_decode_bf16_ctx8.fth";
    if (!std::filesystem::is_regular_file(model_path)) {
        GTEST_SKIP() << "direct Qwen FTH asset is not present";
    }

    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(model_path.string(), feather::demo::QwenBackend::kCommon), 0) << runner.LastError();
    EXPECT_EQ(runner.MaxContext(), 8);
    EXPECT_EQ(runner.TokensProcessed(), 0);
}

TEST(qwen_runner_test, LoadsPersistentContext128DirectSafetensorsExport) {
    const auto model_path = QwenModelDirectory() / "qwen3.5-0.8b_decode_bf16_ctx128.fth";
    if (!std::filesystem::is_regular_file(model_path)) {
        GTEST_SKIP() << "context-128 Qwen FTH asset is not present";
    }

    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(model_path.string(), feather::demo::QwenBackend::kCommon), 0) << runner.LastError();
    EXPECT_EQ(runner.MaxContext(), 128);
}

TEST(qwen_runner_test, DefaultExportMatchesReferenceFirstDecodeToken) {
    const auto model_path = QwenModelDirectory() / "qwen3.5-0.8b_decode_bf16_ctx8.fth";
    if (!std::filesystem::is_regular_file(model_path)) {
        GTEST_SKIP() << "default Qwen FTH asset is not present";
    }

    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(model_path.string(), feather::demo::QwenBackend::kCommon), 0) << runner.LastError();

    std::vector<int64_t> generated;
    ASSERT_EQ(runner.Generate({151644}, 1, {}, &generated), 0) << runner.LastError();
    // Verified with the official Qwen3.5 implementation on the same raw
    // safetensors checkpoint. This catches incorrect per-head q/gate packing
    // in the atomic full-attention graph.
    EXPECT_EQ(generated, (std::vector<int64_t>{10748}));
}

TEST(qwen_runner_test, X86BackendMatchesCommonForReferenceFirstDecodeToken) {
    const auto model_path = QwenModelDirectory() / "qwen3.5-0.8b_decode_bf16_ctx8.fth";
    if (!std::filesystem::is_regular_file(model_path)) {
        GTEST_SKIP() << "default Qwen FTH asset is not present";
    }

    feather::demo::QwenRunner common_runner;
    ASSERT_EQ(common_runner.Load(model_path.string(), feather::demo::QwenBackend::kCommon), 0)
        << common_runner.LastError();
    std::vector<int64_t> common_generated;
    ASSERT_EQ(common_runner.Generate({151644}, 1, {}, &common_generated), 0) << common_runner.LastError();

    feather::demo::QwenRunner x86_runner;
    ASSERT_EQ(x86_runner.Load(model_path.string(), feather::demo::QwenBackend::kX86), 0) << x86_runner.LastError();
    std::vector<int64_t> x86_generated;
    ASSERT_EQ(x86_runner.Generate({151644}, 1, {}, &x86_generated), 0) << x86_runner.LastError();

    EXPECT_EQ(common_generated, (std::vector<int64_t>{10748}));
    EXPECT_EQ(x86_generated, common_generated);
}

TEST(qwen_runner_test, X86BackendRunsFp8E4M3Context128) {
    const auto model_path = QwenModelDirectory() / "qwen3.5-0.8b_decode_fp8e4m3_ctx128.fth";
    if (!std::filesystem::is_regular_file(model_path)) {
        GTEST_SKIP() << "FP8 E4M3 Qwen FTH asset is not present";
    }

    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(model_path.string(), feather::demo::QwenBackend::kX86), 0) << runner.LastError();
    std::vector<int64_t> generated;
    ASSERT_EQ(runner.Generate({151644}, 1, {}, &generated), 0) << runner.LastError();
    ASSERT_EQ(generated.size(), 1U);
}

#ifdef FEATHER_WITH_CUDA
TEST(qwen_runner_test, CudaBackendMatchesCommonForReferenceFirstDecodeToken) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }
    const auto model_path = QwenModelDirectory() / "qwen3.5-0.8b_decode_bf16_ctx8.fth";
    if (!std::filesystem::is_regular_file(model_path)) {
        GTEST_SKIP() << "default Qwen FTH asset is not present";
    }

    feather::demo::QwenRunner common_runner;
    ASSERT_EQ(common_runner.Load(model_path.string(), feather::demo::QwenBackend::kCommon), 0)
        << common_runner.LastError();
    std::vector<int64_t> common_generated;
    ASSERT_EQ(common_runner.Generate({151644}, 1, {}, &common_generated), 0) << common_runner.LastError();

    feather::demo::QwenRunner cuda_runner;
    ASSERT_EQ(cuda_runner.Load(model_path.string(), feather::demo::QwenBackend::kCuda), 0)
        << cuda_runner.LastError();
    std::vector<int64_t> cuda_generated;
    ASSERT_EQ(cuda_runner.Generate({151644}, 1, {}, &cuda_generated), 0) << cuda_runner.LastError();

    EXPECT_EQ(common_generated, (std::vector<int64_t>{10748}));
    EXPECT_EQ(cuda_generated, common_generated);
}

TEST(qwen_runner_test, CudaFp8BackendCapturesDecodeGraphAfterWarmup) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }
    const auto model_path = QwenModelDirectory() / "qwen3.5-0.8b_decode_fp8e4m3_ctx128.fth";
    if (!std::filesystem::is_regular_file(model_path)) {
        GTEST_SKIP() << "FP8 E4M3 Qwen FTH asset is not present";
    }

    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(model_path.string(), feather::demo::QwenBackend::kCuda), 0) << runner.LastError();
    std::vector<int64_t> generated;
    ASSERT_EQ(runner.Generate({151644}, 3, {}, &generated), 0) << runner.LastError();
    ASSERT_EQ(generated.size(), 3U);
    EXPECT_GT(runner.CudaGraphLaunchCount(), 0);
}
#endif
