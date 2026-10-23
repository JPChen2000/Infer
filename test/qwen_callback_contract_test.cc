// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include <atomic>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <unistd.h>

#include "demo/qwen_chat_session.h"
#include "demo/qwen_runner.h"
#include "model/model_io.h"

namespace {
using namespace feather;
using demo::QwenBackend;
using demo::QwenRunner;

class CallbackModel {
 public:
    CallbackModel() {
        static std::atomic<int> id{0};
        path = std::filesystem::temp_directory_path() /
            ("qwen_callback_contract_" + std::to_string(getpid()) + "_" + std::to_string(id++) + ".fth");
        model::ModelDesc m;
        m.name = "callback_contract"; m.version = 1; m.graph.name = "decode";
        m.metadata["qwen.context.version"] = int64_t{1};
        m.metadata["qwen.context.default"] = int64_t{8};
        m.metadata["qwen.context.maximum"] = int64_t{32};
        m.metadata["qwen.context.value.attention_mask"] = std::vector<int64_t>{3, 0};
        m.graph.inputs = {"token_ids", "position_id", "attention_mask", "state"};
        m.graph.outputs = {"logits", "next_state"};
        const auto value = [&](std::string name, std::vector<int64_t> dims, DataType dtype, bool constant = false) {
            model::ValueDesc v; v.tensor.name = std::move(name); v.tensor.dims = std::move(dims);
            v.tensor.data_type = dtype; v.constant = constant; m.graph.values.push_back(std::move(v));
        };
        value("token_ids", {1,1}, DataType::INT64); value("position_id", {1}, DataType::INT64);
        value("attention_mask", {1,1,1,8}, DataType::BF16); value("state", {1}, DataType::FP32);
        value("next_state", {1}, DataType::FP32); value("logits", {1,1,4}, DataType::FP32);
        value("table", {4,4}, DataType::FP32, true);
        model::NodeDesc lookup; lookup.name = "lookup"; lookup.op_type = "Gather";
        lookup.inputs = {"table", "token_ids"}; lookup.outputs = {"logits"}; lookup.attributes["axis"] = int64_t{0};
        model::NodeDesc state; state.name = "state"; state.op_type = "Identity";
        state.inputs = {"state"}; state.outputs = {"next_state"};
        m.graph.nodes = {lookup, state};
        // Tokens follow the literal cycle 0 -> 1 -> 3 -> 2 -> 2.
        auto table = std::make_shared<Tensor>();
        table->Assign<float>({0,5,0,0, 0,0,0,5, 0,0,5,0, 0,0,5,0}, {4,4});
        model::ModelWriter writer;
        EXPECT_TRUE(writer.Save(path.string(), m, {{"table", table}}));
    }
    ~CallbackModel() { std::error_code error; std::filesystem::remove(path, error); }
    std::filesystem::path path;
};

// A callback-induced unload previously dereferenced a retired plan. Keep the
// regression isolated so a signal is reported as a failure instead of killing
// every other contract test.
TEST(QwenCallbackContractTest, CancellationCallbackCannotUnloadActiveRequest) {
    CallbackModel model;
    EXPECT_EXIT({
        QwenRunner runner;
        if (runner.Load(model.path.string(), QwenBackend::kCommon) != 0) _exit(1);
        demo::QwenGenerationOptions options; options.max_new_tokens = 1;
        int nested = 0;
        options.should_cancel = [&] { nested = runner.Load("/tmp/missing_qwen_callback_model.fth"); return false; };
        std::vector<int64_t> output;
        const auto status = runner.Generate({0}, options, {}, &output);
        _exit(status == 0 && nested != 0 && runner.TokensProcessed() == 2 &&
            output == std::vector<int64_t>{1} ? 0 : 2);
    }, testing::ExitedWithCode(0), "");
}

TEST(QwenCallbackContractTest, CreateRequestCannotReplaceItsOwnOwner) {
    CallbackModel model;
    EXPECT_EXIT({
        auto owner = std::make_unique<QwenRunner>();
        if (owner->Load(model.path.string()) != 0 || owner->Consume({0}) != 0) _exit(1);
        const auto* original = owner.get();
        const auto revision = owner->StateRevision();
        if (owner->CreateRequest(&owner) == 0 || owner.get() != original) _exit(2);
        if (owner->StateRevision() != revision || owner->TokensProcessed() != 1) _exit(3);
        if (owner->Reset() != 0) _exit(4);
        std::unique_ptr<QwenRunner> separate;
        if (owner->CreateRequest(&separate) != 0 || !separate ||
            !owner->SharesModelResourcesWith(*separate)) _exit(5);
        _exit(0);
    }, testing::ExitedWithCode(0), "");
}

TEST(QwenCallbackContractTest, PublicMutationsCannotReenterEitherCallbackPhase) {
    CallbackModel model;
    for (const bool during_prompt : {true, false}) {
        for (int mutation = 0; mutation < 9; ++mutation) {
            SCOPED_TRACE(testing::Message() << "prompt=" << during_prompt << " mutation=" << mutation);
            QwenRunner runner;
            demo::QwenLoadOptions load; load.retain_logits = true;
            ASSERT_EQ(runner.Load(model.path.string(), QwenBackend::kCommon, load), 0);
            const auto revision = runner.StateRevision();
            const auto mutate = [&] {
                std::vector<int64_t> output;
                std::vector<float> logits;
                std::unique_ptr<QwenRunner> request;
                switch (mutation) {
                    case 0: EXPECT_NE(runner.Reset(), 0); break;
                    case 1: EXPECT_NE(runner.Consume({2}), 0); break;
                    case 2: EXPECT_NE(runner.EvaluateToken(2, &logits), 0); break;
                    case 3: EXPECT_NE(runner.Generate({2}, 1, {}, &output), 0); break;
                    case 4: EXPECT_NE(runner.ReserveContext(16), 0); break;
                    case 5: EXPECT_NE(runner.Load(model.path.string()), 0); break;
                    case 6: EXPECT_NE(runner.Load(model.path.string(), QwenBackend::kCommon, load), 0); break;
                    case 7: EXPECT_NE(runner.GenerateStream({2}, 1, {}, [](int64_t) {}), 0); break;
                    case 8: EXPECT_NE(runner.CreateRequest(&request), 0); EXPECT_FALSE(request); break;
                }
            };
            demo::QwenGenerationOptions options; options.max_new_tokens = 1;
            options.should_cancel = [&] { if (during_prompt) mutate(); return false; };
            std::vector<int64_t> output;
            ASSERT_EQ(runner.GenerateStream({0}, options, {}, [&](int64_t token) {
                if (!during_prompt) mutate();
                output.push_back(token);
            }), 0) << runner.LastError();
            EXPECT_EQ(output, (std::vector<int64_t>{1}));
            EXPECT_EQ(runner.TokensProcessed(), 2);
            EXPECT_EQ(runner.LastPromptTokensConsumed(), 1);
            EXPECT_EQ(runner.StateRevision(), revision + 1);
            EXPECT_EQ(runner.MaxContext(), 8);
            EXPECT_TRUE(runner.LastError().empty());
            ASSERT_EQ(runner.Reset(), 0);  // The guard must release after return.
        }
    }
}

TEST(QwenCallbackContractTest, CallbackExceptionsReturnErrorsAndReleaseOperationGuard) {
    CallbackModel model;
    QwenRunner runner;
    ASSERT_EQ(runner.Load(model.path.string()), 0);
    demo::QwenGenerationOptions options; options.max_new_tokens = 1;
    int checks = 0;
    options.should_cancel = [&]() -> bool {
        if (++checks == 2) throw std::runtime_error("cancel observer failed");
        return false;
    };
    std::vector<int64_t> output;
    EXPECT_NO_THROW(EXPECT_NE(runner.Generate({0,1,3}, options, {}, &output), 0));
    EXPECT_EQ(runner.TokensProcessed(), 1);
    EXPECT_EQ(runner.LastPromptTokensConsumed(), 1);
    EXPECT_NE(runner.LastError().find("cancel observer failed"), std::string::npos);
    ASSERT_EQ(runner.Reset(), 0);
    EXPECT_NO_THROW(EXPECT_NE(runner.GenerateStream({0}, 1, {}, [](int64_t) { throw 7; }), 0));
    EXPECT_EQ(runner.TokensProcessed(), 1);  // Throwing observer's token was not consumed.
    EXPECT_FALSE(runner.LastError().empty());
    ASSERT_EQ(runner.Reset(), 0);
    ASSERT_EQ(runner.Generate({0}, 1, {}, &output), 0);
    EXPECT_EQ(output, (std::vector<int64_t>{1}));
}

TEST(QwenCallbackContractTest, ThrowingChatObserverRestoresExactCompleteHistory) {
    CallbackModel model;
    QwenRunner runner;
    ASSERT_EQ(runner.Load(model.path.string()), 0);
    demo::QwenChatSession chat(&runner, {}, 2, {0}, {});
    demo::QwenGenerationOptions options; options.max_new_tokens = 1;
    ASSERT_EQ(chat.GenerateTurn({0}, options, {}), 0);
    const auto history = chat.History().ReplayTokens();
    const auto processed = runner.TokensProcessed();
    EXPECT_NO_THROW(EXPECT_NE(chat.GenerateTurn({0}, options,
        [](int64_t) { throw std::runtime_error("stream observer failed"); }), 0));
    EXPECT_EQ(chat.History().ReplayTokens(), history);
    EXPECT_EQ(runner.TokensProcessed(), processed);
    EXPECT_NE(chat.LastError().find("stream observer failed"), std::string::npos);
    ASSERT_EQ(chat.GenerateTurn({0}, options, {}), 0) << chat.LastError();
}

TEST(QwenCallbackContractTest, GenerationAcceptsAliasedPromptAndStopOutputVectors) {
    CallbackModel model;
    QwenRunner runner;
    ASSERT_EQ(runner.Load(model.path.string()), 0);
    std::vector<int64_t> prompt_and_output{0,1,3};
    ASSERT_EQ(runner.Generate(prompt_and_output, 1, {}, &prompt_and_output), 0) << runner.LastError();
    EXPECT_EQ(prompt_and_output, (std::vector<int64_t>{2}));
    EXPECT_EQ(runner.LastPromptTokensConsumed(), 3);
    EXPECT_EQ(runner.TokensProcessed(), 4);
    ASSERT_EQ(runner.Reset(), 0);
    std::vector<int64_t> stops_and_output{1};
    ASSERT_EQ(runner.Generate({0}, 3, stops_and_output, &stops_and_output), 0);
    EXPECT_EQ(stops_and_output, (std::vector<int64_t>{1}));
    EXPECT_EQ(runner.LastFinishReason(), demo::QwenFinishReason::kEos);
    EXPECT_EQ(runner.TokensProcessed(), 2);
    ASSERT_EQ(runner.Reset(), 0);
    std::vector<int64_t> all{0,1};
    ASSERT_EQ(runner.Generate(all, 3, all, &all), 0);
    EXPECT_EQ(all, (std::vector<int64_t>{3,2,2}));
    EXPECT_EQ(runner.LastPromptTokensConsumed(), 2);
    EXPECT_EQ(runner.TokensProcessed(), 5);
}
}  // namespace
