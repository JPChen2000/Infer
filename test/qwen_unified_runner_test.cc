// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <atomic>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef FEATHER_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include "demo/qwen_runner.h"
#include "model/model_io.h"

namespace {
using namespace feather;
using demo::QwenBackend;
using demo::QwenRunner;

model::ValueDesc UnifiedValue(const std::string& name, const std::vector<int64_t>& dims,
                              DataType dtype, bool constant = false) {
    model::ValueDesc value;
    value.tensor.name = name; value.tensor.dims = dims; value.tensor.data_type = dtype;
    value.constant = constant;
    return value;
}

model::NodeDesc UnifiedNode(const std::string& name, const std::string& op,
                            const std::vector<std::string>& inputs,
                            const std::vector<std::string>& outputs) {
    model::NodeDesc node;
    node.name = name; node.op_type = op; node.inputs = inputs; node.outputs = outputs;
    return node;
}

class UnifiedFixture {
 public:
    explicit UnifiedFixture(int64_t context = 32) {
        static std::atomic<int> id{0};
        path = std::filesystem::temp_directory_path() /
            ("qwen_unified_runner_fixture_" + std::to_string(id.fetch_add(1)) + ".fth");
        model::ModelDesc m;
        m.name = "unified_runner_fixture"; m.version = 1;
        m.metadata["qwen.execution.version"] = int64_t{2};
        m.metadata["qwen.execution.sequence"] = int64_t{1};
        m.metadata["qwen.cache.layout"] = std::string("position");
        m.metadata["qwen.weights.id"] = std::string("tiny_unified_recurrence");
        m.metadata["qwen.context.version"] = int64_t{1};
        m.metadata["qwen.context.default"] = context;
        m.metadata["qwen.context.maximum"] = int64_t{64};
        m.metadata["qwen.context.value.attention_mask"] = std::vector<int64_t>{3, 0};
        m.metadata["qwen.sequence.version"] = int64_t{1};
        m.metadata["qwen.sequence.default"] = int64_t{1};
        m.metadata["qwen.sequence.maximum"] = int64_t{8};
        // Explicitly bind only dimensions that vary with T. Other dimensions
        // equal to the canonical sequence length deliberately stay fixed.
        for (const auto& name : {"token_ids", "ones_raw", "ones", "gates", "v_raw", "v", "cores"})
            m.metadata[std::string("qwen.sequence.value.") + name] = std::vector<int64_t>{1, 0};
        for (const auto& name : {"shape_ones", "shape_v"})
            m.metadata[std::string("qwen.sequence.attribute.") + name + ".shape"] = std::vector<int64_t>{1, 0};
        m.graph.name = "unified";
        m.graph.inputs = {"token_ids", "position_id", "attention_mask", "recurrent_state_0"};
        m.graph.outputs = {"next_recurrent_state_0", "logits"};
        m.graph.values = {
            UnifiedValue("token_ids", {1,1}, DataType::INT64),
            UnifiedValue("position_id", {1}, DataType::INT64),
            UnifiedValue("attention_mask", {1,1,1,context}, DataType::BF16),
            UnifiedValue("recurrent_state_0", {1,1,1,2}, DataType::FP32),
            UnifiedValue("next_recurrent_state_0", {1,1,1,2}, DataType::FP32),
            UnifiedValue("one_table", {4,1}, DataType::FP32, true),
            UnifiedValue("gate_table", {4,1}, DataType::FP32, true),
            UnifiedValue("value_table", {4,2}, DataType::FP32, true),
            UnifiedValue("ones_raw", {1,1,1}, DataType::FP32),
            UnifiedValue("ones", {1,1,1,1}, DataType::FP32),
            UnifiedValue("gates", {1,1,1}, DataType::FP32),
            UnifiedValue("v_raw", {1,1,2}, DataType::FP32),
            UnifiedValue("v", {1,1,1,2}, DataType::FP32),
            UnifiedValue("cores", {1,1,1,2}, DataType::FP32),
            UnifiedValue("state_row", {1,2}, DataType::FP32),
            UnifiedValue("logit_weight", {2,4}, DataType::FP32, true),
            UnifiedValue("logit_row", {1,4}, DataType::FP32),
            UnifiedValue("logits", {1,1,4}, DataType::FP32),
        };
        auto ones = UnifiedNode("gather_ones", "Gather", {"one_table","token_ids"}, {"ones_raw"});
        ones.attributes["axis"] = int64_t{0};
        auto shape_ones = UnifiedNode("shape_ones", "Reshape", {"ones_raw"}, {"ones"});
        shape_ones.attributes["shape"] = std::vector<int64_t>{1,1,1,1};
        auto gates = UnifiedNode("gather_gates", "Gather", {"gate_table","token_ids"}, {"gates"});
        gates.attributes["axis"] = int64_t{0};
        auto values = UnifiedNode("gather_values", "Gather", {"value_table","token_ids"}, {"v_raw"});
        values.attributes["axis"] = int64_t{0};
        auto shape_v = UnifiedNode("shape_v", "Reshape", {"v_raw"}, {"v"});
        shape_v.attributes["shape"] = std::vector<int64_t>{1,1,1,2};
        auto recurrence = UnifiedNode("recurrence", "QwenGatedDeltaSequence",
            {"recurrent_state_0","ones","v","gates","gates","ones"},
            {"next_recurrent_state_0","cores"});
        auto state_row = UnifiedNode("state_row", "Reshape", {"next_recurrent_state_0"}, {"state_row"});
        state_row.attributes["shape"] = std::vector<int64_t>{1,2};
        auto projection = UnifiedNode("projection", "Gemm", {"state_row","logit_weight"}, {"logit_row"});
        auto logits = UnifiedNode("logits", "Reshape", {"logit_row"}, {"logits"});
        logits.attributes["shape"] = std::vector<int64_t>{1,1,4};
        m.graph.nodes = {ones,shape_ones,gates,values,shape_v,recurrence,state_row,projection,logits};
        std::unordered_map<std::string, std::shared_ptr<Tensor>> weights;
        const auto weight = [&](const std::string& name, std::vector<float> data, std::vector<int64_t> dims) {
            auto tensor = std::make_shared<Tensor>(); tensor->Assign<float>(data,dims); weights[name] = tensor;
        };
        weight("one_table", {1,1,1,1}, {4,1});
        weight("gate_table", {.5f,.5f,.5f,.5f}, {4,1});
        weight("value_table", {0,0,1,-1,-1,1,0,0}, {4,2});
        weight("logit_weight", {0,1,-1,0,0,-1,1,0}, {2,4});
        // With k=q=1 and beta=decay=.5, each state column follows
        // s[t]=.25*s[t-1]+.5*v[t]. Token 1 leaves a positive sign,
        // token 2 a negative sign, and zero tokens preserve that sign. The
        // projection selects token 1/2 for those signs and token 0 at reset.
        model::ModelWriter writer;
        EXPECT_TRUE(writer.Save(path.string(), m, weights));
    }
    ~UnifiedFixture() { std::error_code error; std::filesystem::remove(path,error); }
    std::filesystem::path path;
};

int ConsumeSerial(QwenRunner& runner, const std::vector<int64_t>& tokens) {
    for (int64_t token : tokens) if (runner.Consume({token}) != 0) return -1;
    return 0;
}

class QwenUnifiedRunnerTest : public testing::TestWithParam<QwenBackend> {
 protected:
    void SetUp() override {
#ifdef FEATHER_WITH_CUDA
        if (GetParam() == QwenBackend::kCuda) {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) GTEST_SKIP();
        }
#endif
    }
};

TEST_P(QwenUnifiedRunnerTest, PromptThirteenMatchesSerialFromNonzeroStateAndReusesFourPlans) {
    UnifiedFixture fixture;
    QwenRunner chunked, serial;
    ASSERT_EQ(chunked.Load(fixture.path.string(), GetParam(), 32), 0) << chunked.LastError();
    ASSERT_EQ(serial.Load(fixture.path.string(), GetParam(), 32), 0) << serial.LastError();
    ASSERT_EQ(chunked.Consume({1,0,0}), 0) << chunked.LastError();
    ASSERT_EQ(ConsumeSerial(serial, {1,0,0}), 0) << serial.LastError();
    ASSERT_EQ(ConsumeSerial(serial, std::vector<int64_t>(12,0)), 0) << serial.LastError();
    std::vector<int64_t> actual, expected;
    ASSERT_EQ(chunked.Generate(std::vector<int64_t>(13,0), 3, {}, &actual), 0) << chunked.LastError();
    ASSERT_EQ(serial.Generate({0}, 3, {}, &expected), 0) << serial.LastError();
    EXPECT_EQ(expected, (std::vector<int64_t>{1,1,1})); EXPECT_EQ(actual, expected);
    EXPECT_EQ(chunked.TokensProcessed(), 19); EXPECT_EQ(serial.TokensProcessed(), 19);
    EXPECT_EQ(chunked.CachedPlanCount(), 4u); EXPECT_EQ(serial.CachedPlanCount(), 1u);
    ASSERT_EQ(chunked.Reset(), 0) << chunked.LastError();
    EXPECT_EQ(chunked.CachedPlanCount(), 4u); EXPECT_EQ(chunked.TokensProcessed(), 0);
    ASSERT_EQ(chunked.Generate(std::vector<int64_t>(13,0), 1, {}, &actual), 0) << chunked.LastError();
    EXPECT_EQ(actual, (std::vector<int64_t>{0})); EXPECT_EQ(chunked.CachedPlanCount(), 4u);
    EXPECT_EQ(chunked.TokensProcessed(), 14);
}

TEST_P(QwenUnifiedRunnerTest, SplitPromptsPreserveStateWhenSwitchingSequencePlans) {
    UnifiedFixture fixture;
    QwenRunner whole, split;
    ASSERT_EQ(whole.Load(fixture.path.string(), GetParam(), 32), 0) << whole.LastError();
    ASSERT_EQ(split.Load(fixture.path.string(), GetParam(), 32), 0) << split.LastError();
    ASSERT_EQ(whole.Consume({2,0,0}), 0); ASSERT_EQ(split.Consume({2,0,0}), 0);
    ASSERT_EQ(whole.Consume(std::vector<int64_t>(13,0)), 0) << whole.LastError();
    for (int length : {3,5,5}) ASSERT_EQ(split.Consume(std::vector<int64_t>(length,0)), 0) << split.LastError();
    std::vector<int64_t> actual, expected;
    ASSERT_EQ(whole.Generate({0}, 3, {}, &expected), 0) << whole.LastError();
    ASSERT_EQ(split.Generate({0}, 3, {}, &actual), 0) << split.LastError();
    EXPECT_EQ(expected, (std::vector<int64_t>{2,2,2})); EXPECT_EQ(actual, expected);
    EXPECT_EQ(split.TokensProcessed(), whole.TokensProcessed());
    EXPECT_EQ(whole.CachedPlanCount(), 4u); EXPECT_EQ(split.CachedPlanCount(), 3u);
}

TEST_P(QwenUnifiedRunnerTest, EvaluationRefreshesRetainedActivationsAfterCaptureWarmup) {
    UnifiedFixture fixture;
    demo::QwenLoadOptions options;
    options.retain_logits = true;
    QwenRunner observed, reference;
    std::vector<float> logits, expected_logits, actual, expected;
    for (bool retain_intermediate : {true, false}) {
        options.additional_outputs = retain_intermediate ? std::vector<std::string>{"v_raw"} :
                                                          std::vector<std::string>{};
        ASSERT_EQ(observed.Load(fixture.path.string(), GetParam(), options), 0) << observed.LastError();
        ASSERT_EQ(reference.Load(fixture.path.string(), QwenBackend::kCommon, options), 0);
        const std::string name = retain_intermediate ? "v_raw" : "next_recurrent_state_0";
        for (const int64_t token : {1, 2, 0, 1, 2, 0, 2}) {
            ASSERT_EQ(observed.EvaluateToken(token, &logits), 0) << observed.LastError();
            ASSERT_EQ(reference.EvaluateToken(token, &expected_logits), 0);
            ASSERT_EQ(observed.ReadOutput(name, &actual), 0);
            ASSERT_EQ(reference.ReadOutput(name, &expected), 0);
            EXPECT_EQ(actual, expected) << "token=" << token << " output=" << name;
            EXPECT_EQ(logits, expected_logits);
        }
        EXPECT_EQ(observed.CudaGraphLaunchCount(), 0);
    }
}

TEST_P(QwenUnifiedRunnerTest, ContextGrowthPreservesStateAndRebuildsPlansLikeLargerFromStart) {
    UnifiedFixture fixture(16);
    QwenRunner grown, large;
    ASSERT_EQ(grown.Load(fixture.path.string(), GetParam(), 16), 0) << grown.LastError();
    ASSERT_EQ(large.Load(fixture.path.string(), GetParam(), 32), 0) << large.LastError();
    ASSERT_EQ(grown.Consume({1,0,0}), 0); ASSERT_EQ(large.Consume({1,0,0}), 0);
    ASSERT_EQ(grown.Consume(std::vector<int64_t>(4,0)), 0) << grown.LastError();
    ASSERT_EQ(large.Consume(std::vector<int64_t>(4,0)), 0) << large.LastError();
    std::vector<int64_t> warm_actual, warm_expected;
    ASSERT_EQ(grown.Generate({0}, 3, {}, &warm_actual), 0) << grown.LastError();
    ASSERT_EQ(large.Generate({0}, 3, {}, &warm_expected), 0) << large.LastError();
    EXPECT_EQ(warm_actual, warm_expected);
    if (GetParam() == QwenBackend::kCuda) EXPECT_GT(grown.CudaGraphLaunchCount(), 0);
    EXPECT_EQ(grown.CachedPlanCount(), 3u);
    ASSERT_EQ(grown.ReserveContext(32), 0) << grown.LastError();
    EXPECT_EQ(grown.TokensProcessed(), 11); EXPECT_EQ(grown.MaxContext(), 32);
    EXPECT_EQ(grown.ContextGrowthCount(), 1); EXPECT_EQ(grown.CachedPlanCount(), 1u);
    std::vector<int64_t> actual, expected;
    ASSERT_EQ(grown.Generate(std::vector<int64_t>(13,0), 3, {}, &actual), 0) << grown.LastError();
    ASSERT_EQ(large.Generate(std::vector<int64_t>(13,0), 3, {}, &expected), 0) << large.LastError();
    EXPECT_EQ(expected, (std::vector<int64_t>{1,1,1})); EXPECT_EQ(actual, expected);
    EXPECT_EQ(grown.TokensProcessed(), large.TokensProcessed()); EXPECT_EQ(grown.CachedPlanCount(), 3u);
}

TEST_P(QwenUnifiedRunnerTest, AutomaticGrowthAcrossPromptAndGenerationKeepsUnifiedState) {
    UnifiedFixture fixture(8);
    QwenRunner automatic, large;
    ASSERT_EQ(automatic.Load(fixture.path.string(), GetParam()), 0) << automatic.LastError();
    ASSERT_EQ(large.Load(fixture.path.string(), GetParam(), 32), 0) << large.LastError();
    ASSERT_EQ(automatic.Consume({2,0,0}), 0); ASSERT_EQ(large.Consume({2,0,0}), 0);
    std::vector<int64_t> actual, expected;
    ASSERT_EQ(automatic.Generate(std::vector<int64_t>(13,0), 3, {}, &actual), 0) << automatic.LastError();
    ASSERT_EQ(large.Generate(std::vector<int64_t>(13,0), 3, {}, &expected), 0) << large.LastError();
    EXPECT_EQ(expected, (std::vector<int64_t>{2,2,2})); EXPECT_EQ(actual, expected);
    EXPECT_EQ(automatic.MaxContext(), 32); EXPECT_EQ(automatic.ContextGrowthCount(), 2);
    EXPECT_EQ(automatic.TokensProcessed(), 19); EXPECT_TRUE(automatic.AutoContextGrowthEnabled());
}

TEST_P(QwenUnifiedRunnerTest, CancellationBetweenChunksKeepsConsumedStateUsable) {
    UnifiedFixture fixture;
    QwenRunner cancelled, serial;
    ASSERT_EQ(cancelled.Load(fixture.path.string(), GetParam(), 32), 0);
    ASSERT_EQ(serial.Load(fixture.path.string(), GetParam(), 32), 0);
    ASSERT_EQ(cancelled.Consume({1,0,0}), 0);
    ASSERT_EQ(ConsumeSerial(serial, {1,0,0}), 0);
    demo::QwenGenerationOptions options;
    options.max_new_tokens = 1;
    int checks = 0;
    options.should_cancel = [&] { return ++checks > 1; };
    std::vector<int64_t> output;
    ASSERT_EQ(cancelled.Generate(std::vector<int64_t>(13,0), options, {}, &output), 0)
        << cancelled.LastError();
    EXPECT_TRUE(output.empty());
    EXPECT_EQ(cancelled.LastFinishReason(), demo::QwenFinishReason::kCancelled);
    EXPECT_EQ(cancelled.LastPromptTokensConsumed(), 8);
    EXPECT_EQ(cancelled.TokensProcessed(), 11);
    ASSERT_EQ(cancelled.Consume(std::vector<int64_t>(5,0)), 0);
    ASSERT_EQ(ConsumeSerial(serial, std::vector<int64_t>(13,0)), 0);
    std::vector<int64_t> expected;
    ASSERT_EQ(cancelled.Generate({0}, 2, {}, &output), 0) << cancelled.LastError();
    ASSERT_EQ(serial.Generate({0}, 2, {}, &expected), 0) << serial.LastError();
    EXPECT_EQ(output, expected);
    EXPECT_EQ(output, (std::vector<int64_t>{1,1}));
}

INSTANTIATE_TEST_SUITE_P(CommonAndCuda, QwenUnifiedRunnerTest, testing::Values(QwenBackend::kCommon
#ifdef FEATHER_WITH_CUDA
    , QwenBackend::kCuda
#endif
));
} // namespace
