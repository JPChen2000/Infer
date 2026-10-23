// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <atomic>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "demo/qwen_runner.h"
#include "model/model_io.h"
#include "util/bf16.h"

#ifdef FEATHER_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace {
using namespace feather;
using demo::QwenRunner;
using demo::QwenBackend;

model::ValueDesc Value(const std::string& name, std::vector<int64_t> dims, DataType dtype,
                       bool constant = false) {
    model::ValueDesc result;
    result.tensor.name = name; result.tensor.dims = std::move(dims); result.tensor.data_type = dtype;
    result.constant = constant;
    return result;
}

model::NodeDesc Node(const std::string& name, const std::string& op, std::vector<std::string> inputs,
                     const std::string& output) {
    model::NodeDesc result;
    result.name = name; result.op_type = op; result.inputs = std::move(inputs); result.outputs = {output};
    return result;
}

std::shared_ptr<Tensor> Floats(std::vector<float> values, std::vector<int64_t> dims) {
    auto result = std::make_shared<Tensor>(); result->Assign<float>(values, dims); return result;
}

class GrowthFixture {
 public:
    explicit GrowthFixture(bool append_cache = false, bool incompatible_growth = false) {
        static std::atomic<int> id{0};
        path = std::filesystem::temp_directory_path() /
            ("qwen_growth_fixture_" + std::to_string(id.fetch_add(1)) + ".fth");
        model::ModelDesc m;
        m.name = "growth_fixture";
        m.version = 1;
        m.metadata["qwen.context.version"] = int64_t{1};
        m.metadata["qwen.context.default"] = int64_t{4};
        m.metadata["qwen.context.maximum"] = int64_t{16};
        m.metadata["qwen.context.value.attention_mask"] = std::vector<int64_t>{3, 0};
        m.graph.name = "decode";
        m.graph.inputs = {"token_ids", "position_id", "attention_mask", "recurrent_state_0"};
        m.graph.outputs = {"next_recurrent_state_0", "logits"};
        m.graph.values = {
            Value("token_ids", {1, 1}, DataType::INT64), Value("position_id", {1}, DataType::INT64),
            Value("attention_mask", {1, 1, 1, 4}, DataType::BF16),
            Value("recurrent_state_0", {1, 4}, DataType::FP32),
            Value("transition", {4, 4}, DataType::FP32, true), Value("bias", {4}, DataType::FP32, true),
            Value("next_state_raw", {1, 4}, DataType::FP32),
            Value("next_recurrent_state_0", {1, 4}, DataType::FP32),
            Value("logits", {1, 1, 4}, DataType::FP32),
        };
        m.graph.nodes = {
            Node("transition", "Gemm", {"recurrent_state_0", "transition", "bias"}, "next_state_raw"),
            Node("next_state", "Identity", {"next_state_raw"}, "next_recurrent_state_0"),
            Node("logits", "Reshape", {"next_state_raw"}, "logits")};
        m.graph.nodes.back().attributes["shape"] = std::vector<int64_t>{1, 1, 4};
        // Valid context expressions can still produce an unsupported graph:
        // enlarging this reshape cannot consume the fixed four-state vector.
        if (incompatible_growth)
            m.metadata["qwen.context.attribute.logits.shape"] = std::vector<int64_t>{2, 0};
        std::unordered_map<std::string, std::shared_ptr<Tensor>> weights;
        weights["transition"] = Floats({0,0,0,0, 0,-2,2,0, 0,0,0,0, 0,0,0,0}, {4,4});
        weights["bias"] = Floats({0,10,0,0}, {4});
        if (append_cache) {
            m.metadata["qwen.context.value.cache"] = std::vector<int64_t>{2, -1};
            m.metadata["qwen.context.constant.last_index"] = std::vector<int64_t>{0, -2};
            m.graph.inputs.push_back("cache"); m.graph.outputs.push_back("next_cache");
            for (const auto& value : {
                Value("cache", {1, 2, 3, 1}, DataType::FP32),
                Value("token_float", {1, 1}, DataType::FP32),
                Value("cache_row", {1, 2}, DataType::FP32),
                Value("cache_weight", {1, 2}, DataType::FP32, true),
                Value("next_cache", {1, 2, 1, 1}, DataType::FP32),
                Value("last_index", {1}, DataType::INT64, true),
                Value("last_cache", {1, 2, 1, 1}, DataType::FP32),
                Value("last_row", {1, 2}, DataType::FP32),
                Value("logit_weight", {2, 4}, DataType::FP32, true),
                Value("logit_bias", {4}, DataType::FP32, true),
                Value("cache_logits", {1, 4}, DataType::FP32)}) m.graph.values.push_back(value);
            m.graph.nodes.pop_back();
            auto cast = Node("token_cast", "Cast", {"token_ids"}, "token_float");
            cast.attributes["to"] = int64_t{1};
            auto next = Node("cache_shape", "Reshape", {"cache_row"}, "next_cache");
            next.attributes["shape"] = std::vector<int64_t>{1,2,1,1};
            auto last = Node("last_cache", "Gather", {"cache", "last_index"}, "last_cache");
            last.attributes["axis"] = int64_t{2};
            auto row = Node("last_row", "Reshape", {"last_cache"}, "last_row");
            row.attributes["shape"] = std::vector<int64_t>{1,2};
            auto logits = Node("logits", "Reshape", {"cache_logits"}, "logits");
            logits.attributes["shape"] = std::vector<int64_t>{1,1,4};
            for (const auto& node : {cast,
                Node("cache_values", "Gemm", {"token_float", "cache_weight"}, "cache_row"), next, last, row,
                Node("cache_logits", "Gemm", {"last_row", "logit_weight", "logit_bias"}, "cache_logits"),
                logits}) m.graph.nodes.push_back(node);
            weights["cache_weight"] = Floats({1,-1}, {1,2});
            weights["logit_weight"] = Floats({1,0,0,0, 0,1,0,0}, {2,4});
            weights["logit_bias"] = Floats({0,0,1.5f,0}, {4});
            auto index = std::make_shared<Tensor>(); index->Assign<int64_t>({2}, {1});
            weights["last_index"] = index;
        }
        model::ModelWriter writer;
        EXPECT_TRUE(writer.Save(path.string(), m, weights));
    }
    ~GrowthFixture() { std::error_code error; std::filesystem::remove(path, error); }
    std::filesystem::path path;
};

TEST(QwenRunnerGrowthTest, PreservesNonzeroRecurrentStateAndProcessedPosition) {
    GrowthFixture fixture;
    QwenRunner grown, large;
    ASSERT_EQ(grown.Load(fixture.path.string(), QwenBackend::kCommon), 0) << grown.LastError();
    ASSERT_EQ(large.Load(fixture.path.string(), QwenBackend::kCommon, 12), 0) << large.LastError();
    EXPECT_TRUE(grown.AutoContextGrowthEnabled());
    ASSERT_EQ(grown.Consume({0,0,0}), 0); ASSERT_EQ(large.Consume({0,0,0}), 0);
    ASSERT_EQ(grown.ReserveContext(12), 0) << grown.LastError();
    EXPECT_EQ(grown.MaxContext(), 12); EXPECT_EQ(grown.TokensProcessed(), 3);
    EXPECT_EQ(grown.ContextGrowthCount(), 1); EXPECT_TRUE(grown.AutoContextGrowthEnabled());
    std::vector<int64_t> actual, expected;
    ASSERT_EQ(grown.Generate({0}, 3, {}, &actual), 0) << grown.LastError();
    ASSERT_EQ(large.Generate({0}, 3, {}, &expected), 0) << large.LastError();
    EXPECT_EQ(expected, (std::vector<int64_t>{2,1,2})); EXPECT_EQ(actual, expected);
    EXPECT_EQ(grown.TokensProcessed(), large.TokensProcessed());
    ASSERT_EQ(grown.Reset(), 0);
    EXPECT_EQ(grown.TokensProcessed(), 0); EXPECT_EQ(grown.MaxContext(), 12);
}

TEST(QwenRunnerGrowthTest, RightAlignsLegacyCacheAcrossHeadsWithNewRowStrides) {
    GrowthFixture fixture(true);
    QwenRunner grown, large;
    ASSERT_EQ(grown.Load(fixture.path.string(), QwenBackend::kCommon, 4), 0) << grown.LastError();
    ASSERT_EQ(large.Load(fixture.path.string(), QwenBackend::kCommon, 8), 0) << large.LastError();
    ASSERT_EQ(grown.Consume({1,2}), 0) << grown.LastError();
    ASSERT_EQ(large.Consume({1,2}), 0) << large.LastError();
    ASSERT_EQ(grown.ReserveContext(8), 0) << grown.LastError();
    std::vector<int64_t> actual, expected;
    ASSERT_EQ(grown.Generate({1}, 1, {}, &actual), 0) << grown.LastError();
    ASSERT_EQ(large.Generate({1}, 1, {}, &expected), 0) << large.LastError();
    EXPECT_EQ(expected, (std::vector<int64_t>{0})); EXPECT_EQ(actual, expected);
}

TEST(QwenRunnerGrowthTest, FailedReplacementLeavesTheExistingSessionUsable) {
    GrowthFixture fixture(false, true);
    QwenRunner runner;
    ASSERT_EQ(runner.Load(fixture.path.string(), QwenBackend::kCommon, 4), 0) << runner.LastError();
    ASSERT_EQ(runner.Consume({0}), 0);
    EXPECT_NE(runner.ReserveContext(8), 0);
    EXPECT_EQ(runner.MaxContext(), 4); EXPECT_EQ(runner.TokensProcessed(), 1);
    EXPECT_EQ(runner.ContextGrowthCount(), 0);
    std::vector<int64_t> generated;
    ASSERT_EQ(runner.Generate({0}, 1, {}, &generated), 0) << runner.LastError();
    EXPECT_EQ(generated, (std::vector<int64_t>{2}));
}

TEST(QwenRunnerGrowthTest, RejectsInvalidRequestsAndAllowsExplicitManualGrowth) {
    GrowthFixture fixture;
    QwenRunner runner;
    EXPECT_NE(runner.ReserveContext(8), 0);
    ASSERT_EQ(runner.Load(fixture.path.string(), QwenBackend::kCommon, 4), 0);
    EXPECT_FALSE(runner.AutoContextGrowthEnabled());
    EXPECT_NE(runner.ReserveContext(-1), 0); EXPECT_NE(runner.ReserveContext(17), 0);
    ASSERT_EQ(runner.ReserveContext(4), 0); EXPECT_EQ(runner.ContextGrowthCount(), 0);
    ASSERT_EQ(runner.ReserveContext(8), 0) << runner.LastError();
    EXPECT_EQ(runner.MaxContext(), 8); EXPECT_FALSE(runner.AutoContextGrowthEnabled());
    ASSERT_EQ(runner.ReserveContext(6), 0); EXPECT_EQ(runner.MaxContext(), 8);
    EXPECT_EQ(runner.ContextGrowthCount(), 1);
}

TEST(QwenRunnerGrowthTest, AutomaticallyGrowsGenerationAndKeepsExplicitCapacityBounded) {
    GrowthFixture fixture;
    QwenRunner automatic, large, bounded;
    ASSERT_EQ(automatic.Load(fixture.path.string(), QwenBackend::kCommon), 0);
    ASSERT_EQ(large.Load(fixture.path.string(), QwenBackend::kCommon, 8), 0);
    ASSERT_EQ(bounded.Load(fixture.path.string(), QwenBackend::kCommon, 4), 0);
    std::vector<int64_t> actual, expected, capped;
    ASSERT_EQ(automatic.Generate({0}, 6, {}, &actual), 0) << automatic.LastError();
    ASSERT_EQ(large.Generate({0}, 6, {}, &expected), 0) << large.LastError();
    EXPECT_EQ(actual, expected); EXPECT_EQ(actual.size(), 6u);
    EXPECT_EQ(automatic.MaxContext(), 8); EXPECT_EQ(automatic.ContextGrowthCount(), 1);
    ASSERT_EQ(bounded.Generate({0}, 6, {}, &capped), 0) << bounded.LastError();
    EXPECT_EQ(capped.size(), 3u); EXPECT_EQ(bounded.MaxContext(), 4);
    EXPECT_EQ(bounded.ContextGrowthCount(), 0);
    EXPECT_EQ(bounded.LastFinishReason(), demo::QwenFinishReason::kContextLimit);
}

class PositionGrowthFixture {
 public:
    PositionGrowthFixture() {
        path = std::filesystem::temp_directory_path() / "qwen_position_growth_fixture.fth";
        model::ModelDesc m;
        m.name = "position_growth"; m.version = 1;
        m.metadata["qwen.context.version"] = int64_t{1};
        m.metadata["qwen.context.default"] = int64_t{4};
        m.metadata["qwen.context.maximum"] = int64_t{16};
        m.metadata["qwen.context.value.attention_mask"] = std::vector<int64_t>{3,0};
        m.metadata["qwen.context.value.cache_k"] = std::vector<int64_t>{2,0};
        m.metadata["qwen.context.value.cache_v"] = std::vector<int64_t>{2,0};
        m.metadata["qwen.execution.version"] = int64_t{1};
        m.metadata["qwen.execution.sequence"] = int64_t{1};
        m.metadata["qwen.cache.layout"] = std::string("position");
        m.metadata["qwen.weights.id"] = std::string("position_growth_fixture");
        m.graph.name = "decode";
        m.graph.inputs = {"token_ids", "position_id", "attention_mask", "cache_k", "cache_v"};
        m.graph.outputs = {"logits"};
        m.graph.values = {
            Value("token_ids", {1,1}, DataType::INT64), Value("position_id", {1}, DataType::INT64),
            Value("attention_mask", {1,1,1,4}, DataType::BF16),
            Value("cache_k", {1,2,4,1}, DataType::BF16), Value("cache_v", {1,2,4,1}, DataType::BF16),
            Value("embedding", {4,2}, DataType::BF16, true), Value("embedded", {1,1,2}, DataType::BF16),
            Value("q", {1,2,1,1}, DataType::BF16, true), Value("k", {1,2,1,1}, DataType::BF16, true),
            Value("v", {1,2,1,1}, DataType::BF16), Value("attended", {1,2,1,1}, DataType::BF16),
            Value("float_attended", {1,2,1,1}, DataType::FP32), Value("row", {1,2}, DataType::FP32),
            Value("weight", {2,4}, DataType::FP32, true), Value("bias", {4}, DataType::FP32, true),
            Value("raw_logits", {1,4}, DataType::FP32), Value("logits", {1,1,4}, DataType::FP32)};
        auto gather = Node("embedding", "Gather", {"embedding", "token_ids"}, "embedded");
        gather.attributes["axis"] = int64_t{0};
        auto reshape = Node("value", "Reshape", {"embedded"}, "v");
        reshape.attributes["shape"] = std::vector<int64_t>{1,2,1,1};
        auto attention = Node("attention", "CachedAttention", {"q","k","v","cache_k","cache_v","position_id"}, "attended");
        auto cast = Node("cast", "Cast", {"attended"}, "float_attended");
        cast.attributes["to"] = int64_t{1};
        auto row = Node("row", "Reshape", {"float_attended"}, "row");
        row.attributes["shape"] = std::vector<int64_t>{1,2};
        auto logits = Node("logits", "Reshape", {"raw_logits"}, "logits");
        logits.attributes["shape"] = std::vector<int64_t>{1,1,4};
        m.graph.nodes = {gather, reshape, attention, cast, row,
            Node("projection", "Gemm", {"row","weight","bias"}, "raw_logits"), logits};
        auto bf16 = [](const std::vector<float>& values, const std::vector<int64_t>& dims) {
            std::vector<BFloat16> data;
            for (float value : values) data.push_back(BFloat16{FloatToBFloat16(value)});
            auto tensor = std::make_shared<Tensor>(); tensor->Assign<BFloat16>(data,dims); return tensor;
        };
        std::unordered_map<std::string, std::shared_ptr<Tensor>> weights;
        weights["embedding"] = bf16({0,0,4,-4,2,-2,0,0}, {4,2});
        weights["q"] = bf16({0,0}, {1,2,1,1}); weights["k"] = bf16({0,0}, {1,2,1,1});
        weights["weight"] = Floats({1,0,0,0,0,1,0,0}, {2,4}); weights["bias"] = Floats({0,0,1,0}, {4});
        model::ModelWriter writer;
        EXPECT_TRUE(writer.Save(path.string(), m, weights));
    }
    ~PositionGrowthFixture() { std::error_code error; std::filesystem::remove(path,error); }
    std::filesystem::path path;
};

void CheckPositionGrowth(QwenBackend backend) {
    PositionGrowthFixture fixture;
    QwenRunner grown, large;
    ASSERT_EQ(grown.Load(fixture.path.string(), backend, 4), 0) << grown.LastError();
    ASSERT_EQ(large.Load(fixture.path.string(), backend, 8), 0) << large.LastError();
    ASSERT_EQ(grown.Consume({1,2}), 0) << grown.LastError();
    ASSERT_EQ(large.Consume({1,2}), 0) << large.LastError();
    ASSERT_EQ(grown.ReserveContext(8), 0) << grown.LastError();
    std::vector<int64_t> actual, expected;
    ASSERT_EQ(grown.Generate({0}, 1, {}, &actual), 0) << grown.LastError();
    ASSERT_EQ(large.Generate({0}, 1, {}, &expected), 0) << large.LastError();
    EXPECT_EQ(expected, (std::vector<int64_t>{0})); EXPECT_EQ(actual, expected);
    EXPECT_EQ(grown.TokensProcessed(), 4);
}

TEST(QwenRunnerGrowthTest, PreservesPositionCachePrefixesAcrossHeadStrides) {
    CheckPositionGrowth(QwenBackend::kCommon);
}

#ifdef FEATHER_WITH_CUDA
TEST(QwenRunnerGrowthTest, CudaPreservesDeviceAuthoritativePositionCacheAndRebuildsCaptures) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) GTEST_SKIP();
    CheckPositionGrowth(QwenBackend::kCuda);
}
#endif
} // namespace
