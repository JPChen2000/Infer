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

class OwnershipFixture {
 public:
    explicit OwnershipFixture(int64_t context = 32) {
        static std::atomic<int> id{0};
        path = std::filesystem::temp_directory_path() /
            ("qwen_ownership_fixture_" + std::to_string(id.fetch_add(1)) + ".fth");
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
    ~OwnershipFixture() { std::error_code error; std::filesystem::remove(path,error); }
    std::filesystem::path path;
};

// Losing shared resource ownership would reopen the removed source or leave
// dangling mappings after the source request is destroyed. Sharing request
// state would change the literal token outputs or the independent revision.
TEST(QwenOwnershipTest, RequestsShareResourcesAndKeepIndependentStateAfterOwnerDestruction) {
    OwnershipFixture fixture;
    auto source = std::make_unique<QwenRunner>();
    ASSERT_EQ(source->Load(fixture.path.string(), QwenBackend::kCommon, 32), 0) << source->LastError();
    ASSERT_EQ(source->Consume({1}), 0);
    std::filesystem::remove(fixture.path);
    std::unique_ptr<QwenRunner> request;
    ASSERT_EQ(source->CreateRequest(&request), 0) << source->LastError();
    ASSERT_TRUE(request);
    EXPECT_TRUE(source->SharesModelResourcesWith(*request));
    EXPECT_EQ(request->TokensProcessed(), 0);
    const auto revision = request->StateRevision();
    ASSERT_EQ(source->Consume({1, 0, 0}), 0);
    EXPECT_EQ(request->StateRevision(), revision);
    EXPECT_EQ(request->TokensProcessed(), 0);
    source.reset();
    std::vector<int64_t> generated;
    ASSERT_EQ(request->Generate(std::vector<int64_t>(13, 0), 1, {}, &generated), 0) << request->LastError();
    EXPECT_EQ(generated, (std::vector<int64_t>{0}));
    EXPECT_EQ(request->CachedPlanCount(), 3u);  // Sequence lengths 8, 4, and 1.
    ASSERT_EQ(request->Reset(), 0);
    ASSERT_EQ(request->Generate({2, 0, 0, 0}, 1, {}, &generated), 0) << request->LastError();
    EXPECT_EQ(generated, (std::vector<int64_t>{2}));
}

TEST(QwenOwnershipTest, ContextReplacementRetainsResourceSharingAndIndependentRevisions) {
    OwnershipFixture fixture(4);
    QwenRunner source;
    ASSERT_EQ(source.Load(fixture.path.string(), QwenBackend::kCommon), 0) << source.LastError();
    std::unique_ptr<QwenRunner> request;
    ASSERT_EQ(source.CreateRequest(&request), 0);
    ASSERT_EQ(request->Consume({2, 0}), 0);
    const auto revision = request->StateRevision();
    std::filesystem::remove(fixture.path);
    ASSERT_EQ(request->ReserveContext(16), 0) << request->LastError();
    EXPECT_TRUE(source.SharesModelResourcesWith(*request));
    EXPECT_EQ(request->StateRevision(), revision);
    EXPECT_EQ(request->TokensProcessed(), 2);
    EXPECT_EQ(source.MaxContext(), 4);
    std::vector<int64_t> generated;
    ASSERT_EQ(request->Generate(std::vector<int64_t>(5, 0), 1, {}, &generated), 0) << request->LastError();
    EXPECT_EQ(generated, (std::vector<int64_t>{2}));
    ASSERT_EQ(source.Generate({0}, 1, {}, &generated), 0) << source.LastError();
    EXPECT_EQ(generated, (std::vector<int64_t>{0}));
}
}  // namespace
