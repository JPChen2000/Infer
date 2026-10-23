// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include <filesystem>
#include <limits>

#include "demo/qwen_sequence.h"

namespace {
using namespace feather;
using namespace feather::model;

ValueDesc SequenceValue(const std::string& name, std::vector<int64_t> dims, DataType dtype,
                        bool constant = false) {
    ValueDesc value;
    value.tensor.name = name;
    value.tensor.dims = std::move(dims);
    value.tensor.data_type = dtype;
    value.constant = constant;
    return value;
}

ModelDesc SequenceFixture() {
    ModelDesc model;
    model.name = "sequence_bindings";
    model.graph.inputs = {"token_ids", "hidden"};
    model.graph.outputs = {"last"};
    model.graph.values = {
        SequenceValue("token_ids", {1, 1}, DataType::INT64),
        SequenceValue("hidden", {1, 1, 4, 4}, DataType::FP32),
        SequenceValue("last", {1, 1, 4, 4}, DataType::FP32),
        SequenceValue("joined", {1, 4, 4}, DataType::BF16),
        SequenceValue("qwen_chunk_offsets", {1}, DataType::FP32, true),
        SequenceValue("shape", {4}, DataType::INT64, true),
    };
    NodeDesc node;
    node.name = "last";
    node.op_type = "Slice";
    node.inputs = {"hidden"};
    node.outputs = {"last"};
    node.attributes["start"] = int64_t{0};
    node.attributes["end"] = int64_t{1};
    node.attributes["axis"] = int64_t{1};
    node.attributes["shape"] = std::vector<int64_t>{1, 1, 4, 4};
    model.graph.nodes = {node};
    model.metadata = {
        {"qwen.execution.version", int64_t{2}},
        {"qwen.execution.sequence", int64_t{1}},
        {"qwen.sequence.version", int64_t{1}},
        {"qwen.sequence.default", int64_t{1}},
        {"qwen.sequence.maximum", int64_t{8}},
        {"qwen.sequence.value.token_ids", std::vector<int64_t>{1, 0}},
        {"qwen.sequence.value.hidden", std::vector<int64_t>{1, 0}},
        {"qwen.sequence.value.joined", std::vector<int64_t>{2, 3}},
        {"qwen.sequence.value.qwen_chunk_offsets", std::vector<int64_t>{0, 0}},
        {"qwen.sequence.arange.qwen_chunk_offsets", int64_t{1}},
        {"qwen.sequence.constant.shape", std::vector<int64_t>{1, 0}},
        {"qwen.sequence.attribute.last.start", std::vector<int64_t>{0, -1}},
        {"qwen.sequence.attribute.last.end", std::vector<int64_t>{0, 0}},
        {"qwen.sequence.attribute.last.shape", std::vector<int64_t>{1, 0}},
    };
    return model;
}

bool LoadSequenceFixture(ModelLoader* loader, const ModelDesc& model) {
    std::unordered_map<std::string, std::shared_ptr<Tensor>> weights;
    weights["shape"] = std::make_shared<Tensor>();
    weights["shape"]->Assign<int64_t>({1, 1, 4, 4}, {4});
    weights["qwen_chunk_offsets"] = std::make_shared<Tensor>();
    weights["qwen_chunk_offsets"]->Assign<float>({0}, {1});
    const auto path = std::filesystem::temp_directory_path() / "qwen_sequence_bindings.fth";
    ModelWriter writer;
    return writer.Save(path.string(), model, weights) && loader->Load(path.string());
}

const TensorDesc& SequenceDesc(const ModelDesc& model, const std::string& name) {
    for (const auto& value : model.graph.values) if (value.tensor.name == name) return value.tensor;
    throw std::runtime_error("missing sequence fixture value");
}
}  // namespace

TEST(qwen_sequence_test, SpecializesEveryLengthAndKeepsContextOverridesAndStaticDimensions) {
    ModelLoader loader;
    ASSERT_TRUE(LoadSequenceFixture(&loader, SequenceFixture()));
    for (int64_t sequence : {int64_t{1}, int64_t{2}, int64_t{3}, int64_t{7}, int64_t{8}}) {
        SCOPED_TRACE(sequence);
        ModelDesc effective = loader.model();
        // Simulate prior context specialization, including a shared shape payload.
        for (auto& value : effective.graph.values) {
            if (value.tensor.name == "hidden" || value.tensor.name == "last") value.tensor.dims[2] = 16;
        }
        effective.graph.nodes[0].attributes["shape"] = std::vector<int64_t>{1, 1, 16, 4};
        std::unordered_map<std::string, std::shared_ptr<Tensor>> overrides;
        auto sentinel = std::make_shared<Tensor>();
        overrides["context_sentinel"] = sentinel;
        overrides["shape"] = std::make_shared<Tensor>();
        overrides["shape"]->Assign<int64_t>({1, 1, 16, 4}, {4});
        const auto context_shape = overrides["shape"];
        std::string error;
        ASSERT_TRUE(demo::SpecializeQwenSequence(loader, sequence, &effective, &overrides, &error)) << error;
        EXPECT_EQ(SequenceDesc(effective, "token_ids").dims, (std::vector<int64_t>{1, sequence}));
        EXPECT_EQ(SequenceDesc(effective, "hidden").dims, (std::vector<int64_t>{1, sequence, 16, 4}));
        EXPECT_EQ(SequenceDesc(effective, "last").dims, (std::vector<int64_t>{1, 1, 16, 4}));
        EXPECT_EQ(SequenceDesc(effective, "joined").dims, (std::vector<int64_t>{1, 4, sequence + 3}));
        EXPECT_EQ(SequenceDesc(effective, "qwen_chunk_offsets").dims, (std::vector<int64_t>{sequence}));
        EXPECT_EQ(overrides.at("context_sentinel"), sentinel);
        EXPECT_EQ(overrides.at("shape")->data<int64_t>()[1], sequence);
        EXPECT_EQ(overrides.at("shape")->data<int64_t>()[2], 16);
        EXPECT_EQ(overrides.at("shape")->data<int64_t>()[3], 4);
        EXPECT_EQ(context_shape->data<int64_t>()[1], 1);
        EXPECT_EQ(overrides.at("qwen_chunk_offsets")->dims().data(), (std::vector<int64_t>{sequence}));
        ASSERT_EQ(overrides.at("qwen_chunk_offsets")->memory_size(), sequence * sizeof(float));
        for (int64_t i = 0; i < sequence; ++i)
            EXPECT_EQ(overrides.at("qwen_chunk_offsets")->data<float>()[i], static_cast<float>(i));
        EXPECT_EQ(std::get<int64_t>(effective.graph.nodes[0].attributes.at("start")), sequence - 1);
        EXPECT_EQ(std::get<int64_t>(effective.graph.nodes[0].attributes.at("end")), sequence);
        EXPECT_EQ(std::get<std::vector<int64_t>>(effective.graph.nodes[0].attributes.at("shape")),
                  (std::vector<int64_t>{1, sequence, 16, 4}));
        EXPECT_EQ(std::get<int64_t>(effective.metadata.at("qwen.execution.sequence")), sequence);
        EXPECT_EQ(SequenceDesc(loader.model(), "token_ids").dims, (std::vector<int64_t>{1, 1}));
        EXPECT_EQ(loader.CreateWeightTensor("shape")->data<int64_t>()[1], 1);
    }
}

TEST(qwen_sequence_test, DefaultAndLegacyFixedGraphsRemainUsable) {
    for (bool legacy : {false, true}) {
        auto model = SequenceFixture();
        if (legacy) {
            model.metadata.clear();
            model.metadata["qwen.execution.version"] = int64_t{1};
        }
        ModelLoader loader;
        ASSERT_TRUE(LoadSequenceFixture(&loader, model));
        ModelDesc effective = loader.model();
        std::unordered_map<std::string, std::shared_ptr<Tensor>> overrides;
        std::string error;
        EXPECT_TRUE(demo::SpecializeQwenSequence(loader, 0, &effective, &overrides, &error)) << error;
        if (legacy) {
            EXPECT_TRUE(overrides.empty());
            EXPECT_TRUE(demo::SpecializeQwenSequence(loader, 1, &effective, &overrides, &error));
            EXPECT_FALSE(demo::SpecializeQwenSequence(loader, 2, &effective, &overrides, &error));
        }
    }
}

TEST(qwen_sequence_test, InvalidMetadataAndRangesLeaveBothOutputsUnchanged) {
    std::vector<ModelDesc> invalid;
    auto model = SequenceFixture();
    model.metadata["qwen.sequence.version"] = int64_t{2}; invalid.push_back(model);
    model = SequenceFixture(); model.metadata.erase("qwen.sequence.maximum"); invalid.push_back(model);
    model = SequenceFixture(); model.metadata["qwen.sequence.default"] = std::string("1"); invalid.push_back(model);
    model = SequenceFixture(); model.metadata["qwen.sequence.maximum"] = int64_t{0}; invalid.push_back(model);
    model = SequenceFixture(); model.metadata.erase("qwen.sequence.value.token_ids"); invalid.push_back(model);
    model = SequenceFixture(); model.metadata["qwen.sequence.value.token_ids"] = std::vector<int64_t>{0, 0}; invalid.push_back(model);
    model = SequenceFixture(); model.metadata["qwen.sequence.value.hidden"] = std::vector<int64_t>{1}; invalid.push_back(model);
    model = SequenceFixture(); model.metadata["qwen.sequence.value.hidden"] = std::vector<int64_t>{1, 0, 1, 0}; invalid.push_back(model);
    model = SequenceFixture(); model.metadata["qwen.sequence.value.hidden"] = std::vector<int64_t>{9, 0}; invalid.push_back(model);
    model = SequenceFixture(); model.metadata["qwen.sequence.value.hidden"] = std::vector<int64_t>{1, 1}; invalid.push_back(model);
    model = SequenceFixture(); model.metadata["qwen.sequence.value.hidden"] = std::vector<int64_t>{1, std::numeric_limits<int64_t>::max()}; invalid.push_back(model);
    model = SequenceFixture(); model.metadata["qwen.sequence.value.missing"] = std::vector<int64_t>{0, 0}; invalid.push_back(model);
    model = SequenceFixture(); model.metadata["qwen.sequence.value.shape"] = std::vector<int64_t>{0, 3}; invalid.push_back(model);
    model = SequenceFixture(); model.metadata["qwen.sequence.constant.hidden"] = std::vector<int64_t>{1, 0}; invalid.push_back(model);
    model = SequenceFixture(); model.metadata["qwen.sequence.constant.shape"] = std::vector<int64_t>{1, 1}; invalid.push_back(model);
    model = SequenceFixture(); model.metadata["qwen.sequence.attribute.last.missing"] = std::vector<int64_t>{0, 0}; invalid.push_back(model);
    model = SequenceFixture(); model.metadata["qwen.sequence.attribute.last.start"] = std::vector<int64_t>{1, 0}; invalid.push_back(model);
    model = SequenceFixture(); model.graph.nodes[0].attributes["start"] = 0.0f; invalid.push_back(model);
    model = SequenceFixture(); model.metadata["qwen.sequence.arange.qwen_chunk_offsets"] = int64_t{9}; invalid.push_back(model);
    model = SequenceFixture(); model.metadata.erase("qwen.sequence.value.qwen_chunk_offsets"); invalid.push_back(model);
    model = SequenceFixture(); model.metadata.erase("qwen.sequence.arange.qwen_chunk_offsets"); invalid.push_back(model);
    model = SequenceFixture(); model.metadata.erase("qwen.sequence.arange.qwen_chunk_offsets");
    model.metadata.erase("qwen.sequence.value.qwen_chunk_offsets"); invalid.push_back(model);
    model = SequenceFixture(); model.metadata.clear();
    model.metadata["qwen.execution.version"] = int64_t{2}; invalid.push_back(model);
    model = SequenceFixture(); model.metadata["qwen.sequence.unknown"] = int64_t{1}; invalid.push_back(model);
    for (const auto& item : invalid) {
        ModelLoader loader;
        ASSERT_TRUE(LoadSequenceFixture(&loader, item));
        ModelDesc effective = loader.model();
        effective.name = "unchanged";
        std::unordered_map<std::string, std::shared_ptr<Tensor>> overrides;
        auto sentinel = std::make_shared<Tensor>();
        overrides["sentinel"] = sentinel;
        std::string error;
        EXPECT_FALSE(demo::SpecializeQwenSequence(loader, 8, &effective, &overrides, &error));
        EXPECT_EQ(effective.name, "unchanged");
        EXPECT_EQ(SequenceDesc(effective, "token_ids").dims, (std::vector<int64_t>{1, 1}));
        ASSERT_EQ(overrides.size(), 1U);
        EXPECT_EQ(overrides.at("sentinel"), sentinel);
        EXPECT_FALSE(error.empty());
    }
    ModelLoader loader;
    ASSERT_TRUE(LoadSequenceFixture(&loader, SequenceFixture()));
    for (int64_t sequence : {int64_t{-1}, int64_t{9}}) {
        auto effective = loader.model();
        std::unordered_map<std::string, std::shared_ptr<Tensor>> overrides;
        std::string error;
        EXPECT_FALSE(demo::SpecializeQwenSequence(loader, sequence, &effective, &overrides, &error));
    }
}
