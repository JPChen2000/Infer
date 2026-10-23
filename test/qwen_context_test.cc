// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include <filesystem>
#include <limits>

#include "demo/qwen_context.h"

namespace {
using namespace feather;
using namespace feather::model;

ValueDesc ContextValue(const std::string& name, std::vector<int64_t> dims, DataType dtype, bool constant = false) {
    ValueDesc value;
    value.tensor.name = name;
    value.tensor.dims = std::move(dims);
    value.tensor.data_type = dtype;
    value.constant = constant;
    return value;
}

ModelDesc ContextFixture() {
    ModelDesc model;
    model.name = "context_bindings";
    model.graph.inputs = {"attention_mask", "input", "cache"};
    model.graph.outputs = {"output"};
    model.graph.values = {
        ContextValue("attention_mask", {1, 1, 1, 4}, DataType::BF16),
        ContextValue("input", {1, 4, 4}, DataType::FP32),
        ContextValue("cache", {1, 3, 4}, DataType::BF16),
        ContextValue("output", {1, 4, 4}, DataType::FP32),
        ContextValue("shape", {3}, DataType::INT64, true),
        ContextValue("unrelated", {2}, DataType::INT64, true),
    };
    NodeDesc node;
    node.name = "reshape";
    node.op_type = "Reshape";
    node.inputs = {"input", "shape"};
    node.outputs = {"output"};
    node.attributes["shape"] = std::vector<int64_t>{1, 4, 4};
    model.graph.nodes = {node};
    model.metadata = {
        {"qwen.context.version", int64_t{1}},
        {"qwen.context.default", int64_t{4}},
        {"qwen.context.maximum", int64_t{16}},
        {"qwen.context.value.attention_mask", std::vector<int64_t>{3, 0}},
        {"qwen.context.value.input", std::vector<int64_t>{1, 0}},
        {"qwen.context.value.cache", std::vector<int64_t>{1, -1}},
        {"qwen.context.value.output", std::vector<int64_t>{1, 0}},
        {"qwen.context.constant.shape", std::vector<int64_t>{1, 0}},
        {"qwen.context.attribute.reshape.shape", std::vector<int64_t>{1, 0}},
    };
    return model;
}

bool LoadFixture(ModelLoader* loader, const ModelDesc& model) {
    std::unordered_map<std::string, std::shared_ptr<Tensor>> weights;
    weights["shape"] = std::make_shared<Tensor>();
    weights["shape"]->Assign<int64_t>({1, 4, 4}, {3});
    weights["unrelated"] = std::make_shared<Tensor>();
    weights["unrelated"]->Assign<int64_t>({4, 3}, {2});
    const auto path = std::filesystem::temp_directory_path() / "qwen_context_bindings.fth";
    ModelWriter writer;
    return writer.Save(path.string(), model, weights) && loader->Load(path.string());
}

const TensorDesc& Desc(const ModelDesc& model, const std::string& name) {
    for (const auto& value : model.graph.values) if (value.tensor.name == name) return value.tensor;
    throw std::runtime_error("missing fixture value");
}
}  // namespace

TEST(qwen_context_test, SpecializesAllExplicitTargetsWithoutChangingEqualStaticDimensions) {
    ModelLoader loader;
    ASSERT_TRUE(LoadFixture(&loader, ContextFixture()));
    for (int64_t context : {int64_t{2}, int64_t{8}, int64_t{16}}) {
        SCOPED_TRACE(context);
        ModelDesc effective;
        std::unordered_map<std::string, std::shared_ptr<Tensor>> overrides;
        int64_t limit = 0;
        std::string error;
        ASSERT_TRUE(feather::demo::SpecializeQwenContext(loader, context, &effective, &overrides, &limit, &error)) << error;
        EXPECT_EQ(limit, 16);
        EXPECT_EQ(Desc(effective, "attention_mask").dims, (std::vector<int64_t>{1, 1, 1, context}));
        EXPECT_EQ(Desc(effective, "input").dims, (std::vector<int64_t>{1, context, 4}));
        EXPECT_EQ(Desc(effective, "output").dims, (std::vector<int64_t>{1, context, 4}));
        EXPECT_EQ(Desc(effective, "cache").dims, (std::vector<int64_t>{1, context - 1, 4}));
        EXPECT_EQ(std::get<std::vector<int64_t>>(effective.graph.nodes[0].attributes.at("shape")),
                  (std::vector<int64_t>{1, context, 4}));
        ASSERT_EQ(overrides.size(), 1U);
        ASSERT_EQ(overrides.count("shape"), 1U);
        EXPECT_EQ(overrides.at("shape")->data<int64_t>()[1], context);
        EXPECT_EQ(overrides.at("shape")->data<int64_t>()[2], 4);
        EXPECT_EQ(loader.CreateWeightTensor("shape")->data<int64_t>()[1], 4);
        EXPECT_EQ(loader.CreateWeightTensor("unrelated")->data<int64_t>()[0], 4);
        EXPECT_EQ(Desc(loader.model(), "input").dims, (std::vector<int64_t>{1, 4, 4}));
    }
}

TEST(qwen_context_test, DefaultsToExportMetadataAndPreservesLegacyFixedModels) {
    for (bool legacy : {false, true}) {
        auto model = ContextFixture();
        if (legacy) model.metadata.clear();
        ModelLoader loader;
        ASSERT_TRUE(LoadFixture(&loader, model));
        ModelDesc effective;
        std::unordered_map<std::string, std::shared_ptr<Tensor>> overrides;
        int64_t limit = 0;
        std::string error;
        ASSERT_TRUE(feather::demo::SpecializeQwenContext(loader, 0, &effective, &overrides, &limit, &error)) << error;
        EXPECT_EQ(Desc(effective, "attention_mask").dims.back(), 4);
        EXPECT_EQ(limit, legacy ? 4 : 16);
        if (legacy) {
            EXPECT_TRUE(overrides.empty());
            EXPECT_FALSE(feather::demo::SpecializeQwenContext(loader, 8, &effective, &overrides, &limit, &error));
            EXPECT_FALSE(error.empty());
        }
    }
}

TEST(qwen_context_test, RejectsInvalidCapacityAndBindingsBeforeCommittingOutputs) {
    std::vector<ModelDesc> invalid;
    auto model = ContextFixture();
    model.metadata["qwen.context.version"] = int64_t{99}; invalid.push_back(model);
    model = ContextFixture(); model.metadata["qwen.context.default"] = std::string("4"); invalid.push_back(model);
    model = ContextFixture(); model.metadata["qwen.context.value.input"] = std::vector<int64_t>{1}; invalid.push_back(model);
    model = ContextFixture(); model.metadata["qwen.context.value.input"] = std::vector<int64_t>{1, 0, 1, 0}; invalid.push_back(model);
    model = ContextFixture(); model.metadata["qwen.context.value.input"] = std::vector<int64_t>{9, 0}; invalid.push_back(model);
    model = ContextFixture(); model.metadata["qwen.context.value.input"] = std::vector<int64_t>{1, 1}; invalid.push_back(model);
    model = ContextFixture(); model.metadata["qwen.context.value.missing"] = std::vector<int64_t>{1, 0}; invalid.push_back(model);
    model = ContextFixture(); model.metadata["qwen.context.constant.unrelated"] = std::vector<int64_t>{0, 1}; invalid.push_back(model);
    model = ContextFixture(); model.metadata["qwen.context.attribute.reshape.missing"] = std::vector<int64_t>{0, 0}; invalid.push_back(model);
    model = ContextFixture(); model.metadata["qwen.context.value.input"] = std::vector<int64_t>{1, std::numeric_limits<int64_t>::max()}; invalid.push_back(model);
    model = ContextFixture(); model.metadata.erase("qwen.context.version"); invalid.push_back(model);
    model = ContextFixture(); model.metadata.erase("qwen.context.value.attention_mask"); invalid.push_back(model);
    model = ContextFixture(); model.metadata.erase("qwen.context.default"); invalid.push_back(model);
    model = ContextFixture(); model.metadata.erase("qwen.context.maximum"); invalid.push_back(model);
    model = ContextFixture(); model.metadata["qwen.context.maximum"] = int64_t{3}; invalid.push_back(model);
    model = ContextFixture(); model.metadata["qwen.context.unknown"] = int64_t{1}; invalid.push_back(model);
    model = ContextFixture(); model.metadata["qwen.context.value.input"] = 1.0f; invalid.push_back(model);
    model = ContextFixture(); model.metadata["qwen.context.value.input"] = std::vector<int64_t>{}; invalid.push_back(model);
    model = ContextFixture(); model.metadata["qwen.context.value.input"] = std::vector<int64_t>{-1, 0}; invalid.push_back(model);
    model = ContextFixture(); model.metadata["qwen.context.value.shape"] = std::vector<int64_t>{0, -1}; invalid.push_back(model);
    model = ContextFixture(); model.metadata["qwen.context.constant.input"] = std::vector<int64_t>{1, 0}; invalid.push_back(model);
    model = ContextFixture(); model.graph.nodes[0].attributes["shape"] = int64_t{4}; invalid.push_back(model);
    model = ContextFixture(); model.metadata["qwen.context.value.attention_mask"] = std::vector<int64_t>{2, -3}; invalid.push_back(model);
    for (const auto& item : invalid) {
        ModelLoader loader;
        ASSERT_TRUE(LoadFixture(&loader, item));
        ModelDesc effective; effective.name = "unchanged";
        std::unordered_map<std::string, std::shared_ptr<Tensor>> overrides;
        auto sentinel = std::make_shared<Tensor>();
        overrides.emplace("sentinel", sentinel);
        int64_t limit = 42;
        std::string error;
        EXPECT_FALSE(feather::demo::SpecializeQwenContext(loader, 8, &effective, &overrides, &limit, &error));
        EXPECT_EQ(effective.name, "unchanged");
        EXPECT_EQ(limit, 42);
        ASSERT_EQ(overrides.size(), 1U);
        EXPECT_EQ(overrides.at("sentinel"), sentinel);
        EXPECT_FALSE(error.empty());
    }
    ModelLoader loader;
    ASSERT_TRUE(LoadFixture(&loader, ContextFixture()));
    for (int64_t capacity : {int64_t{-1}, int64_t{1}, int64_t{17}}) {
        ModelDesc effective;
        std::unordered_map<std::string, std::shared_ptr<Tensor>> overrides;
        int64_t limit = 0;
        std::string error;
        EXPECT_FALSE(feather::demo::SpecializeQwenContext(loader, capacity, &effective, &overrides, &limit, &error));
    }
}
