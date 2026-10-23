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

#include <memory>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/tensor.h"
#include "model/model_io.h"
#include "util/bf16.h"
#include "python_test_utils.h"

using feather::DataType;
using feather::Tensor;
using feather::model::ModelDesc;
using feather::model::ModelLoader;
using feather::model::ModelWriter;
using feather::model::NodeDesc;
using feather::model::ValueDesc;

TEST(model_io_test, LoadsPythonQwenExporterMetadataAndQuantization) {
    if (!feather::test::HasPythonModules("numpy, ml_dtypes, safetensors")) {
        GTEST_SKIP() << "Qwen exporter Python dependencies are unavailable";
    }
    const auto root = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto path = std::filesystem::temp_directory_path() / "qwen_python_fth_v4.fth";
    const std::string script =
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import export_qwen35_model as e; "
        "g=e.AtomicGraph(); g.input('value',[2],e.FP8E5M2,quantization_scale=0.25); "
        "g.output('value'); g.metadata['qwen.context.default']=4096; "
        "g.write_fth(sys.argv[2],'python_fixture')";
    const auto command = feather::test::PythonCommand() + " -B -c " + feather::test::ShellQuote(script) +
        " " + feather::test::ShellQuote((root / "tools").string()) + " " + feather::test::ShellQuote(path.string());
    ASSERT_EQ(std::system(command.c_str()), 0);
    ModelLoader loader;
    ASSERT_TRUE(loader.Load(path.string()));
    EXPECT_EQ(std::get<int64_t>(loader.model().metadata.at("qwen.context.default")), 4096);
    ASSERT_EQ(loader.model().graph.values.size(), 1U);
    const auto& tensor = loader.model().graph.values[0].tensor;
    EXPECT_EQ(tensor.dims, (std::vector<int64_t>{2}));
    EXPECT_EQ(tensor.data_type, DataType::FP8E5M2);
    EXPECT_FLOAT_EQ(tensor.quantization.scale, 0.25f);
    EXPECT_EQ(tensor.quantization.zero_point, 0);
    std::filesystem::remove(path);
}

namespace {

template <typename T>
void AppendPod(std::vector<char>* bytes, T value) {
    const auto* data = reinterpret_cast<const char*>(&value);
    bytes->insert(bytes->end(), data, data + sizeof(value));
}

void AppendString(std::vector<char>* bytes, const std::string& value) {
    AppendPod<uint64_t>(bytes, value.size());
    bytes->insert(bytes->end(), value.begin(), value.end());
}

// Independent fixture encoding: model, empty graph, then optional v4 metadata.
std::vector<char> EmptyGraphFixture() {
    std::vector<char> bytes;
    AppendString(&bytes, "fixture");
    AppendPod<int64_t>(&bytes, 9);
    AppendString(&bytes, "main");
    for (int i = 0; i < 4; ++i) AppendPod<uint64_t>(&bytes, 0);
    return bytes;
}

void WriteFixture(const std::string& path, uint32_t version, const std::vector<char>& metadata) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    const char magic[8] = {'F', 'T', 'H', 'M', 'O', 'D', 'L', '\0'};
    out.write(magic, sizeof(magic));
    const uint32_t reserved = 0;
    const uint64_t size = metadata.size();
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&reserved), sizeof(reserved));
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    out.write(metadata.data(), metadata.size());
}

std::vector<char> ReadFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

}  // namespace

TEST(model_io_test, MetadataRoundTripsEveryAttributeTypeWithoutChangingWeights) {
    const std::string path = "/tmp/feather_model_io_metadata.fth";
    ModelDesc model;
    model.name = "context_model";
    model.metadata = {{"qwen.context.version", int64_t{1}},
                      {"qwen.context.default", int64_t{2048}},
                      {"qwen.context.value.cache", std::vector<int64_t>{2, 0, 3, -1}},
                      {"description", std::string("context\0model", 13)},
                      {"scale", 0.25f},
                      {"scales", std::vector<float>{0.5f, 2.0f}},
                      {"empty", std::vector<int64_t>{}}};
    ValueDesc value;
    value.tensor.name = "weight";
    value.tensor.dims = {2};
    value.tensor.data_type = DataType::FP32;
    value.constant = true;
    model.graph.values = {value};
    auto weight = std::make_shared<Tensor>();
    weight->Assign<float>({1.25f, -3.0f}, {2});

    ASSERT_TRUE(ModelWriter().Save(path, model, {{"weight", weight}}));
    ModelLoader loader;
    ASSERT_TRUE(loader.Load(path));
    EXPECT_EQ(loader.model().metadata, model.metadata);
    auto loaded_weight = loader.CreateWeightTensor("weight");
    ASSERT_NE(loaded_weight, nullptr);
    EXPECT_FLOAT_EQ(loaded_weight->data<float>()[0], 1.25f);
    EXPECT_FLOAT_EQ(loaded_weight->data<float>()[1], -3.0f);
}

TEST(model_io_test, MetadataSerializationIsIndependentOfMapInsertionOrder) {
    ModelDesc first;
    first.metadata.emplace("z", int64_t{3});
    first.metadata.emplace("a", int64_t{1});
    first.metadata.emplace("m", int64_t{2});
    ModelDesc second;
    second.metadata.emplace("m", int64_t{2});
    second.metadata.emplace("a", int64_t{1});
    second.metadata.emplace("z", int64_t{3});
    const std::string first_path = "/tmp/feather_model_io_metadata_order1.fth";
    const std::string second_path = "/tmp/feather_model_io_metadata_order2.fth";
    ASSERT_TRUE(ModelWriter().Save(first_path, first, {}));
    ASSERT_TRUE(ModelWriter().Save(second_path, second, {}));
    EXPECT_EQ(ReadFile(first_path), ReadFile(second_path));
}

TEST(model_io_test, LoadsVersionsOneThroughThreeWithTheirOriginalQuantizationEncoding) {
    for (uint32_t version = 1; version <= 3; ++version) {
        SCOPED_TRACE(version);
        std::vector<char> bytes;
        AppendString(&bytes, "legacy");
        AppendPod<int64_t>(&bytes, 7);
        AppendString(&bytes, "main");
        AppendPod<uint64_t>(&bytes, 0);  // Graph inputs.
        AppendPod<uint64_t>(&bytes, 0);  // Graph outputs.
        AppendPod<uint64_t>(&bytes, 1);  // One tensor value.
        AppendString(&bytes, "activation");
        AppendPod<uint64_t>(&bytes, 1);  // One dimension.
        AppendPod<int64_t>(&bytes, 4);
        AppendPod<int32_t>(&bytes, static_cast<int32_t>(DataType::FP32));
        AppendPod<int32_t>(&bytes, static_cast<int32_t>(feather::DataLayout::ND));
        if (version >= 2) {
            AppendPod<uint8_t>(&bytes, 1);
            AppendPod<float>(&bytes, 0.5f);
            AppendPod<int32_t>(&bytes, 0);  // Per tensor.
            AppendPod<int64_t>(&bytes, -1);
            AppendPod<int64_t>(&bytes, 0);
        }
        if (version >= 3) {
            AppendPod<int32_t>(&bytes, 3);  // Zero point.
            AppendPod<uint64_t>(&bytes, 2);
            AppendPod<float>(&bytes, 0.25f);
            AppendPod<float>(&bytes, 0.75f);
            AppendPod<uint64_t>(&bytes, 2);
            AppendPod<int32_t>(&bytes, 3);
            AppendPod<int32_t>(&bytes, 4);
        }
        AppendPod<uint8_t>(&bytes, 0);  // Not a constant.
        AppendString(&bytes, "");
        AppendString(&bytes, "");
        AppendPod<uint64_t>(&bytes, 0);
        AppendPod<uint64_t>(&bytes, 0);
        AppendString(&bytes, "");
        AppendPod<uint64_t>(&bytes, 0);  // No nodes.
        const std::string path = "/tmp/feather_model_io_legacy" + std::to_string(version) + ".fth";
        WriteFixture(path, version, bytes);
        ModelLoader loader;
        ASSERT_TRUE(loader.Load(path));
        EXPECT_TRUE(loader.model().metadata.empty());
        EXPECT_EQ(loader.model().version, 7);
        ASSERT_EQ(loader.model().graph.values.size(), 1U);
        const auto& tensor = loader.model().graph.values[0].tensor;
        EXPECT_EQ(tensor.dims, std::vector<int64_t>({4}));
        EXPECT_EQ(tensor.quantization.enabled, version >= 2);
        if (version == 2) {
            EXPECT_EQ(tensor.quantization.scales, std::vector<float>({0.5f}));
            EXPECT_EQ(tensor.quantization.zero_points, std::vector<int32_t>({0}));
        } else if (version == 3) {
            EXPECT_EQ(tensor.quantization.scales, std::vector<float>({0.25f, 0.75f}));
            EXPECT_EQ(tensor.quantization.zero_points, std::vector<int32_t>({3, 4}));
        }
    }
}

TEST(model_io_test, LoadsVersionFourMetadataAfterGraph) {
    auto bytes = EmptyGraphFixture();
    AppendPod<uint64_t>(&bytes, 1);
    AppendString(&bytes, "qwen.context.default");
    AppendPod<uint32_t>(&bytes, 0);
    AppendPod<int64_t>(&bytes, 2048);
    const std::string path = "/tmp/feather_model_io_v4_fixture.fth";
    WriteFixture(path, 4, bytes);
    ModelLoader loader;
    ASSERT_TRUE(loader.Load(path));
    EXPECT_EQ(loader.model().metadata.at("qwen.context.default"),
              feather::model::AttributeValue(int64_t{2048}));
}

TEST(model_io_test, RejectsMalformedVersionFourMetadata) {
    const auto base = EmptyGraphFixture();
    std::vector<std::vector<char>> invalid;
    invalid.push_back(base);  // Missing metadata count.
    auto bytes = base;
    AppendPod<uint64_t>(&bytes, 2);
    for (int i = 0; i < 2; ++i) {
        AppendString(&bytes, "duplicate");
        AppendPod<uint32_t>(&bytes, 0);
        AppendPod<int64_t>(&bytes, i);
    }
    invalid.push_back(bytes);
    bytes = base;
    AppendPod<uint64_t>(&bytes, 1);
    AppendString(&bytes, "bad_tag");
    AppendPod<uint32_t>(&bytes, 99);
    invalid.push_back(bytes);
    bytes = base;
    AppendPod<uint64_t>(&bytes, 1);
    AppendString(&bytes, "truncated_vector");
    AppendPod<uint32_t>(&bytes, 3);
    AppendPod<uint64_t>(&bytes, 2);
    AppendPod<int64_t>(&bytes, 1);
    invalid.push_back(bytes);
    bytes = base;
    AppendPod<uint64_t>(&bytes, UINT64_MAX);
    invalid.push_back(bytes);
    bytes = base;
    AppendPod<uint64_t>(&bytes, 0);
    bytes.push_back('!');
    invalid.push_back(bytes);
    for (size_t i = 0; i < invalid.size(); ++i) {
        SCOPED_TRACE(i);
        const std::string path = "/tmp/feather_model_io_invalid_metadata.fth";
        WriteFixture(path, 4, invalid[i]);
        EXPECT_FALSE(ModelLoader().Load(path));
    }
}

TEST(model_io_test, SaveAndLoadBinaryModel) {
    const std::string path = "/tmp/feather_model_io_test.fth";

    auto weight = std::make_shared<Tensor>();
    weight->Assign<float>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});

    ModelDesc model;
    model.name = "tiny_fc";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = {"output"};

    ValueDesc input;
    input.tensor.name = "input";
    input.tensor.dims = {1, 2};
    input.tensor.data_type = DataType::FP32;

    ValueDesc weight_value;
    weight_value.tensor.name = "linear.weight";
    weight_value.tensor.dims = {2, 2};
    weight_value.tensor.data_type = DataType::FP32;
    weight_value.constant = true;

    ValueDesc output;
    output.tensor.name = "output";
    output.tensor.dims = {1, 2};
    output.tensor.data_type = DataType::FP32;

    NodeDesc node;
    node.name = "linear";
    node.op_type = "Gemm";
    node.inputs = {"input", "linear.weight"};
    node.outputs = {"output"};

    model.graph.values = {input, weight_value, output};
    model.graph.nodes = {node};

    ModelWriter writer;
    ASSERT_TRUE(writer.Save(path, model, {{"linear.weight", weight}}));

    ModelLoader loader;
    ASSERT_TRUE(loader.Load(path));
    EXPECT_EQ(loader.model().name, "tiny_fc");
    ASSERT_EQ(loader.model().graph.nodes.size(), 1);
    EXPECT_EQ(loader.model().graph.nodes[0].op_type, "Gemm");

    auto loaded_weight = loader.CreateWeightTensor("linear.weight");
    ASSERT_NE(loaded_weight, nullptr);
    EXPECT_EQ(loaded_weight->dims().data(), std::vector<int64_t>({2, 2}));
    EXPECT_EQ(loaded_weight->data_type(), DataType::FP32);
    EXPECT_EQ(loaded_weight->data<float>()[3], 4.0f);
}

TEST(model_io_test, SaveAndLoadBFloat16WeightPayload) {
    const std::string path = "/tmp/feather_model_io_bfloat16_test.fth";

    auto weight = std::make_shared<Tensor>();
    weight->Assign<feather::BFloat16>(
        {{feather::FloatToBFloat16(1.0f)}, {feather::FloatToBFloat16(-2.5f)}}, {2});

    ModelDesc model;
    model.name = "bf16_weight";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = {"output"};

    ValueDesc input;
    input.tensor.name = "input";
    input.tensor.dims = {1, 2};
    input.tensor.data_type = DataType::BF16;

    ValueDesc weight_value;
    weight_value.tensor.name = "weight";
    weight_value.tensor.dims = {2};
    weight_value.tensor.data_type = DataType::BF16;
    weight_value.constant = true;

    ValueDesc output;
    output.tensor.name = "output";
    output.tensor.dims = {1, 2};
    output.tensor.data_type = DataType::BF16;

    model.graph.values = {input, weight_value, output};

    ModelWriter writer;
    ASSERT_TRUE(writer.Save(path, model, {{"weight", weight}}));

    ModelLoader loader;
    ASSERT_TRUE(loader.Load(path));
    auto loaded_weight = loader.CreateWeightTensor("weight");
    ASSERT_NE(loaded_weight, nullptr);
    EXPECT_EQ(loaded_weight->data_type(), DataType::BF16);
    EXPECT_EQ(loaded_weight->memory_size(), 2U * sizeof(feather::BFloat16));
    EXPECT_FLOAT_EQ(feather::BFloat16ToFloat(loaded_weight->data<feather::BFloat16>()[0].bits), 1.0f);
    EXPECT_FLOAT_EQ(feather::BFloat16ToFloat(loaded_weight->data<feather::BFloat16>()[1].bits), -2.5f);
}
