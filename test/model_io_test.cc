#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/tensor.h"
#include "model/model_io.h"
#include "util/bf16.h"

using feather::DataType;
using feather::Tensor;
using feather::model::ModelDesc;
using feather::model::ModelLoader;
using feather::model::ModelWriter;
using feather::model::NodeDesc;
using feather::model::ValueDesc;

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
