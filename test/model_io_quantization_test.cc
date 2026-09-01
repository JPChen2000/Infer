#include <cstdio>
#include <memory>
#include <string>
#include <unordered_map>

#include <gtest/gtest.h>

#include "core/tensor.h"
#include "model/model_io.h"

namespace {

TEST(model_io_quantization_test, RoundTripsVectorScalesAndZeroPoints) {
    const std::string path = "/tmp/feather_int8_quantization_metadata.fth";
    auto weight = std::make_shared<feather::Tensor>();
    weight->Assign<int8_t>({-4, -1, 2, 7}, {2, 2});

    feather::model::ModelDesc model;
    model.name = "int8_metadata";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = {"weight"};

    feather::model::ValueDesc input;
    input.tensor.name = "input";
    input.tensor.dims = {1, 2};
    input.tensor.data_type = feather::DataType::INT8;
    input.tensor.quantization.enabled = true;
    input.tensor.quantization.scale = 0.25f;
    input.tensor.quantization.zero_point = -3;

    feather::model::ValueDesc value;
    value.tensor.name = "weight";
    value.tensor.dims = {2, 2};
    value.tensor.data_type = feather::DataType::INT8;
    value.tensor.quantization.enabled = true;
    value.tensor.quantization.granularity = feather::QuantizationGranularity::kPerChannel;
    value.tensor.quantization.axis = 0;
    value.tensor.quantization.scales = {0.125f, 0.5f};
    value.tensor.quantization.zero_points = {-2, 3};
    value.constant = true;

    model.graph.values = {input, value};
    ASSERT_TRUE((feather::model::ModelWriter{}).Save(path, model, {{"weight", weight}}));

    feather::model::ModelLoader loader;
    ASSERT_TRUE(loader.Load(path));
    const auto& loaded = loader.model().graph.values[1].tensor;
    EXPECT_EQ(loaded.data_type, feather::DataType::INT8);
    EXPECT_TRUE(loaded.quantization.enabled);
    EXPECT_EQ(loaded.quantization.granularity, feather::QuantizationGranularity::kPerChannel);
    EXPECT_EQ(loaded.quantization.axis, 0);
    EXPECT_FLOAT_EQ(loaded.quantization.scales[0], 0.125f);
    EXPECT_FLOAT_EQ(loaded.quantization.scales[1], 0.5f);
    EXPECT_EQ(loaded.quantization.zero_points, (std::vector<int32_t>{-2, 3}));

    auto loaded_weight = loader.CreateWeightTensor("weight");
    ASSERT_NE(loaded_weight, nullptr);
    EXPECT_EQ(loaded_weight->data_type(), feather::DataType::INT8);
    EXPECT_EQ(loaded_weight->data<int8_t>()[3], 7);
    std::remove(path.c_str());
}

}  // namespace
