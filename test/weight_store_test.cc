#include <gtest/gtest.h>

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "model/weight_store.h"

using feather::DataType;
using feather::model::TensorDesc;
using feather::model::WeightLocation;
using feather::model::WeightStore;

TEST(weight_store_test, CreateTensorViewFromShard) {
    const std::string path = "/tmp/feather_weight_store_test.bin";
    const std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f};

    {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
    }

    TensorDesc desc;
    desc.name = "linear.weight";
    desc.dims = {2, 2};
    desc.data_type = DataType::FP32;

    WeightLocation location;
    location.tensor_name = desc.name;
    location.shard_path = path;
    location.offset = 0;
    location.byte_size = values.size() * sizeof(float);

    WeightStore store;
    auto tensor = store.CreateTensorView(desc, location);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->dims().data(), desc.dims);
    EXPECT_EQ(tensor->data_type(), DataType::FP32);
    EXPECT_EQ(tensor->data<float>()[2], 3.0f);
}
