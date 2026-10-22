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
