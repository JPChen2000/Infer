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

#ifndef FEATHER_MODEL_MODEL_FORMAT_H
#define FEATHER_MODEL_MODEL_FORMAT_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "util/types.h"

namespace feather {
namespace model {

using AttributeValue = std::variant<int64_t, float, std::string, std::vector<int64_t>, std::vector<float>>;

struct TensorDesc {
    std::string name;
    std::vector<int64_t> dims;
    DataType data_type{DataType::UNKNOWN};
    DataLayout layout{DataLayout::ND};
    QuantizationParams quantization{};
};

struct WeightLocation {
    std::string tensor_name;
    std::string shard_path;
    uint64_t offset{};
    uint64_t byte_size{};
    std::string checksum;
};

struct ValueDesc {
    TensorDesc tensor;
    bool constant{};
    WeightLocation weight;
};

struct NodeDesc {
    std::string name;
    std::string op_type;
    std::string domain;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::unordered_map<std::string, AttributeValue> attributes;
};

struct GraphDesc {
    std::string name;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<ValueDesc> values;
    std::vector<NodeDesc> nodes;
};

struct ModelDesc {
    std::string name;
    int64_t version{};
    GraphDesc graph;
};

}  // namespace model
}  // namespace feather

#endif  // FEATHER_MODEL_MODEL_FORMAT_H
