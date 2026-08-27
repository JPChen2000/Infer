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
