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

#include "model/model_io.h"
#include "core/shape_utils.h"
#include <stdexcept>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace feather {
namespace model {
namespace {

constexpr char kMagic[8] = {'F', 'T', 'H', 'M', 'O', 'D', 'L', '\0'};
constexpr uint32_t kFormatVersion = 5;
constexpr uint64_t kWeightAlignment = 64;

struct FileHeader {
    char magic[8];
    uint32_t version;
    uint32_t reserved;
    uint64_t metadata_size;
};

uint64_t AlignUp(uint64_t value, uint64_t align) {
    return (value + align - 1) / align * align;
}

constexpr uint64_t kMaxMetadataBytes = 256ULL * 1024 * 1024;
constexpr uint64_t kMaxCollectionSize = 1024ULL * 1024;


class BinaryWriter {
   public:
    void WriteRaw(const void* data, size_t size) {
        const auto* bytes = static_cast<const char*>(data);
        buffer_.insert(buffer_.end(), bytes, bytes + size);
    }

    template <typename T>
    void WritePod(const T& value) {
        WriteRaw(&value, sizeof(T));
    }

    void WriteString(const std::string& value) {
        uint64_t size = value.size();
        WritePod(size);
        WriteRaw(value.data(), value.size());
    }

    template <typename T>
    void WriteVector(const std::vector<T>& values) {
        uint64_t size = values.size();
        WritePod(size);
        for (const auto& value : values) {
            WritePod(value);
        }
    }

    const std::vector<char>& data() const { return buffer_; }

   private:
    std::vector<char> buffer_;
};

class BinaryReader {
   public:
    BinaryReader(const char* data, size_t size) : data_(data), size_(size) {}

    template <typename T>
    bool ReadPod(T* value) {
        if (offset_ > size_ || sizeof(T) > size_ - offset_) {
            return false;
        }
        std::memcpy(value, data_ + offset_, sizeof(T));
        offset_ += sizeof(T);
        return true;
    }

    bool ReadString(std::string* value) {
        uint64_t size = 0;
        if (!ReadPod(&size) || size > std::numeric_limits<size_t>::max()) {
            return false;
        }
        auto n = static_cast<size_t>(size);
        if (offset_ > size_ || n > size_ - offset_) {
            return false;
        }
        value->assign(data_ + offset_, n);
        offset_ += n;
        return true;
    }

    template <typename T>
    bool ReadVector(std::vector<T>* values) {
        uint64_t size = 0;
        if (!ReadPod(&size) || size > Remaining() / sizeof(T)) {
            return false;
        }
        values->resize(static_cast<size_t>(size));
        for (auto& value : *values) {
            if (!ReadPod(&value)) {
                return false;
            }
        }
        return true;
    }

    size_t Remaining() const { return size_ - offset_; }
    bool CanReadCollection(uint64_t count, size_t min_item_bytes) const {
        return count <= kMaxCollectionSize && count <= Remaining() / min_item_bytes;
    }
    bool Finished() const { return offset_ == size_; }

   private:
    const char* data_;
    size_t size_;
    size_t offset_{};
};

void WriteStringVector(BinaryWriter* writer, const std::vector<std::string>& values) {
    uint64_t size = values.size();
    writer->WritePod(size);
    for (const auto& value : values) {
        writer->WriteString(value);
    }
}

bool ReadStringVector(BinaryReader* reader, std::vector<std::string>* values) {
    uint64_t size = 0;
    if (!reader->ReadPod(&size) || !reader->CanReadCollection(size, sizeof(uint64_t))) {
        return false;
    }
    values->resize(static_cast<size_t>(size));
    for (auto& value : *values) {
        if (!reader->ReadString(&value)) {
            return false;
        }
    }
    return true;
}

void WriteAttribute(BinaryWriter* writer, const AttributeValue& attr) {
    uint32_t tag = attr.index();
    writer->WritePod(tag);
    switch (tag) {
        case 0:
            writer->WritePod(std::get<int64_t>(attr));
            break;
        case 1:
            writer->WritePod(std::get<float>(attr));
            break;
        case 2:
            writer->WriteString(std::get<std::string>(attr));
            break;
        case 3:
            writer->WriteVector(std::get<std::vector<int64_t>>(attr));
            break;
        case 4:
            writer->WriteVector(std::get<std::vector<float>>(attr));
            break;
        default:
            break;
    }
}

bool ReadAttribute(BinaryReader* reader, AttributeValue* attr) {
    uint32_t tag = 0;
    if (!reader->ReadPod(&tag)) {
        return false;
    }
    switch (tag) {
        case 0: {
            int64_t value = 0;
            if (!reader->ReadPod(&value)) return false;
            *attr = value;
            return true;
        }
        case 1: {
            float value = 0.0f;
            if (!reader->ReadPod(&value)) return false;
            *attr = value;
            return true;
        }
        case 2: {
            std::string value;
            if (!reader->ReadString(&value)) return false;
            *attr = value;
            return true;
        }
        case 3: {
            std::vector<int64_t> value;
            if (!reader->ReadVector(&value)) return false;
            *attr = value;
            return true;
        }
        case 4: {
            std::vector<float> value;
            if (!reader->ReadVector(&value)) return false;
            *attr = value;
            return true;
        }
        default:
            return false;
    }
}

void WriteTensorDesc(BinaryWriter* writer, const TensorDesc& desc, bool include_quantization,
                     bool include_vector_quantization) {
    writer->WriteString(desc.name);
    writer->WriteVector(desc.dims);
    writer->WritePod(static_cast<int32_t>(desc.data_type));
    writer->WritePod(static_cast<int32_t>(desc.layout));
    if (include_quantization) {
        const uint8_t enabled = desc.quantization.enabled ? 1 : 0;
        writer->WritePod(enabled);
        writer->WritePod(desc.quantization.scale);
        writer->WritePod(static_cast<int32_t>(desc.quantization.granularity));
        writer->WritePod(desc.quantization.axis);
        writer->WritePod(desc.quantization.block_size);
        if (include_vector_quantization) {
            writer->WritePod(desc.quantization.zero_point);
            writer->WriteVector(desc.quantization.scales);
            writer->WriteVector(desc.quantization.zero_points);
        }
    }
}

bool ReadTensorDesc(BinaryReader* reader, TensorDesc* desc, bool include_quantization,
                    bool include_vector_quantization) {
    int32_t data_type = 0;
    int32_t layout = 0;
    if (!reader->ReadString(&desc->name) || !reader->ReadVector(&desc->dims) ||
        !reader->ReadPod(&data_type) || !reader->ReadPod(&layout)) {
        return false;
    }
    desc->data_type = static_cast<DataType>(data_type);
    desc->layout = static_cast<DataLayout>(layout);
    if (include_quantization) {
        uint8_t enabled = 0;
        float scale = 1.0f;
        int32_t granularity = 0;
        if (!reader->ReadPod(&enabled) || !reader->ReadPod(&scale) || !reader->ReadPod(&granularity) ||
            !reader->ReadPod(&desc->quantization.axis) || !reader->ReadPod(&desc->quantization.block_size)) {
            return false;
        }
        desc->quantization.enabled = enabled != 0;
        desc->quantization.scale = scale;
        desc->quantization.granularity = static_cast<QuantizationGranularity>(granularity);
        desc->quantization.zero_point = 0;
        desc->quantization.scales = {scale};
        desc->quantization.zero_points = {0};
        if (include_vector_quantization &&
            (!reader->ReadPod(&desc->quantization.zero_point) ||
             !reader->ReadVector(&desc->quantization.scales) ||
             !reader->ReadVector(&desc->quantization.zero_points))) {
            return false;
        }
        if (desc->quantization.scales.empty()) {
            desc->quantization.scales = {desc->quantization.scale};
        }
        if (desc->quantization.zero_points.empty()) {
            desc->quantization.zero_points = {desc->quantization.zero_point};
        }
        desc->quantization.scale = desc->quantization.scales.front();
        desc->quantization.zero_point = desc->quantization.zero_points.front();
    }
    return true;
}

void WriteWeightLocation(BinaryWriter* writer, const WeightLocation& location) {
    writer->WriteString(location.tensor_name);
    writer->WriteString(location.shard_path);
    writer->WritePod(location.offset);
    writer->WritePod(location.byte_size);
    writer->WriteString(location.checksum);
}

bool ReadWeightLocation(BinaryReader* reader, WeightLocation* location) {
    return reader->ReadString(&location->tensor_name) && reader->ReadString(&location->shard_path) &&
           reader->ReadPod(&location->offset) && reader->ReadPod(&location->byte_size) &&
           reader->ReadString(&location->checksum);
}

void WriteValueDesc(BinaryWriter* writer, const ValueDesc& value, bool include_quantization,
                    bool include_vector_quantization) {
    WriteTensorDesc(writer, value.tensor, include_quantization, include_vector_quantization);
    uint8_t constant = value.constant ? 1 : 0;
    writer->WritePod(constant);
    WriteWeightLocation(writer, value.weight);
}

bool ReadValueDesc(BinaryReader* reader, ValueDesc* value, bool include_quantization,
                   bool include_vector_quantization) {
    uint8_t constant = 0;
    if (!ReadTensorDesc(reader, &value->tensor, include_quantization, include_vector_quantization) ||
        !reader->ReadPod(&constant) ||
        !ReadWeightLocation(reader, &value->weight)) {
        return false;
    }
    value->constant = constant != 0;
    return true;
}

void WriteNodeDesc(BinaryWriter* writer, const NodeDesc& node) {
    writer->WriteString(node.name);
    writer->WriteString(node.op_type);
    writer->WriteString(node.domain);
    WriteStringVector(writer, node.inputs);
    WriteStringVector(writer, node.outputs);
    uint64_t attr_size = node.attributes.size();
    writer->WritePod(attr_size);
    for (const auto& attr : node.attributes) {
        writer->WriteString(attr.first);
        WriteAttribute(writer, attr.second);
    }
}

bool ReadNodeDesc(BinaryReader* reader, NodeDesc* node) {
    if (!reader->ReadString(&node->name) || !reader->ReadString(&node->op_type) ||
        !reader->ReadString(&node->domain) || !ReadStringVector(reader, &node->inputs) ||
        !ReadStringVector(reader, &node->outputs)) {
        return false;
    }
    uint64_t attr_size = 0;
    if (!reader->ReadPod(&attr_size) || !reader->CanReadCollection(attr_size, sizeof(uint64_t) + sizeof(uint32_t))) {
        return false;
    }
    for (uint64_t i = 0; i < attr_size; ++i) {
        std::string key;
        AttributeValue value;
        if (!reader->ReadString(&key) || !ReadAttribute(reader, &value)) {
            return false;
        }
        node->attributes.emplace(std::move(key), std::move(value));
    }
    return true;
}

void WriteGraphDesc(BinaryWriter* writer, const GraphDesc& graph, bool include_quantization,
                    bool include_vector_quantization) {
    writer->WriteString(graph.name);
    WriteStringVector(writer, graph.inputs);
    WriteStringVector(writer, graph.outputs);
    uint64_t value_size = graph.values.size();
    writer->WritePod(value_size);
    for (const auto& value : graph.values) {
        WriteValueDesc(writer, value, include_quantization, include_vector_quantization);
    }
    uint64_t node_size = graph.nodes.size();
    writer->WritePod(node_size);
    for (const auto& node : graph.nodes) {
        WriteNodeDesc(writer, node);
    }
}

bool ReadGraphDesc(BinaryReader* reader, GraphDesc* graph, bool include_quantization,
                   bool include_vector_quantization) {
    if (!reader->ReadString(&graph->name) || !ReadStringVector(reader, &graph->inputs) ||
        !ReadStringVector(reader, &graph->outputs)) {
        return false;
    }
    uint64_t value_size = 0;
    if (!reader->ReadPod(&value_size) || !reader->CanReadCollection(value_size, 65)) {
        return false;
    }
    graph->values.resize(static_cast<size_t>(value_size));
    for (auto& value : graph->values) {
        if (!ReadValueDesc(reader, &value, include_quantization, include_vector_quantization)) {
            return false;
        }
    }
    uint64_t node_size = 0;
    if (!reader->ReadPod(&node_size) || !reader->CanReadCollection(node_size, 48)) {
        return false;
    }
    graph->nodes.resize(static_cast<size_t>(node_size));
    for (auto& node : graph->nodes) {
        if (!ReadNodeDesc(reader, &node)) {
            return false;
        }
    }
    return true;
}

void WriteMetadata(BinaryWriter* writer, const std::unordered_map<std::string, AttributeValue>& metadata) {
    writer->WritePod(static_cast<uint64_t>(metadata.size()));
    std::vector<std::string> keys;
    for (const auto& entry : metadata) keys.push_back(entry.first);
    std::sort(keys.begin(), keys.end());
    for (const auto& key : keys) { writer->WriteString(key); WriteAttribute(writer, metadata.at(key)); }
}

bool ReadMetadata(BinaryReader* reader, std::unordered_map<std::string, AttributeValue>* metadata) {
    uint64_t count = 0;
    if (!reader->ReadPod(&count) || !reader->CanReadCollection(count, 12)) return false;
    for (uint64_t i = 0; i < count; ++i) {
        std::string key; AttributeValue value;
        if (!reader->ReadString(&key) || !ReadAttribute(reader, &value) ||
            !metadata->emplace(std::move(key), std::move(value)).second) return false;
    }
    return true;
}

std::vector<char> SerializeModel(const ModelDesc& model, bool include_quantization,
                                 bool include_vector_quantization) {
    BinaryWriter writer;
    writer.WriteString(model.name); writer.WritePod(model.version);
    WriteGraphDesc(&writer, model.graph, include_quantization, include_vector_quantization);
    WriteMetadata(&writer, model.metadata);
    writer.WritePod(static_cast<uint64_t>(model.variants.size()));
    for (const auto& variant : model.variants) {
        writer.WriteString(variant.name);
        WriteGraphDesc(&writer, variant.graph, include_quantization, include_vector_quantization);
        WriteMetadata(&writer, variant.metadata);
    }
    return writer.data();
}

bool DeserializeModel(const char* data, size_t size, ModelDesc* model, bool include_quantization,
                      bool include_vector_quantization, bool include_metadata, bool include_variants) {
    BinaryReader reader(data, size);
    if (!reader.ReadString(&model->name) || !reader.ReadPod(&model->version) ||
        !ReadGraphDesc(&reader, &model->graph, include_quantization, include_vector_quantization) ||
        (include_metadata && !ReadMetadata(&reader, &model->metadata))) return false;
    if (include_variants) {
        uint64_t count = 0;
        if (!reader.ReadPod(&count) || !reader.CanReadCollection(count, 48)) return false;
        std::unordered_set<std::string> names;
        for (uint64_t i = 0; i < count; ++i) {
            ModelDesc::Variant variant;
            if (!reader.ReadString(&variant.name) || variant.name.empty() || !names.insert(variant.name).second ||
                !ReadGraphDesc(&reader, &variant.graph, include_quantization, include_vector_quantization) ||
                !ReadMetadata(&reader, &variant.metadata)) return false;
            model->variants.push_back(std::move(variant));
        }
    }
    return reader.Finished();
}

}  // namespace

bool ModelWriter::Save(const std::string& path, const ModelDesc& model,
                       const std::unordered_map<std::string, std::shared_ptr<Tensor>>& weights) {
    ModelDesc output = model;
    struct Payload { ValueDesc* value; std::shared_ptr<Tensor> tensor; };
    std::vector<Payload> payloads;
    const auto collect = [&](GraphDesc& graph, const std::string& prefix) {
        for (auto& value : graph.values) if (value.constant) {
            auto it = weights.find(prefix + value.tensor.name);
            if (it == weights.end() && !prefix.empty()) it = weights.find(value.tensor.name);
            size_t bytes = 0;
            if (it == weights.end() || !it->second || !it->second->IsInitialized() ||
                !CheckedTensorBytes(value.tensor.dims, value.tensor.data_type, &bytes) ||
                it->second->memory_size() != bytes || it->second->dims().data() != value.tensor.dims ||
                it->second->data_type() != value.tensor.data_type) return false;
            payloads.push_back({&value, it->second});
        }
        return true;
    };
    if (!collect(output.graph, "")) return false;
    std::unordered_set<std::string> names;
    for (auto& variant : output.variants)
        if (variant.name.empty() || !names.insert(variant.name).second ||
            !collect(variant.graph, variant.name + "::")) return false;
    std::vector<char> metadata;
    for (int pass = 0; pass < 4; ++pass) {
        metadata = SerializeModel(output, true, true);
        auto offset = AlignUp(sizeof(FileHeader) + metadata.size(), kWeightAlignment);
        std::map<std::pair<uintptr_t, size_t>, uint64_t> locations;
        for (auto& payload : payloads) {
            auto& weight = payload.value->weight;
            weight.tensor_name = payload.value->tensor.name; weight.shard_path = path;
            weight.byte_size = payload.tensor->memory_size();
            const auto key = std::make_pair(reinterpret_cast<uintptr_t>(payload.tensor->raw_data()), payload.tensor->memory_size());
            auto found = locations.emplace(key, offset);
            weight.offset = found.first->second;
            if (found.second) {
                if (offset > std::numeric_limits<uint64_t>::max() - kWeightAlignment ||
                    weight.byte_size > std::numeric_limits<uint64_t>::max() - offset - kWeightAlignment) return false;
                offset = AlignUp(offset + weight.byte_size, kWeightAlignment);
            }
        }
    }
    metadata = SerializeModel(output, true, true);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.good()) return false;
    FileHeader header {};
    std::copy(std::begin(kMagic), std::end(kMagic), std::begin(header.magic));
    header.version = kFormatVersion; header.metadata_size = metadata.size();
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    out.write(metadata.data(), metadata.size());
    uint64_t current = sizeof(header) + metadata.size();
    std::unordered_set<uint64_t> written;
    for (const auto& payload : payloads) {
        const auto& weight = payload.value->weight;
        if (!weight.byte_size && current >= weight.offset) continue;
        if (weight.byte_size && !written.insert(weight.offset).second) continue;
        if (current > weight.offset) return false;
        std::vector<char> padding(static_cast<size_t>(weight.offset - current), 0);
        out.write(padding.data(), padding.size());
        out.write(static_cast<const char*>(payload.tensor->raw_data()), weight.byte_size);
        current = weight.offset + weight.byte_size;
    }
    return out.good();
}

bool ModelLoader::Load(const std::string& path) {
    try {
        std::ifstream in(path, std::ios::binary | std::ios::ate);
        if (!in.good()) return false;
        const auto length = in.tellg();
        if (length < static_cast<std::streamoff>(sizeof(FileHeader))) return false;
        in.seekg(0);
        FileHeader header {};
        in.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (!in.good() || std::memcmp(header.magic, kMagic, sizeof(kMagic)) != 0 ||
            header.version < 1 || header.version > kFormatVersion ||
            header.metadata_size > kMaxMetadataBytes ||
            header.metadata_size > static_cast<uint64_t>(length) - sizeof(header)) return false;

        std::vector<char> metadata(static_cast<size_t>(header.metadata_size));
        in.read(metadata.data(), metadata.size());
        ModelDesc parsed;
        if (!in.good() || !DeserializeModel(metadata.data(), metadata.size(), &parsed,
                                            header.version >= 2, header.version >= 3,
                                            header.version >= 4, header.version >= 5)) return false;
        WeightStore new_store;
        if (!new_store.AddShard(path)) return false;
        const auto validate_weights = [&](GraphDesc& graph) {
            for (auto& value : graph.values) if (value.constant) {
                value.weight.shard_path = path;
                if (value.weight.offset < sizeof(header) + header.metadata_size ||
                    new_store.CreateTensorView(value.tensor, value.weight) == nullptr) return false;
            }
            return true;
        };
        if (!validate_weights(parsed.graph)) return false;
        for (auto& variant : parsed.variants) if (!validate_weights(variant.graph)) return false;
        path_ = path;
        model_ = std::move(parsed);
        weight_store_ = std::move(new_store);
        return true;
    } catch (const std::bad_alloc&) {
        return false;
    } catch (const std::length_error&) {
        return false;
    }
}

bool ModelLoader::SelectVariant(const std::string& name, ModelLoader* destination) const {
    if (!destination || destination == this) return false;
    for (const auto& variant : model_.variants) if (variant.name == name) {
        ModelDesc selected;
        selected.name = model_.name + ":" + name; selected.version = model_.version;
        selected.graph = variant.graph; selected.metadata = variant.metadata;
        destination->path_ = path_; destination->model_ = std::move(selected);
        destination->weight_store_ = weight_store_;
        return true;
    }
    return false;
}

std::shared_ptr<Tensor> ModelLoader::CreateWeightTensor(const std::string& tensor_name) {
    auto value = FindValue(tensor_name);
    if (value == nullptr || !value->constant) {
        return nullptr;
    }
    return weight_store_.CreateTensorView(value->tensor, value->weight);
}

const ValueDesc* ModelLoader::FindValue(const std::string& tensor_name) const {
    for (const auto& value : model_.graph.values) {
        if (value.tensor.name == tensor_name) {
            return &value;
        }
    }
    return nullptr;
}

}  // namespace model
}  // namespace feather
