#include "model/model_io.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <utility>
#include <vector>

namespace feather {
namespace model {
namespace {

constexpr char kMagic[8] = {'F', 'T', 'H', 'M', 'O', 'D', 'L', '\0'};
constexpr uint32_t kFormatVersion = 1;
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

size_t DataTypeSize(DataType dtype) {
    switch (dtype) {
        case DataType::INT4:
            return 0;
        case DataType::INT8:
        case DataType::UINT8:
        case DataType::BOOL:
            return 1;
        case DataType::FP16:
            return 2;
        case DataType::FP32:
        case DataType::INT32:
            return 4;
        case DataType::INT64:
            return 8;
        default:
            return 0;
    }
}

uint64_t TensorByteSize(const TensorDesc& desc) {
    auto elem_size = DataTypeSize(desc.data_type);
    if (elem_size == 0) {
        return 0;
    }
    uint64_t numel = 1;
    for (auto dim : desc.dims) {
        if (dim < 0) {
            return 0;
        }
        numel *= static_cast<uint64_t>(dim);
    }
    return numel * elem_size;
}

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
        if (!ReadPod(&size) || size > std::numeric_limits<size_t>::max()) {
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
    if (!reader->ReadPod(&size) || size > std::numeric_limits<size_t>::max()) {
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

void WriteTensorDesc(BinaryWriter* writer, const TensorDesc& desc) {
    writer->WriteString(desc.name);
    writer->WriteVector(desc.dims);
    writer->WritePod(static_cast<int32_t>(desc.data_type));
    writer->WritePod(static_cast<int32_t>(desc.layout));
}

bool ReadTensorDesc(BinaryReader* reader, TensorDesc* desc) {
    int32_t data_type = 0;
    int32_t layout = 0;
    if (!reader->ReadString(&desc->name) || !reader->ReadVector(&desc->dims) ||
        !reader->ReadPod(&data_type) || !reader->ReadPod(&layout)) {
        return false;
    }
    desc->data_type = static_cast<DataType>(data_type);
    desc->layout = static_cast<DataLayout>(layout);
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

void WriteValueDesc(BinaryWriter* writer, const ValueDesc& value) {
    WriteTensorDesc(writer, value.tensor);
    uint8_t constant = value.constant ? 1 : 0;
    writer->WritePod(constant);
    WriteWeightLocation(writer, value.weight);
}

bool ReadValueDesc(BinaryReader* reader, ValueDesc* value) {
    uint8_t constant = 0;
    if (!ReadTensorDesc(reader, &value->tensor) || !reader->ReadPod(&constant) ||
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
    if (!reader->ReadPod(&attr_size) || attr_size > std::numeric_limits<size_t>::max()) {
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

void WriteGraphDesc(BinaryWriter* writer, const GraphDesc& graph) {
    writer->WriteString(graph.name);
    WriteStringVector(writer, graph.inputs);
    WriteStringVector(writer, graph.outputs);
    uint64_t value_size = graph.values.size();
    writer->WritePod(value_size);
    for (const auto& value : graph.values) {
        WriteValueDesc(writer, value);
    }
    uint64_t node_size = graph.nodes.size();
    writer->WritePod(node_size);
    for (const auto& node : graph.nodes) {
        WriteNodeDesc(writer, node);
    }
}

bool ReadGraphDesc(BinaryReader* reader, GraphDesc* graph) {
    if (!reader->ReadString(&graph->name) || !ReadStringVector(reader, &graph->inputs) ||
        !ReadStringVector(reader, &graph->outputs)) {
        return false;
    }
    uint64_t value_size = 0;
    if (!reader->ReadPod(&value_size) || value_size > std::numeric_limits<size_t>::max()) {
        return false;
    }
    graph->values.resize(static_cast<size_t>(value_size));
    for (auto& value : graph->values) {
        if (!ReadValueDesc(reader, &value)) {
            return false;
        }
    }
    uint64_t node_size = 0;
    if (!reader->ReadPod(&node_size) || node_size > std::numeric_limits<size_t>::max()) {
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

std::vector<char> SerializeModel(const ModelDesc& model) {
    BinaryWriter writer;
    writer.WriteString(model.name);
    writer.WritePod(model.version);
    WriteGraphDesc(&writer, model.graph);
    return writer.data();
}

bool DeserializeModel(const char* data, size_t size, ModelDesc* model) {
    BinaryReader reader(data, size);
    return reader.ReadString(&model->name) && reader.ReadPod(&model->version) &&
           ReadGraphDesc(&reader, &model->graph) && reader.Finished();
}

}  // namespace

bool ModelWriter::Save(const std::string& path, const ModelDesc& model,
                       const std::unordered_map<std::string, std::shared_ptr<Tensor>>& weights) {
    ModelDesc output = model;
    std::vector<char> metadata;
    for (int pass = 0; pass < 4; ++pass) {
        metadata = SerializeModel(output);
        auto weight_offset = AlignUp(sizeof(FileHeader) + metadata.size(), kWeightAlignment);

        for (auto& value : output.graph.values) {
            if (!value.constant) {
                continue;
            }
            auto it = weights.find(value.tensor.name);
            if (it == weights.end() || it->second == nullptr || !it->second->IsInitialized()) {
                return false;
            }
            auto expected_size = TensorByteSize(value.tensor);
            auto actual_size = static_cast<uint64_t>(it->second->memory_size());
            if (expected_size != 0 && expected_size != actual_size) {
                return false;
            }
            value.weight.tensor_name = value.tensor.name;
            value.weight.shard_path = path;
            value.weight.offset = weight_offset;
            value.weight.byte_size = actual_size;
            weight_offset = AlignUp(weight_offset + actual_size, kWeightAlignment);
        }

        auto updated_metadata = SerializeModel(output);
        if (updated_metadata.size() == metadata.size()) {
            metadata = std::move(updated_metadata);
            break;
        }
        metadata = std::move(updated_metadata);
    }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.good()) {
        return false;
    }

    FileHeader header {};
    std::copy(std::begin(kMagic), std::end(kMagic), std::begin(header.magic));
    header.version = kFormatVersion;
    header.metadata_size = metadata.size();
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    out.write(metadata.data(), metadata.size());

    uint64_t current = sizeof(header) + metadata.size();
    for (const auto& value : output.graph.values) {
        if (!value.constant) {
            continue;
        }
        while (current < value.weight.offset) {
            char zero = 0;
            out.write(&zero, 1);
            ++current;
        }
        auto it = weights.find(value.tensor.name);
        out.write(static_cast<const char*>(it->second->raw_data()), it->second->memory_size());
        current += it->second->memory_size();
    }

    return out.good();
}

bool ModelLoader::Load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.good()) {
        return false;
    }

    FileHeader header {};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in.good() || std::memcmp(header.magic, kMagic, sizeof(kMagic)) != 0 ||
        header.version != kFormatVersion ||
        header.metadata_size > std::numeric_limits<size_t>::max()) {
        return false;
    }

    std::vector<char> metadata(static_cast<size_t>(header.metadata_size));
    in.read(metadata.data(), metadata.size());
    if (!in.good() || !DeserializeModel(metadata.data(), metadata.size(), &model_)) {
        return false;
    }

    path_ = path;
    for (auto& value : model_.graph.values) {
        if (value.constant) {
            value.weight.shard_path = path_;
        }
    }
    return weight_store_.AddShard(path_);
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
