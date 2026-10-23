// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#include "demo/qwen_sequence.h"

#include <algorithm>
#include <limits>
#include <unordered_set>

namespace feather {
namespace demo {
namespace {
constexpr char kPrefix[] = "qwen.sequence.";
constexpr char kValue[] = "qwen.sequence.value.";
constexpr char kConstant[] = "qwen.sequence.constant.";
constexpr char kAttribute[] = "qwen.sequence.attribute.";
constexpr char kArange[] = "qwen.sequence.arange.";
constexpr size_t kMaxShapeElements = 4096;

bool StartsWith(const std::string& text, const std::string& prefix) {
    return text.compare(0, prefix.size(), prefix) == 0;
}

bool Resolve(int64_t sequence, int64_t offset, int64_t* value) {
    if ((offset > 0 && sequence > std::numeric_limits<int64_t>::max() - offset) ||
        (offset < 0 && sequence < std::numeric_limits<int64_t>::min() - offset)) return false;
    *value = sequence + offset;
    return *value >= 0;
}

bool Apply(const model::AttributeValue& binding, int64_t original, int64_t sequence,
           int64_t maximum, std::vector<int64_t>* values) {
    const auto* pairs = std::get_if<std::vector<int64_t>>(&binding);
    if (!pairs || pairs->empty() || pairs->size() % 2) return false;
    std::unordered_set<int64_t> indices;
    for (size_t i = 0; i < pairs->size(); i += 2) {
        const int64_t index = (*pairs)[i], offset = (*pairs)[i + 1];
        int64_t expected = 0, resolved = 0, largest = 0;
        if (index < 0 || static_cast<uint64_t>(index) >= values->size() ||
            !indices.insert(index).second || !Resolve(original, offset, &expected) ||
            !Resolve(sequence, offset, &resolved) || !Resolve(maximum, offset, &largest) ||
            (*values)[index] != expected) return false;
        (*values)[index] = resolved;
    }
    return true;
}
}  // namespace

bool SpecializeQwenSequence(model::ModelLoader& loader, int64_t sequence,
                            model::ModelDesc* effective_model,
                            std::unordered_map<std::string, std::shared_ptr<Tensor>>* constant_overrides,
                            std::string* error) {
    const auto fail = [error](const std::string& message) {
        if (error) *error = message;
        return false;
    };
    if (!effective_model || !constant_overrides) return fail("Missing sequence specialization output");
    if (sequence < 0) return fail("Sequence length must be zero or positive");
    auto candidate = *effective_model;
    auto overrides = *constant_overrides;
    std::unordered_map<std::string, model::ValueDesc*> values;
    for (auto& value : candidate.graph.values) {
        if (!values.emplace(value.tensor.name, &value).second)
            return fail("Duplicate graph value: " + value.tensor.name);
    }
    const auto input = values.find("token_ids");
    if (input == values.end() || input->second->constant ||
        input->second->tensor.data_type != DataType::INT64 ||
        input->second->tensor.dims.size() != 2 || input->second->tensor.dims[0] != 1 ||
        input->second->tensor.dims[1] < 1 ||
        std::find(candidate.graph.inputs.begin(), candidate.graph.inputs.end(), "token_ids") ==
            candidate.graph.inputs.end()) return fail("token_ids must be an INT64 [1,sequence] input");
    const auto scalar = [&candidate](const std::string& key) -> const int64_t* {
        const auto it = candidate.metadata.find(key);
        return it == candidate.metadata.end() ? nullptr : std::get_if<int64_t>(&it->second);
    };
    bool annotated = false;
    for (const auto& entry : candidate.metadata) annotated |= StartsWith(entry.first, kPrefix);
    if (!annotated) {
        const auto* version = scalar("qwen.execution.version");
        if (version && *version >= 2) return fail("Missing Qwen sequence metadata");
        if (sequence && sequence != input->second->tensor.dims[1])
            return fail("Legacy Qwen graph has a fixed sequence length");
        if (error) error->clear();
        return true;
    }
    const auto* version = scalar("qwen.sequence.version");
    const auto* original = scalar("qwen.sequence.default");
    const auto* maximum = scalar("qwen.sequence.maximum");
    const auto* execution = scalar("qwen.execution.sequence");
    const auto* execution_version = scalar("qwen.execution.version");
    if (!version || *version != 1 || !original || *original < 1 || !maximum || *maximum < *original ||
        !execution_version || *execution_version != 2 || !execution || *execution != *original ||
        input->second->tensor.dims[1] != *original)
        return fail("Invalid or incomplete Qwen sequence metadata");
    sequence = sequence == 0 ? *original : sequence;
    if (sequence < 1 || sequence > *maximum) return fail("Requested sequence length is outside model limits");
    const auto token_binding = candidate.metadata.find("qwen.sequence.value.token_ids");
    if (token_binding == candidate.metadata.end() ||
        !std::holds_alternative<std::vector<int64_t>>(token_binding->second) ||
        std::get<std::vector<int64_t>>(token_binding->second) != std::vector<int64_t>({1, 0}))
        return fail("token_ids sequence binding must target axis 1 with offset 0");

    // Identify generated constants first: unordered metadata traversal must not
    // affect whether a constant descriptor is accepted as a dimension target.
    std::unordered_set<std::string> generated;
    for (const auto& entry : candidate.metadata) {
        if (!StartsWith(entry.first, kArange)) continue;
        const auto name = entry.first.substr(sizeof(kArange) - 1);
        const auto it = values.find(name);
        const auto* kind = std::get_if<int64_t>(&entry.second);
        const auto binding = candidate.metadata.find(std::string(kValue) + name);
        if (!kind || *kind != 1 || it == values.end() || !it->second->constant ||
            it->second->tensor.data_type != DataType::FP32 ||
            it->second->tensor.dims != std::vector<int64_t>({*original}) ||
            binding == candidate.metadata.end() ||
            !std::holds_alternative<std::vector<int64_t>>(binding->second) ||
            std::get<std::vector<int64_t>>(binding->second) != std::vector<int64_t>({0, 0}))
            return fail("Invalid sequence arange descriptor: " + entry.first);
        generated.insert(name);
    }
    if (values.count("qwen_chunk_offsets") && !generated.count("qwen_chunk_offsets"))
        return fail("Missing runtime generation binding for qwen_chunk_offsets");
    for (const auto& entry : candidate.metadata) {
        const auto& key = entry.first;
        if (!StartsWith(key, kPrefix) || key == "qwen.sequence.version" ||
            key == "qwen.sequence.default" || key == "qwen.sequence.maximum") continue;
        if (StartsWith(key, kValue)) {
            const auto name = key.substr(sizeof(kValue) - 1);
            const auto it = values.find(name);
            if (it == values.end() || (it->second->constant && !generated.count(name)) ||
                !Apply(entry.second, *original, sequence, *maximum, &it->second->tensor.dims))
                return fail("Invalid sequence dimension binding: " + key);
            size_t bytes = 0;
            if (!CheckedTensorBytes(it->second->tensor.dims, it->second->tensor.data_type, &bytes) ||
                std::any_of(it->second->tensor.dims.begin(), it->second->tensor.dims.end(),
                            [](int64_t dim) { return dim < 1; }))
                return fail("Invalid specialized tensor size: " + name);
        } else if (StartsWith(key, kConstant)) {
            const auto name = key.substr(sizeof(kConstant) - 1);
            const auto it = values.find(name);
            size_t count = 0;
            if (it == values.end() || !it->second->constant || generated.count(name) ||
                it->second->tensor.data_type != DataType::INT64 ||
                !CheckedElementCount(it->second->tensor.dims, &count) || !count || count > kMaxShapeElements)
                return fail("Sequence payload binding must target a small INT64 shape tensor: " + key);
            const auto previous = overrides.find(name);
            auto source = previous == overrides.end() ? loader.CreateWeightTensor(name) : previous->second;
            if (!source || source->data_type() != DataType::INT64 || source->dims().data() != it->second->tensor.dims ||
                source->memory_size() != count * sizeof(int64_t))
                return fail("Cannot read sequence shape tensor: " + name);
            const auto* data = source->data<int64_t>();
            std::vector<int64_t> payload(data, data + count);
            if (!Apply(entry.second, *original, sequence, *maximum, &payload))
                return fail("Invalid sequence shape payload binding: " + key);
            auto replacement = std::make_shared<Tensor>();
            replacement->Assign<int64_t>(payload, it->second->tensor.dims);
            replacement->set_layout(source->layout());
            replacement->set_quantization(source->quantization());
            replacement->set_immutable(true);
            overrides[name] = std::move(replacement);
        } else if (StartsWith(key, kAttribute)) {
            const auto target = key.substr(sizeof(kAttribute) - 1);
            model::AttributeValue* attribute = nullptr;
            for (auto& node : candidate.graph.nodes) {
                for (auto& item : node.attributes) {
                    if (node.name + "." + item.first != target) continue;
                    if (attribute) return fail("Ambiguous sequence attribute target: " + key);
                    attribute = &item.second;
                }
            }
            if (!attribute) return fail("Missing sequence attribute target: " + key);
            if (auto* vector = std::get_if<std::vector<int64_t>>(attribute)) {
                if (!Apply(entry.second, *original, sequence, *maximum, vector))
                    return fail("Invalid sequence vector attribute binding: " + key);
            } else if (auto* scalar_value = std::get_if<int64_t>(attribute)) {
                std::vector<int64_t> payload{*scalar_value};
                if (!Apply(entry.second, *original, sequence, *maximum, &payload))
                    return fail("Invalid sequence scalar attribute binding: " + key);
                *scalar_value = payload[0];
            } else return fail("Sequence attribute binding must target integers: " + key);
        } else if (!StartsWith(key, kArange)) return fail("Unknown Qwen sequence metadata key: " + key);
    }
    for (const auto& name : generated) {
        const auto& desc = values.at(name)->tensor;
        if (desc.dims != std::vector<int64_t>({sequence}))
            return fail("Generated sequence arange shape is inconsistent: " + name);
        std::vector<float> payload(static_cast<size_t>(sequence));
        for (int64_t i = 0; i < sequence; ++i) payload[static_cast<size_t>(i)] = static_cast<float>(i);
        auto replacement = std::make_shared<Tensor>();
        replacement->Assign<float>(payload, desc.dims);
        replacement->set_layout(desc.layout);
        replacement->set_quantization(desc.quantization);
        replacement->set_immutable(true);
        overrides[name] = std::move(replacement);
    }
    candidate.metadata["qwen.execution.sequence"] = sequence;
    *effective_model = std::move(candidate);
    *constant_overrides = std::move(overrides);
    if (error) error->clear();
    return true;
}
}  // namespace demo
}  // namespace feather
