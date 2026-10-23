// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#include "demo/qwen_context.h"

#include <algorithm>
#include <limits>
#include <unordered_set>

namespace feather {
namespace demo {
namespace {
constexpr char kPrefix[] = "qwen.context.";
constexpr char kValuePrefix[] = "qwen.context.value.";
constexpr char kConstantPrefix[] = "qwen.context.constant.";
constexpr char kAttributePrefix[] = "qwen.context.attribute.";
constexpr size_t kMaxShapeElements = 4096;

bool StartsWith(const std::string& text, const std::string& prefix) {
    return text.compare(0, prefix.size(), prefix) == 0;
}

bool AddOffset(int64_t context, int64_t offset, int64_t* result) {
    if ((offset > 0 && context > std::numeric_limits<int64_t>::max() - offset) ||
        (offset < 0 && context < std::numeric_limits<int64_t>::min() - offset)) return false;
    *result = context + offset;
    return *result > 0;
}

bool ApplyBinding(const model::AttributeValue& binding, int64_t default_context,
                  int64_t context, int64_t maximum, std::vector<int64_t>* values) {
    const auto* pairs = std::get_if<std::vector<int64_t>>(&binding);
    if (!pairs || pairs->empty() || pairs->size() % 2 != 0) return false;
    std::unordered_set<int64_t> indices;
    for (size_t i = 0; i < pairs->size(); i += 2) {
        const int64_t index = (*pairs)[i];
        const int64_t offset = (*pairs)[i + 1];
        int64_t original = 0, resolved = 0, largest = 0;
        if (index < 0 || static_cast<uint64_t>(index) >= values->size() ||
            !indices.insert(index).second || !AddOffset(default_context, offset, &original) ||
            !AddOffset(context, offset, &resolved) || !AddOffset(maximum, offset, &largest) ||
            (*values)[index] != original) return false;
        (*values)[index] = resolved;
    }
    return true;
}
}  // namespace

bool SpecializeQwenContext(model::ModelLoader& loader, int64_t requested_context,
                           model::ModelDesc* effective_model,
                           std::unordered_map<std::string, std::shared_ptr<Tensor>>* constant_overrides,
                           int64_t* model_context_limit, std::string* error) {
    const auto fail = [error](const std::string& message) {
        if (error) *error = message;
        return false;
    };
    if (!effective_model || !constant_overrides || !model_context_limit)
        return fail("Missing context specialization output");
    if (requested_context < 0) return fail("Context capacity must be zero or positive");

    model::ModelDesc candidate = loader.model();
    std::unordered_map<std::string, model::ValueDesc*> values;
    for (auto& value : candidate.graph.values) {
        if (!values.emplace(value.tensor.name, &value).second)
            return fail("Duplicate graph value: " + value.tensor.name);
    }
    const auto mask_it = values.find("attention_mask");
    if (mask_it == values.end() || mask_it->second->constant ||
        std::find(candidate.graph.inputs.begin(), candidate.graph.inputs.end(), "attention_mask") ==
            candidate.graph.inputs.end()) return fail("Missing attention_mask graph input");
    const auto& mask = mask_it->second->tensor;
    if (mask.dims.size() != 4 || mask.dims[0] != 1 || mask.dims[1] != 1 || mask.dims[2] != 1 ||
        mask.dims[3] < 2 || mask.data_type != DataType::BF16)
        return fail("attention_mask must be BF16 [1,1,1,context] with context at least 2");

    bool annotated = false;
    for (const auto& entry : candidate.metadata) annotated |= StartsWith(entry.first, kPrefix);
    std::unordered_map<std::string, std::shared_ptr<Tensor>> overrides;
    int64_t context = mask.dims[3];
    int64_t maximum = context;
    if (!annotated) {
        if (requested_context != 0 && requested_context != context)
            return fail("Legacy Qwen model has fixed context capacity " + std::to_string(context) +
                        "; re-export the model to select a different context");
    } else {
        const auto scalar = [&candidate](const std::string& key) -> const int64_t* {
            const auto it = candidate.metadata.find(key);
            return it == candidate.metadata.end() ? nullptr : std::get_if<int64_t>(&it->second);
        };
        const auto* version = scalar("qwen.context.version");
        const auto* default_context = scalar("qwen.context.default");
        const auto* maximum_context = scalar("qwen.context.maximum");
        if (!version || *version != 1 || !default_context || !maximum_context ||
            *default_context < 2 || *maximum_context < *default_context || mask.dims[3] != *default_context)
            return fail("Invalid or incomplete Qwen context metadata");
        maximum = *maximum_context;
        context = requested_context == 0 ? *default_context : requested_context;
        if (context < 2 || context > maximum) return fail("Requested context capacity is outside model limits");
        const auto mask_binding = candidate.metadata.find("qwen.context.value.attention_mask");
        if (mask_binding == candidate.metadata.end()) return fail("Missing attention_mask context binding");
        const auto* mask_pairs = std::get_if<std::vector<int64_t>>(&mask_binding->second);
        bool binds_mask_axis = false;
        if (mask_pairs && mask_pairs->size() % 2 == 0) {
            for (size_t i = 0; i < mask_pairs->size(); i += 2)
                binds_mask_axis |= (*mask_pairs)[i] == 3 && (*mask_pairs)[i + 1] == 0;
        }
        if (!binds_mask_axis) return fail("attention_mask context binding must include axis 3 with offset 0");

        for (const auto& entry : candidate.metadata) {
            const auto& key = entry.first;
            if (!StartsWith(key, kPrefix) || key == "qwen.context.version" ||
                key == "qwen.context.default" || key == "qwen.context.maximum") continue;
            if (StartsWith(key, kValuePrefix)) {
                const auto name = key.substr(sizeof(kValuePrefix) - 1);
                const auto it = values.find(name);
                if (it == values.end() || it->second->constant)
                    return fail("Context dimension binding must target a nonconstant value: " + key);
                if (!ApplyBinding(entry.second, *default_context, context, maximum, &it->second->tensor.dims))
                    return fail("Invalid context dimension binding: " + key);
                size_t bytes = 0;
                if (!CheckedTensorBytes(it->second->tensor.dims, it->second->tensor.data_type, &bytes))
                    return fail("Specialized tensor size overflows: " + name);
            } else if (StartsWith(key, kConstantPrefix)) {
                const auto name = key.substr(sizeof(kConstantPrefix) - 1);
                const auto it = values.find(name);
                size_t elements = 0;
                if (it == values.end() || !it->second->constant ||
                    it->second->tensor.data_type != DataType::INT64 ||
                    !CheckedElementCount(it->second->tensor.dims, &elements) ||
                    elements == 0 || elements > kMaxShapeElements)
                    return fail("Context constant binding must target a small INT64 shape tensor: " + key);
                auto source = loader.CreateWeightTensor(name);
                if (!source || source->memory_size() != elements * sizeof(int64_t))
                    return fail("Cannot read context shape tensor: " + name);
                const auto* data = source->data<int64_t>();
                std::vector<int64_t> payload(data, data + elements);
                if (!ApplyBinding(entry.second, *default_context, context, maximum, &payload))
                    return fail("Invalid context shape payload binding: " + key);
                auto replacement = std::make_shared<Tensor>();
                replacement->Assign<int64_t>(payload, source->dims().data());
                replacement->set_layout(source->layout());
                replacement->set_quantization(source->quantization());
                replacement->set_immutable(true);
                overrides.emplace(name, std::move(replacement));
            } else if (StartsWith(key, kAttributePrefix)) {
                const auto target = key.substr(sizeof(kAttributePrefix) - 1);
                model::AttributeValue* attribute = nullptr;
                for (auto& node : candidate.graph.nodes) {
                    for (auto& item : node.attributes) {
                        if (node.name + "." + item.first != target) continue;
                        if (attribute) return fail("Ambiguous context attribute target: " + key);
                        attribute = &item.second;
                    }
                }
                auto* payload = attribute ? std::get_if<std::vector<int64_t>>(attribute) : nullptr;
                if (!payload || !ApplyBinding(entry.second, *default_context, context, maximum, payload))
                    return fail("Invalid context attribute binding: " + key);
            } else {
                return fail("Unknown Qwen context metadata key: " + key);
            }
        }
        if (mask.dims != std::vector<int64_t>({1, 1, 1, context}))
            return fail("Specialized attention_mask shape is inconsistent");
    }
    *effective_model = std::move(candidate);
    *constant_overrides = std::move(overrides);
    *model_context_limit = maximum;
    if (error) error->clear();
    return true;
}
}  // namespace demo
}  // namespace feather
