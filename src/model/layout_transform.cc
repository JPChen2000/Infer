#include "model/layout_transform.h"

#include <algorithm>
#include <vector>

namespace feather {
namespace model {

namespace {

std::vector<int64_t> PermuteNchwToNhwc4D(const std::vector<int64_t>& dims) {
    if (dims.size() != 4) {
        return dims;
    }
    return {dims[0], dims[2], dims[3], dims[1]};
}

int64_t RemapAxisNchwToNhwc(int64_t axis, size_t rank) {
    if (rank != 4) {
        return axis;
    }
    const int64_t normalized = axis >= 0 ? axis : axis + static_cast<int64_t>(rank);
    switch (normalized) {
        case 0:
            return 0;
        case 1:
            return 3;
        case 2:
            return 1;
        case 3:
            return 2;
        default:
            return normalized;
    }
}

bool RewriteVectorIntAttribute(model::NodeDesc* node, const std::string& key, const std::vector<int64_t>& values) {
    if (node == nullptr) {
        return false;
    }
    auto it = node->attributes.find(key);
    if (it == node->attributes.end()) {
        return false;
    }
    it->second = values;
    return true;
}

bool RewriteIntAttribute(model::NodeDesc* node, const std::string& key, int64_t value) {
    if (node == nullptr) {
        return false;
    }
    auto it = node->attributes.find(key);
    if (it == node->attributes.end()) {
        return false;
    }
    it->second = value;
    return true;
}

bool RewriteFloatVectorAttribute(model::NodeDesc* node, const std::string& key, const std::vector<float>& values) {
    if (node == nullptr) {
        return false;
    }
    auto it = node->attributes.find(key);
    if (it == node->attributes.end()) {
        return false;
    }
    it->second = values;
    return true;
}

std::vector<int64_t> RemapYoloHeadShape(const std::vector<int64_t>& shape) {
    if (shape.size() != 5) {
        return shape;
    }
    return {shape[0], shape[3], shape[4], shape[1], shape[2]};
}

std::vector<float> RemapResizeScales(const std::vector<float>& scales) {
    if (scales.size() != 4) {
        return scales;
    }
    return {scales[0], scales[2], scales[3], scales[1]};
}

const ValueDesc* FindValue(const ModelDesc& model, const std::string& name) {
    for (const auto& value : model.graph.values) {
        if (value.tensor.name == name) {
            return &value;
        }
    }
    return nullptr;
}

ValueDesc* FindValueMutable(ModelDesc* model, const std::string& name) {
    if (model == nullptr) {
        return nullptr;
    }
    for (auto& value : model->graph.values) {
        if (value.tensor.name == name) {
            return &value;
        }
    }
    return nullptr;
}

}  // namespace

bool ConvertModelToNhwcInPlace(ModelDesc* model) {
    if (model == nullptr) {
        return false;
    }

    for (auto& value : model->graph.values) {
        if (value.constant || value.tensor.dims.size() != 4) {
            continue;
        }
        value.tensor.dims = PermuteNchwToNhwc4D(value.tensor.dims);
        value.tensor.layout = DataLayout::NHWC;
    }

    for (auto& node : model->graph.nodes) {
        if ((node.op_type == "Concat" || node.op_type == "Split") && !node.inputs.empty()) {
            const auto* input_value = FindValue(*model, node.inputs[0]);
            if (input_value == nullptr) {
                continue;
            }
            auto it = node.attributes.find("axis");
            if (it == node.attributes.end()) {
                continue;
            }
            if (auto axis = std::get_if<int64_t>(&it->second); axis != nullptr) {
                RewriteIntAttribute(&node, "axis", RemapAxisNchwToNhwc(*axis, input_value->tensor.dims.size()));
            }
        } else if (node.op_type == "Resize") {
            auto it = node.attributes.find("scales");
            if (it == node.attributes.end()) {
                continue;
            }
            if (auto scales = std::get_if<std::vector<float>>(&it->second); scales != nullptr) {
                RewriteFloatVectorAttribute(&node, "scales", RemapResizeScales(*scales));
            }
        } else if (node.op_type == "Reshape") {
            auto it = node.attributes.find("shape");
            if (it == node.attributes.end()) {
                continue;
            }
            if (auto shape = std::get_if<std::vector<int64_t>>(&it->second); shape != nullptr && shape->size() == 5) {
                RewriteVectorIntAttribute(&node, "shape", RemapYoloHeadShape(*shape));
                if (!node.outputs.empty()) {
                    auto* output_value = FindValueMutable(model, node.outputs[0]);
                    if (output_value != nullptr) {
                        output_value->tensor.dims = RemapYoloHeadShape(output_value->tensor.dims);
                    }
                }
            }
        } else if (node.op_type == "Transpose") {
            auto it = node.attributes.find("perm");
            if (it == node.attributes.end()) {
                continue;
            }
            if (auto perm = std::get_if<std::vector<int64_t>>(&it->second); perm != nullptr &&
                *perm == std::vector<int64_t>({0, 1, 3, 4, 2})) {
                RewriteVectorIntAttribute(&node, "perm", {0, 3, 1, 2, 4});
            }
        }
    }

    return true;
}

}  // namespace model
}  // namespace feather
