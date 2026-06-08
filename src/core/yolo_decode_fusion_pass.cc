#include "core/yolo_decode_fusion_pass.h"

#include <algorithm>
#include <vector>

#include "core/static_graph.h"
#include "model/model_format.h"

namespace feather {

namespace {

struct YoloDecodePattern {
    std::string final_reshape;
    std::string concat;
    std::string mul_xy_stride;
    std::string add_xy;
    std::string mul_xy;
    std::string mul_wh_anchor;
    std::string pow_wh;
    std::string mul_wh;
    std::string split;
    std::string sigmoid;
    std::string transpose;
    std::string raw_reshape;

    std::string raw_input;
    std::string xy_scale;
    std::string grid;
    std::string stride;
    std::string wh_scale;
    std::string anchor_grid;
};

bool IsActiveOp(const StaticNode* node, const std::string& op_type) {
    return node != nullptr && !node->removed && node->op_type == op_type;
}

const StaticNode* ProducerNode(const StaticGraph& graph, const std::string& value_name,
                               const std::string& op_type = std::string()) {
    const auto producer = graph.GetProducer(value_name);
    if (producer.empty()) {
        return nullptr;
    }
    const auto* node = graph.GetNode(producer);
    if (node == nullptr || node->removed) {
        return nullptr;
    }
    if (!op_type.empty() && node->op_type != op_type) {
        return nullptr;
    }
    return node;
}

bool HasIntAttribute(const model::NodeDesc& node, const std::string& key, int64_t expected) {
    auto it = node.attributes.find(key);
    if (it == node.attributes.end()) {
        return false;
    }
    if (auto value = std::get_if<int64_t>(&it->second); value != nullptr) {
        return *value == expected;
    }
    return false;
}

bool HasFloatAttribute(const model::NodeDesc& node, const std::string& key, float expected) {
    auto it = node.attributes.find(key);
    if (it == node.attributes.end()) {
        return false;
    }
    if (auto value = std::get_if<float>(&it->second); value != nullptr) {
        return *value == expected;
    }
    if (auto value = std::get_if<int64_t>(&it->second); value != nullptr) {
        return static_cast<float>(*value) == expected;
    }
    return false;
}

bool HasVectorAttribute(const model::NodeDesc& node, const std::string& key, const std::vector<int64_t>& expected) {
    auto it = node.attributes.find(key);
    if (it == node.attributes.end()) {
        return false;
    }
    if (auto value = std::get_if<std::vector<int64_t>>(&it->second); value != nullptr) {
        return *value == expected;
    }
    return false;
}

const model::NodeDesc* ModelNode(const StaticGraph& graph, const std::string& name) {
    for (const auto& node : graph.model().graph.nodes) {
        if (node.name == name) {
            return &node;
        }
    }
    return nullptr;
}

bool UsersAreExactly(const StaticGraph& graph, const std::string& value_name,
                     const std::vector<std::string>& expected_users) {
    auto users = graph.GetUsers(value_name);
    auto expected = expected_users;
    std::sort(users.begin(), users.end());
    std::sort(expected.begin(), expected.end());
    return users == expected;
}

bool TensorExists(const StaticGraph& graph, const std::string& value_name) {
    return graph.GetTensor(value_name) != nullptr;
}

std::string OtherInput(const StaticNode& node, const std::string& known_input) {
    if (node.inputs.size() != 2) {
        return {};
    }
    if (node.inputs[0] == known_input) {
        return node.inputs[1];
    }
    if (node.inputs[1] == known_input) {
        return node.inputs[0];
    }
    return {};
}

bool MatchYoloDecodePattern(const StaticGraph& graph, const StaticNode& final_reshape,
                            YoloDecodePattern* pattern) {
    if (pattern == nullptr || !IsActiveOp(&final_reshape, "Reshape") ||
        final_reshape.inputs.size() != 1 || final_reshape.outputs.size() != 1) {
        return false;
    }

    const auto* concat = ProducerNode(graph, final_reshape.inputs[0], "Concat");
    if (concat == nullptr || concat->inputs.size() != 3 || concat->outputs.size() != 1) {
        return false;
    }
    const auto* concat_desc = ModelNode(graph, concat->name);
    if (concat_desc == nullptr || !HasIntAttribute(*concat_desc, "axis", 4)) {
        return false;
    }

    const auto* mul_xy_stride = ProducerNode(graph, concat->inputs[0], "Mul");
    const auto* mul_wh_anchor = ProducerNode(graph, concat->inputs[1], "Mul");
    if (mul_xy_stride == nullptr || mul_wh_anchor == nullptr) {
        return false;
    }

    const StaticNode* add_xy = nullptr;
    for (const auto& input : mul_xy_stride->inputs) {
        if (const auto* producer = ProducerNode(graph, input, "Add"); producer != nullptr) {
            add_xy = producer;
            break;
        }
    }
    if (add_xy == nullptr || add_xy->outputs.size() != 1) {
        return false;
    }
    const std::string stride = OtherInput(*mul_xy_stride, add_xy->outputs[0]);
    if (stride.empty()) {
        return false;
    }

    const StaticNode* mul_xy = nullptr;
    for (const auto& input : add_xy->inputs) {
        if (const auto* producer = ProducerNode(graph, input, "Mul"); producer != nullptr) {
            mul_xy = producer;
            break;
        }
    }
    if (mul_xy == nullptr || mul_xy->outputs.size() != 1) {
        return false;
    }
    const std::string grid = OtherInput(*add_xy, mul_xy->outputs[0]);
    if (grid.empty()) {
        return false;
    }

    const StaticNode* pow_wh = nullptr;
    for (const auto& input : mul_wh_anchor->inputs) {
        if (const auto* producer = ProducerNode(graph, input, "Pow"); producer != nullptr) {
            pow_wh = producer;
            break;
        }
    }
    if (pow_wh == nullptr || pow_wh->inputs.size() != 1 || pow_wh->outputs.size() != 1) {
        return false;
    }
    const auto* pow_desc = ModelNode(graph, pow_wh->name);
    if (pow_desc == nullptr || !HasFloatAttribute(*pow_desc, "exponent", 2.0f)) {
        return false;
    }
    const std::string anchor_grid = OtherInput(*mul_wh_anchor, pow_wh->outputs[0]);
    if (anchor_grid.empty()) {
        return false;
    }

    const auto* mul_wh = ProducerNode(graph, pow_wh->inputs[0], "Mul");
    if (mul_wh == nullptr || mul_wh->outputs.size() != 1) {
        return false;
    }

    const auto* split = ProducerNode(graph, concat->inputs[2], "Split");
    if (split == nullptr || split->inputs.size() != 1 || split->outputs.size() != 3) {
        return false;
    }
    const auto* split_desc = ModelNode(graph, split->name);
    const auto split_rest_tensor = graph.GetTensor(split->outputs[2]);
    if (split_rest_tensor == nullptr || split_rest_tensor->dims().size() != 5) {
        return false;
    }
    if (split_desc == nullptr || !HasIntAttribute(*split_desc, "axis", 4) ||
        !HasVectorAttribute(*split_desc, "split_sizes", {2, 2, split_rest_tensor->dims()[4]})) {
        return false;
    }
    if (split->outputs[0].empty() || split->outputs[1].empty() || concat->inputs[2] != split->outputs[2]) {
        return false;
    }
    if (ProducerNode(graph, split->outputs[0], "Split") != split ||
        ProducerNode(graph, split->outputs[1], "Split") != split) {
        return false;
    }

    const std::string xy_scale = OtherInput(*mul_xy, split->outputs[0]);
    const std::string wh_scale = OtherInput(*mul_wh, split->outputs[1]);
    if (xy_scale.empty() || wh_scale.empty()) {
        return false;
    }

    const auto* sigmoid = ProducerNode(graph, split->inputs[0], "Sigmoid");
    if (sigmoid == nullptr || sigmoid->inputs.size() != 1) {
        return false;
    }
    const auto* transpose = ProducerNode(graph, sigmoid->inputs[0], "Transpose");
    if (transpose == nullptr || transpose->inputs.size() != 1) {
        return false;
    }
    const auto* transpose_desc = ModelNode(graph, transpose->name);
    if (transpose_desc == nullptr || !HasVectorAttribute(*transpose_desc, "perm", {0, 1, 3, 4, 2})) {
        return false;
    }
    const auto* raw_reshape = ProducerNode(graph, transpose->inputs[0], "Reshape");
    if (raw_reshape == nullptr || raw_reshape->inputs.size() != 1 || raw_reshape->outputs.size() != 1) {
        return false;
    }

    const std::vector<std::pair<std::string, std::vector<std::string>>> expected_users = {
        {raw_reshape->outputs[0], {transpose->name}},
        {transpose->outputs[0], {sigmoid->name}},
        {sigmoid->outputs[0], {split->name}},
        {split->outputs[0], {mul_xy->name}},
        {split->outputs[1], {mul_wh->name}},
        {split->outputs[2], {concat->name}},
        {mul_xy->outputs[0], {add_xy->name}},
        {add_xy->outputs[0], {mul_xy_stride->name}},
        {mul_xy_stride->outputs[0], {concat->name}},
        {mul_wh->outputs[0], {pow_wh->name}},
        {pow_wh->outputs[0], {mul_wh_anchor->name}},
        {mul_wh_anchor->outputs[0], {concat->name}},
        {concat->outputs[0], {final_reshape.name}},
    };
    for (const auto& item : expected_users) {
        if (!UsersAreExactly(graph, item.first, item.second)) {
            return false;
        }
    }

    const std::vector<std::string> required_tensors = {
        raw_reshape->inputs[0], xy_scale, grid, stride, wh_scale, anchor_grid, final_reshape.outputs[0],
    };
    for (const auto& value : required_tensors) {
        if (!TensorExists(graph, value)) {
            return false;
        }
    }

    pattern->final_reshape = final_reshape.name;
    pattern->concat = concat->name;
    pattern->mul_xy_stride = mul_xy_stride->name;
    pattern->add_xy = add_xy->name;
    pattern->mul_xy = mul_xy->name;
    pattern->mul_wh_anchor = mul_wh_anchor->name;
    pattern->pow_wh = pow_wh->name;
    pattern->mul_wh = mul_wh->name;
    pattern->split = split->name;
    pattern->sigmoid = sigmoid->name;
    pattern->transpose = transpose->name;
    pattern->raw_reshape = raw_reshape->name;
    pattern->raw_input = raw_reshape->inputs[0];
    pattern->xy_scale = xy_scale;
    pattern->grid = grid;
    pattern->stride = stride;
    pattern->wh_scale = wh_scale;
    pattern->anchor_grid = anchor_grid;
    return true;
}

bool ApplyPattern(StaticGraph* graph, const YoloDecodePattern& pattern) {
    if (graph == nullptr) {
        return false;
    }
    if (!graph->ReplaceNodeOp(pattern.final_reshape, "YoloDecode",
                              {pattern.raw_input, pattern.xy_scale, pattern.grid, pattern.stride,
                               pattern.wh_scale, pattern.anchor_grid})) {
        return false;
    }

    const std::vector<std::string> remove_order = {
        pattern.concat,     pattern.mul_xy_stride, pattern.mul_wh_anchor, pattern.add_xy,
        pattern.pow_wh,     pattern.mul_xy,        pattern.mul_wh,        pattern.split,
        pattern.sigmoid,    pattern.transpose,     pattern.raw_reshape,
    };
    for (const auto& node_name : remove_order) {
        if (!graph->RemoveNode(node_name)) {
            return false;
        }
    }
    return true;
}

}  // namespace

const std::string& YoloDecodeFusionPass::name() const {
    static const std::string kName = "YoloDecodeFusionPass";
    return kName;
}

int32_t YoloDecodeFusionPass::Run(StaticGraph* graph) {
    if (graph == nullptr) {
        return -1;
    }

    std::vector<YoloDecodePattern> patterns;
    for (const auto& node : graph->nodes()) {
        YoloDecodePattern pattern;
        if (MatchYoloDecodePattern(*graph, node, &pattern)) {
            patterns.push_back(std::move(pattern));
        }
    }

    for (const auto& pattern : patterns) {
        if (graph->GetNode(pattern.final_reshape) == nullptr) {
            continue;
        }
        if (!ApplyPattern(graph, pattern)) {
            return -1;
        }
    }
    return 0;
}

}  // namespace feather
