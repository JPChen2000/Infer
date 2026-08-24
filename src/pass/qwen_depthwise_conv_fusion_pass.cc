#include "pass/qwen_depthwise_conv_fusion_pass.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

#include "core/static_graph.h"
#include "model/model_format.h"

namespace feather {
namespace {

const std::string kQwenDepthwiseConvFusionPassName = "QwenDepthwiseConvFusionPass";

struct QwenDepthwiseConvPattern {
    std::string concat;
    std::string reshape;
    std::string conv;
    std::string split;
    std::string state;
    std::string mixed;
    std::string weight;
    std::string conv_out;
    std::string discarded_prefix;
    std::string next_state;
};

bool IsQwenModel(const StaticGraph& graph) {
    std::string model_name = graph.model().name + " " + graph.model().graph.name;
    std::transform(model_name.begin(), model_name.end(), model_name.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return model_name.find("qwen") != std::string::npos;
}

const model::NodeDesc* ModelNode(const StaticGraph& graph, const std::string& node_name) {
    for (const auto& node : graph.model().graph.nodes) {
        if (node.name == node_name) {
            return &node;
        }
    }
    return nullptr;
}

bool ReadIntAttribute(const StaticGraph& graph, const StaticNode& node, const std::string& key, int64_t* value) {
    const auto* model_node = ModelNode(graph, node.name);
    if (model_node == nullptr || value == nullptr) {
        return false;
    }
    const auto it = model_node->attributes.find(key);
    if (it == model_node->attributes.end()) {
        return false;
    }
    const auto* attribute = std::get_if<int64_t>(&it->second);
    if (attribute == nullptr) {
        return false;
    }
    *value = *attribute;
    return true;
}

bool HasIntAttributeOrDefault(const StaticGraph& graph, const StaticNode& node, const std::string& key,
                              int64_t expected) {
    int64_t value = expected;
    return !ReadIntAttribute(graph, node, key, &value) || value == expected;
}

bool HasVectorAttributeOrDefault(const StaticGraph& graph, const StaticNode& node, const std::string& key,
                                 const std::vector<int64_t>& expected) {
    const auto* model_node = ModelNode(graph, node.name);
    if (model_node == nullptr) {
        return false;
    }
    const auto it = model_node->attributes.find(key);
    if (it == model_node->attributes.end()) {
        return true;
    }
    const auto* value = std::get_if<std::vector<int64_t>>(&it->second);
    return value != nullptr && *value == expected;
}

bool HasShape(const std::shared_ptr<Tensor>& tensor, const std::vector<int64_t>& shape) {
    return tensor != nullptr && tensor->data_type() == DataType::BF16 && tensor->dims().data() == shape;
}

bool HasOnlyUsers(const StaticGraph& graph, const std::string& value, std::vector<std::string> expected) {
    auto users = graph.GetUsers(value);
    std::sort(users.begin(), users.end());
    std::sort(expected.begin(), expected.end());
    return users == expected;
}

bool IsConvNode(const StaticNode& node) { return node.op_type == "Conv" || node.op_type == "Conv2D"; }

bool HasQwenConvDefaults(const StaticGraph& graph, const StaticNode& conv, int64_t channels) {
    int64_t group = 1;
    if (!ReadIntAttribute(graph, conv, "group", &group) || group != channels) {
        return false;
    }
    return HasIntAttributeOrDefault(graph, conv, "stride_h", 1) &&
           HasIntAttributeOrDefault(graph, conv, "stride_w", 1) &&
           HasIntAttributeOrDefault(graph, conv, "pad_h", 0) && HasIntAttributeOrDefault(graph, conv, "pad_w", 0) &&
           HasIntAttributeOrDefault(graph, conv, "dilation_h", 1) &&
           HasIntAttributeOrDefault(graph, conv, "dilation_w", 1) &&
           HasVectorAttributeOrDefault(graph, conv, "strides", {1, 1}) &&
           HasVectorAttributeOrDefault(graph, conv, "pads", {0, 0, 0, 0}) &&
           HasVectorAttributeOrDefault(graph, conv, "dilations", {1, 1});
}

const StaticNode* ProducerNode(const StaticGraph& graph, const std::string& value_name,
                               const std::string& expected_op_type) {
    const auto producer_name = graph.GetProducer(value_name);
    const auto* producer = producer_name.empty() ? nullptr : graph.GetNode(producer_name);
    return producer != nullptr && !producer->removed && producer->op_type == expected_op_type ? producer : nullptr;
}

bool MatchQwenDepthwiseConvPattern(const StaticGraph& graph, const StaticNode& conv,
                                   QwenDepthwiseConvPattern* pattern) {
    if (pattern == nullptr || conv.removed || !IsConvNode(conv) || conv.inputs.size() != 2 ||
        conv.outputs.size() != 1) {
        return false;
    }

    const auto* reshape = ProducerNode(graph, conv.inputs[0], "Reshape");
    if (reshape == nullptr || reshape->inputs.size() != 2 || reshape->outputs.size() != 1 ||
        !HasOnlyUsers(graph, reshape->outputs[0], {conv.name})) {
        return false;
    }
    const auto* concat = ProducerNode(graph, reshape->inputs[0], "Concat");
    if (concat == nullptr || concat->inputs.size() != 2 || concat->outputs.size() != 1) {
        return false;
    }

    const auto concat_users = graph.GetUsers(concat->outputs[0]);
    if (concat_users.size() != 2) {
        return false;
    }
    const StaticNode* split = nullptr;
    for (const auto& user_name : concat_users) {
        if (user_name == reshape->name) {
            continue;
        }
        const auto* candidate = graph.GetNode(user_name);
        if (candidate == nullptr || candidate->op_type != "Split") {
            return false;
        }
        split = candidate;
    }
    if (split == nullptr || split->inputs.size() != 1 || split->inputs[0] != concat->outputs[0] ||
        split->outputs.size() != 2 || graph.IsGraphOutputValue(split->outputs[0]) ||
        !graph.GetUsers(split->outputs[0]).empty()) {
        return false;
    }

    int64_t concat_axis = 0;
    int64_t split_axis = 0;
    if (!ReadIntAttribute(graph, *concat, "axis", &concat_axis) || !ReadIntAttribute(graph, *split, "axis", &split_axis) ||
        concat_axis != 2 || split_axis != 2 ||
        !HasVectorAttributeOrDefault(graph, *split, "split_sizes", {1, 3})) {
        return false;
    }

    const auto state = graph.GetTensor(concat->inputs[0]);
    const auto mixed = graph.GetTensor(concat->inputs[1]);
    const auto joined = graph.GetTensor(concat->outputs[0]);
    const auto conv_input = graph.GetTensor(reshape->outputs[0]);
    const auto weight = graph.GetTensor(conv.inputs[1]);
    const auto conv_out = graph.GetTensor(conv.outputs[0]);
    const auto discarded_prefix = graph.GetTensor(split->outputs[0]);
    const auto next_state = graph.GetTensor(split->outputs[1]);
    if (state == nullptr || state->dims().size() != 3 || state->dims()[0] != 1 || state->dims()[1] <= 0 ||
        state->dims()[2] != 3) {
        return false;
    }
    const int64_t channels = state->dims()[1];
    if (!HasShape(state, {1, channels, 3}) || !HasShape(mixed, {1, channels, 1}) ||
        !HasShape(joined, {1, channels, 4}) || !HasShape(conv_input, {1, channels, 1, 4}) ||
        !HasShape(weight, {channels, 1, 1, 4}) || !HasShape(conv_out, {1, channels, 1, 1}) ||
        !HasShape(discarded_prefix, {1, channels, 1}) || !HasShape(next_state, {1, channels, 3}) ||
        !HasQwenConvDefaults(graph, conv, channels)) {
        return false;
    }

    pattern->concat = concat->name;
    pattern->reshape = reshape->name;
    pattern->conv = conv.name;
    pattern->split = split->name;
    pattern->state = concat->inputs[0];
    pattern->mixed = concat->inputs[1];
    pattern->weight = conv.inputs[1];
    pattern->conv_out = conv.outputs[0];
    pattern->discarded_prefix = split->outputs[0];
    pattern->next_state = split->outputs[1];
    return true;
}

}  // namespace

const std::string& QwenDepthwiseConvFusionPass::name() const { return kQwenDepthwiseConvFusionPassName; }

int32_t QwenDepthwiseConvFusionPass::Run(StaticGraph* graph) {
    if (graph == nullptr) {
        return -1;
    }
    if ((graph->KernelDevice() != DeviceType::X86 && graph->KernelDevice() != DeviceType::CUDA) ||
        !IsQwenModel(*graph)) {
        return 0;
    }

    std::vector<std::string> conv_nodes;
    for (const auto& node : graph->nodes()) {
        if (!node.removed && IsConvNode(node)) {
            conv_nodes.push_back(node.name);
        }
    }

    for (const auto& name : conv_nodes) {
        const auto* conv = graph->GetNode(name);
        if (conv == nullptr) {
            continue;
        }
        QwenDepthwiseConvPattern pattern;
        if (!MatchQwenDepthwiseConvPattern(*graph, *conv, &pattern)) {
            continue;
        }

        model::NodeDesc replacement;
        replacement.name = pattern.conv;
        replacement.op_type = "QwenDepthwiseConvState";
        replacement.inputs = {pattern.state, pattern.mixed, pattern.weight};
        replacement.outputs = {pattern.conv_out, pattern.discarded_prefix, pattern.next_state};
        if (!graph->ReplaceNodeDescAndAbsorbNode(replacement, pattern.split) || !graph->RemoveNode(pattern.reshape) ||
            !graph->RemoveNode(pattern.concat)) {
            return -1;
        }
    }
    return 0;
}

}  // namespace feather
