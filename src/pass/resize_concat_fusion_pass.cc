#include "pass/resize_concat_fusion_pass.h"

#include <algorithm>
#include <string>
#include <vector>

#include "core/operator_registry.h"
#include "core/static_graph.h"

namespace feather {

namespace {

const std::string kResizeConcatFusionPassName = "ResizeConcatFusionPass";

bool IsGraphOutput(const StaticGraph& graph, const std::string& value_name) {
    return graph.IsGraphOutputValue(value_name);
}

const model::NodeDesc* FindModelNode(const StaticGraph& graph, const std::string& node_name) {
    for (const auto& node : graph.model().graph.nodes) {
        if (node.name == node_name) {
            return &node;
        }
    }
    return nullptr;
}

const std::vector<float>* FindFloatVectorAttribute(const model::NodeDesc& node, const std::string& key) {
    auto it = node.attributes.find(key);
    if (it == node.attributes.end()) {
        return nullptr;
    }
    return std::get_if<std::vector<float>>(&it->second);
}

bool MatchResizeConcatPattern(const StaticGraph& graph, const StaticNode& concat_node, model::NodeDesc* fused_desc,
                              std::string* resize_node_name) {
    if (fused_desc == nullptr || resize_node_name == nullptr || concat_node.removed || concat_node.op_type != "Concat" ||
        concat_node.inputs.size() != 2 || concat_node.outputs.size() != 1) {
        return false;
    }
    if (IsGraphOutput(graph, concat_node.inputs[0]) || IsGraphOutput(graph, concat_node.inputs[1])) {
        return false;
    }

    const auto* concat_model_node = FindModelNode(graph, concat_node.name);
    if (concat_model_node == nullptr) {
        return false;
    }
    auto axis_it = concat_model_node->attributes.find("axis");
    if (axis_it == concat_model_node->attributes.end()) {
        return false;
    }
    const auto* axis_value = std::get_if<int64_t>(&axis_it->second);
    if (axis_value == nullptr) {
        return false;
    }

    for (size_t resize_input_index = 0; resize_input_index < 2; ++resize_input_index) {
        const auto& resize_output_name = concat_node.inputs[resize_input_index];
        const auto& passthrough_input_name = concat_node.inputs[1 - resize_input_index];
        const auto resize_node_name_candidate = graph.GetProducer(resize_output_name);
        if (resize_node_name_candidate.empty()) {
            continue;
        }
        const auto* resize_node = graph.GetNode(resize_node_name_candidate);
        if (resize_node == nullptr || resize_node->op_type != "Resize" || resize_node->inputs.size() != 1 ||
            resize_node->outputs.size() != 1) {
            continue;
        }
        if (graph.GetUsers(resize_output_name) != std::vector<std::string>{concat_node.name}) {
            continue;
        }
        const auto* resize_model_node = FindModelNode(graph, resize_node->name);
        if (resize_model_node == nullptr) {
            continue;
        }
        const auto* scales = FindFloatVectorAttribute(*resize_model_node, "scales");
        if (scales == nullptr || scales->size() != 4) {
            continue;
        }

        const auto resize_input = graph.GetTensor(resize_node->inputs[0]);
        const auto passthrough_input = graph.GetTensor(passthrough_input_name);
        const auto concat_output = graph.GetTensor(concat_node.outputs[0]);
        if (resize_input == nullptr || passthrough_input == nullptr || concat_output == nullptr) {
            continue;
        }
        if (resize_input->dims().size() != 4 || passthrough_input->dims().size() != 4 || concat_output->dims().size() != 4) {
            continue;
        }
        const auto layout = NormalizeDataLayout(concat_output->layout());
        const int64_t normalized_axis =
            *axis_value < 0 ? *axis_value + static_cast<int64_t>(concat_output->dims().size()) : *axis_value;
        const int64_t channel_axis = layout == DataLayout::NHWC ? 3 : 1;
        if (normalized_axis != channel_axis) {
            continue;
        }

        fused_desc->name = concat_node.name;
        fused_desc->op_type = "ResizeConcat";
        fused_desc->domain = concat_model_node->domain;
        fused_desc->inputs = {resize_node->inputs[0], passthrough_input_name};
        fused_desc->outputs = concat_node.outputs;
        fused_desc->attributes["axis"] = *axis_value;
        fused_desc->attributes["resize_input_index"] = static_cast<int64_t>(resize_input_index);
        fused_desc->attributes["scales"] = *scales;
        *resize_node_name = resize_node->name;
        return true;
    }
    return false;
}

}  // namespace

const std::string& ResizeConcatFusionPass::name() const { return kResizeConcatFusionPassName; }

int32_t ResizeConcatFusionPass::Run(StaticGraph* graph) {
    if (graph == nullptr) {
        return -1;
    }
    if (graph->KernelDevice() != DeviceType::CUDA) {
        return 0;
    }

    std::vector<std::string> concat_node_names;
    for (const auto& node : graph->nodes()) {
        if (!node.removed && node.op_type == "Concat") {
            concat_node_names.push_back(node.name);
        }
    }

    for (const auto& concat_node_name : concat_node_names) {
        const auto* concat_node = graph->GetNode(concat_node_name);
        if (concat_node == nullptr) {
            continue;
        }
        model::NodeDesc fused_desc;
        std::string resize_node_name;
        if (!MatchResizeConcatPattern(*graph, *concat_node, &fused_desc, &resize_node_name)) {
            continue;
        }
        if (!graph->ReplaceNodeDesc(fused_desc)) {
            return -1;
        }
        if (!graph->RemoveNode(resize_node_name)) {
            return -1;
        }
    }
    return 0;
}

}  // namespace feather
