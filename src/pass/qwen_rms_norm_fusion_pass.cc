#include "pass/qwen_rms_norm_fusion_pass.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "core/static_graph.h"
#include "core/tensor.h"
#include "model/model_format.h"
#include "util/bf16.h"

namespace feather {
namespace {

const std::string kQwenRmsNormFusionPassName = "QwenRmsNormFusionPass";

struct RmsNormPattern {
    std::string terminal;
    std::string input;
    std::string weight;
    std::string epsilon;
    float weight_offset{0.0f};
    std::vector<std::string> removable;
};

bool IsQwenModel(const StaticGraph& graph) {
    std::string model_name = graph.model().name + " " + graph.model().graph.name;
    std::transform(model_name.begin(), model_name.end(), model_name.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return model_name.find("qwen") != std::string::npos;
}

bool IsGraphOutput(const StaticGraph& graph, const std::string& value_name) {
    return graph.IsGraphOutputValue(value_name);
}

const StaticNode* Producer(const StaticGraph& graph, const std::string& value_name,
                           const std::string& op_type = std::string()) {
    const auto producer_name = graph.GetProducer(value_name);
    if (producer_name.empty()) {
        return nullptr;
    }
    const auto* node = graph.GetNode(producer_name);
    if (node == nullptr || (!op_type.empty() && node->op_type != op_type)) {
        return nullptr;
    }
    return node;
}

const model::NodeDesc* ModelNode(const StaticGraph& graph, const std::string& node_name) {
    for (const auto& node : graph.model().graph.nodes) {
        if (node.name == node_name) {
            return &node;
        }
    }
    return nullptr;
}

bool IntAttribute(const StaticGraph& graph, const StaticNode& node, const std::string& key, int64_t expected) {
    const auto* model_node = ModelNode(graph, node.name);
    if (model_node == nullptr) {
        return false;
    }
    const auto it = model_node->attributes.find(key);
    if (it == model_node->attributes.end()) {
        return false;
    }
    const auto* value = std::get_if<int64_t>(&it->second);
    return value != nullptr && *value == expected;
}

bool AxesAttribute(const StaticGraph& graph, const StaticNode& node, const std::vector<int64_t>& expected) {
    const auto* model_node = ModelNode(graph, node.name);
    if (model_node == nullptr) {
        return false;
    }
    const auto it = model_node->attributes.find("axes");
    if (it == model_node->attributes.end()) {
        return false;
    }
    const auto* value = std::get_if<std::vector<int64_t>>(&it->second);
    return value != nullptr && *value == expected;
}

bool CastTo(const StaticGraph& graph, const StaticNode& node, int64_t to) {
    return node.op_type == "Cast" && node.inputs.size() == 1 && node.outputs.size() == 1 &&
           IntAttribute(graph, node, "to", to);
}

bool IsFloating(const std::shared_ptr<Tensor>& tensor) {
    return tensor != nullptr && (tensor->data_type() == DataType::FP32 || tensor->data_type() == DataType::BF16);
}

bool IsScalar(const Tensor* tensor) {
    return tensor != nullptr && tensor->IsInitialized() && tensor->numel() == 1 &&
           (tensor->data_type() == DataType::FP32 || tensor->data_type() == DataType::BF16);
}

bool IsOne(const Tensor* tensor) {
    if (!IsScalar(tensor)) {
        return false;
    }
    float value = 0.0f;
    if (tensor->data_type() == DataType::FP32) {
        value = tensor->data<float>()[0];
    } else {
        value = BFloat16ToFloat(tensor->data<BFloat16>()[0].bits);
    }
    return std::isfinite(value) && std::fabs(value - 1.0f) <= 1.0e-6f;
}

bool IsSingleUser(const StaticGraph& graph, const std::string& value_name, const std::string& node_name) {
    const auto users = graph.GetUsers(value_name);
    return users.size() == 1 && users.front() == node_name;
}

bool HasOnlyUsers(const StaticGraph& graph, const std::string& value_name, std::vector<std::string> expected) {
    auto users = graph.GetUsers(value_name);
    std::sort(users.begin(), users.end());
    std::sort(expected.begin(), expected.end());
    return users == expected;
}

bool IsSameShape(const Tensor* lhs, const Tensor* rhs) {
    return lhs != nullptr && rhs != nullptr && lhs->dims() == rhs->dims();
}

bool MatchBinary(const StaticNode* node, const std::string& op_type) {
    return node != nullptr && node->op_type == op_type && node->inputs.size() == 2 && node->outputs.size() == 1;
}

bool MatchRmsNorm(const StaticGraph& graph, const StaticNode& terminal, RmsNormPattern* pattern) {
    if (pattern == nullptr || terminal.removed || terminal.outputs.size() != 1 ||
        (terminal.op_type != "Cast" && terminal.op_type != "Mul")) {
        return false;
    }

    std::string scaled_name;
    std::string input_name;
    std::string weight_name;
    std::string epsilon_name;
    std::vector<std::string> removable;

    const Tensor* output_tensor = graph.GetTensor(terminal.outputs[0]).get();
    if (output_tensor == nullptr || !IsFloating(graph.GetTensor(terminal.outputs[0]))) {
        return false;
    }

    if (terminal.op_type == "Cast") {
        if (!CastTo(graph, terminal, 16)) {
            return false;
        }
        scaled_name = terminal.inputs[0];
    } else {
        if (terminal.inputs.size() != 2) {
            return false;
        }
        scaled_name = terminal.outputs[0];
    }
    const auto* scale_mul = terminal.op_type == "Mul" ? &terminal : Producer(graph, scaled_name, "Mul");
    if (!MatchBinary(scale_mul, "Mul")) {
        return false;
    }
    const auto* normalized = Producer(graph, scale_mul->inputs[0], "Div");
    const auto* scale = Producer(graph, scale_mul->inputs[1], "Add");
    if (normalized == nullptr || scale == nullptr) {
        // The two Mul inputs are commutative. Retry with the opposite order.
        normalized = Producer(graph, scale_mul->inputs[1], "Div");
        scale = Producer(graph, scale_mul->inputs[0], "Add");
    }
    if (!MatchBinary(normalized, "Div") || !MatchBinary(scale, "Add")) {
        return false;
    }
    if (!IsSingleUser(graph, normalized->outputs[0], scale_mul->name) ||
        !IsSingleUser(graph, scale->outputs[0], scale_mul->name)) {
        return false;
    }

    const auto* root = Producer(graph, normalized->inputs[1], "Sqrt");
    if (root == nullptr || !IsSingleUser(graph, root->outputs[0], normalized->name)) {
        return false;
    }
    const auto* mean_eps = Producer(graph, root->inputs[0], "Add");
    if (!MatchBinary(mean_eps, "Add") || !IsSingleUser(graph, mean_eps->outputs[0], root->name)) {
        return false;
    }
    const StaticNode* mean = nullptr;
    for (const auto& input : mean_eps->inputs) {
        const auto* candidate = Producer(graph, input, "ReduceMean");
        if (candidate != nullptr) {
            mean = candidate;
            epsilon_name = mean_eps->inputs[0] == input ? mean_eps->inputs[1] : mean_eps->inputs[0];
            break;
        }
    }
    if (mean == nullptr || !AxesAttribute(graph, *mean, {-1}) || !IntAttribute(graph, *mean, "keepdims", 1) ||
        !IsScalar(graph.GetTensor(epsilon_name).get()) || !IsSingleUser(graph, mean->outputs[0], mean_eps->name)) {
        return false;
    }

    const auto* input_cast = Producer(graph, normalized->inputs[0], "Cast");
    const std::string normalized_input_name = normalized->inputs[0];
    if (input_cast != nullptr && !CastTo(graph, *input_cast, 1)) {
        return false;
    }
    input_name = input_cast == nullptr ? normalized_input_name : input_cast->inputs[0];
    const auto* square = Producer(graph, mean->inputs[0], "Mul");
    if (!MatchBinary(square, "Mul") || square->inputs[0] != normalized_input_name ||
        square->inputs[1] != normalized_input_name || !IsSingleUser(graph, square->outputs[0], mean->name) ||
        (input_cast != nullptr && !HasOnlyUsers(graph, input_cast->outputs[0], {normalized->name, square->name}))) {
        return false;
    }

    const auto input_tensor = graph.GetTensor(input_name);
    const StaticNode* weight_cast = nullptr;
    std::string one_name;
    for (const auto& scale_input : scale->inputs) {
        const auto* candidate_cast = Producer(graph, scale_input, "Cast");
        if (candidate_cast != nullptr && CastTo(graph, *candidate_cast, 1) &&
            IsSingleUser(graph, candidate_cast->outputs[0], scale->name)) {
            if (weight_cast != nullptr) {
                return false;
            }
            weight_cast = candidate_cast;
            weight_name = candidate_cast->inputs[0];
            continue;
        }
        if (IsOne(graph.GetTensor(scale_input).get())) {
            if (!one_name.empty()) {
                return false;
            }
            one_name = scale_input;
        }
    }
    const auto weight_tensor = weight_name.empty() ? std::shared_ptr<Tensor>() : graph.GetTensor(weight_name);
    if (input_tensor == nullptr || weight_tensor == nullptr || !IsFloating(input_tensor) || weight_cast == nullptr ||
        one_name.empty() || weight_tensor->dims().size() != 1 ||
        weight_tensor->dims()[0] != input_tensor->dims()[input_tensor->dims().size() - 1] ||
        !IsSameShape(graph.GetTensor(terminal.outputs[0]).get(), input_tensor.get())) {
        return false;
    }

    removable = {terminal.op_type == "Cast" ? scale_mul->name : "", scale->name, weight_cast->name,
                 normalized->name, root->name, mean_eps->name, mean->name, square->name,
                 input_cast == nullptr ? std::string() : input_cast->name};
    removable.erase(std::remove(removable.begin(), removable.end(), std::string()), removable.end());
    pattern->terminal = terminal.name;
    pattern->input = input_name;
    pattern->weight = weight_name;
    pattern->epsilon = epsilon_name;
    pattern->weight_offset = 1.0f;
    pattern->removable = std::move(removable);
    return true;
}

bool MatchGatedRmsNorm(const StaticGraph& graph, const StaticNode& terminal, RmsNormPattern* pattern) {
    if (pattern == nullptr || terminal.removed || terminal.op_type != "Mul" || !MatchBinary(&terminal, "Mul")) {
        return false;
    }
    const auto* normalized = Producer(graph, terminal.inputs[0], "Div");
    if (normalized == nullptr) {
        normalized = Producer(graph, terminal.inputs[1], "Div");
    }
    if (normalized == nullptr) {
        return false;
    }
    const std::string normalized_name = normalized->outputs[0];
    const std::string weight_name = terminal.inputs[0] == normalized_name ? terminal.inputs[1] : terminal.inputs[0];
    const auto weight = graph.GetTensor(weight_name);
    if (weight == nullptr || weight->dims().size() != 1 || !IsFloating(weight)) {
        return false;
    }
    const auto* root = Producer(graph, normalized->inputs[1], "Sqrt");
    const auto* input_cast = Producer(graph, normalized->inputs[0], "Cast");
    if (root == nullptr || input_cast == nullptr || !CastTo(graph, *input_cast, 1)) {
        return false;
    }
    const auto* mean_eps = Producer(graph, root->inputs[0], "Add");
    if (!MatchBinary(mean_eps, "Add")) {
        return false;
    }
    const auto* mean = Producer(graph, mean_eps->inputs[0], "ReduceMean");
    std::string epsilon_name = mean_eps->inputs[1];
    if (mean == nullptr) {
        mean = Producer(graph, mean_eps->inputs[1], "ReduceMean");
        epsilon_name = mean_eps->inputs[0];
    }
    const auto* square = mean == nullptr ? nullptr : Producer(graph, mean->inputs[0], "Mul");
    if (mean == nullptr || square == nullptr || !AxesAttribute(graph, *mean, {-1}) ||
        !IntAttribute(graph, *mean, "keepdims", 1) || !IsScalar(graph.GetTensor(epsilon_name).get()) ||
        square->inputs[0] != input_cast->outputs[0] || square->inputs[1] != input_cast->outputs[0] ||
        !HasOnlyUsers(graph, input_cast->outputs[0], {normalized->name, square->name})) {
        return false;
    }
    const auto input = graph.GetTensor(input_cast->inputs[0]);
    const auto output = graph.GetTensor(terminal.outputs[0]);
    if (input == nullptr || output == nullptr || !IsFloating(input) || !IsFloating(output) ||
        input->dims() != output->dims() || weight->dims()[0] != input->dims()[input->dims().size() - 1]) {
        return false;
    }
    const std::vector<std::string> nodes = {terminal.name, normalized->name, root->name, mean_eps->name, mean->name,
                                            square->name, input_cast->name};
    for (const auto& node_name : nodes) {
        const auto* node = graph.GetNode(node_name);
        if (node == nullptr) {
            return false;
        }
        for (const auto& output_name : node->outputs) {
            if (IsGraphOutput(graph, output_name) && node_name != terminal.name) {
                return false;
            }
        }
    }
    for (const auto& node_name : nodes) {
        const auto* node = graph.GetNode(node_name);
        if (node_name == terminal.name) {
            continue;
        }
        if (node_name == input_cast->name) {
            if (!HasOnlyUsers(graph, input_cast->outputs[0], {normalized->name, square->name})) {
                return false;
            }
            continue;
        }
        if (node == nullptr || (node->outputs.size() == 1 && graph.GetUsers(node->outputs[0]).size() != 1)) {
            return false;
        }
    }
    pattern->terminal = terminal.name;
    pattern->input = input_cast->inputs[0];
    pattern->weight = weight_name;
    pattern->epsilon = epsilon_name;
    pattern->weight_offset = 0.0f;
    pattern->removable = {normalized->name, root->name, mean_eps->name, mean->name, square->name, input_cast->name};
    return true;
}

bool RemoveNodes(StaticGraph* graph, const RmsNormPattern& pattern) {
    for (const auto& node_name : pattern.removable) {
        if (graph->GetNode(node_name) != nullptr && !graph->RemoveNode(node_name)) {
            return false;
        }
    }
    return true;
}

bool ReplaceWithQwenRmsNorm(StaticGraph* graph, const RmsNormPattern& pattern) {
    if (graph == nullptr) {
        return false;
    }
    const auto* terminal = graph->GetNode(pattern.terminal);
    if (terminal == nullptr) {
        return false;
    }
    model::NodeDesc desc;
    desc.name = terminal->name;
    desc.op_type = "QwenRmsNorm";
    desc.inputs = {pattern.input, pattern.weight, pattern.epsilon};
    desc.outputs = terminal->outputs;
    if (pattern.weight_offset != 0.0f) {
        desc.attributes["weight_offset"] = pattern.weight_offset;
    }
    return graph->ReplaceNodeDesc(desc);
}

}  // namespace

const std::string& QwenRmsNormFusionPass::name() const { return kQwenRmsNormFusionPassName; }

int32_t QwenRmsNormFusionPass::Run(StaticGraph* graph) {
    if (graph == nullptr) {
        return -1;
    }
    if ((graph->KernelDevice() != DeviceType::X86 && graph->KernelDevice() != DeviceType::CUDA) ||
        !IsQwenModel(*graph)) {
        return 0;
    }

    std::vector<RmsNormPattern> patterns;
    for (const auto& node : graph->nodes()) {
        if (node.removed) {
            continue;
        }
        RmsNormPattern pattern;
        const bool standard_match = MatchRmsNorm(*graph, node, &pattern);
        const bool gated_match = !standard_match && MatchGatedRmsNorm(*graph, node, &pattern);
        if (standard_match || gated_match) {
            bool removable = true;
            for (const auto& node_name : pattern.removable) {
                const auto* candidate = graph->GetNode(node_name);
                if (candidate == nullptr) {
                    removable = false;
                    break;
                }
                for (const auto& output_name : candidate->outputs) {
                    if (IsGraphOutput(*graph, output_name)) {
                        removable = false;
                    }
                }
            }
            if (removable) {
                patterns.push_back(std::move(pattern));
            }
        }
    }

    for (const auto& pattern : patterns) {
        if (graph->GetNode(pattern.terminal) == nullptr) {
            continue;
        }
        if (!ReplaceWithQwenRmsNorm(graph, pattern) || !RemoveNodes(graph, pattern)) {
            return -1;
        }
    }
    return 0;
}

}  // namespace feather
