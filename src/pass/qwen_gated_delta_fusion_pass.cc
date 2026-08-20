#include "pass/qwen_gated_delta_fusion_pass.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <utility>
#include <vector>

#include "core/static_graph.h"
#include "model/model_format.h"

namespace feather {

namespace {

struct GatedDeltaPattern {
    std::string state_update;
    std::string next_state;
    std::string state;
    std::string decay;
    std::string k;
    std::string v;
    std::string beta;
    std::string state_decay;
    std::string decay_unsqueezed;
    std::string decay_broadcast;
    std::string k_col;
    std::string kv_mem_full;
    std::string kv_mem;
    std::string v_delta;
    std::string beta_broadcast;
    std::string delta;
    std::string delta_row;
    std::string update;
    std::string output;
    std::string output_full;
    std::string q_col;
    std::string q;
};

const std::string kQwenGatedDeltaFusionPassName = "QwenGatedDeltaFusionPass";

const StaticNode* ProducerNode(const StaticGraph& graph, const std::string& value_name,
                               const std::string& op_type = std::string()) {
    const auto producer = graph.GetProducer(value_name);
    if (producer.empty()) {
        return nullptr;
    }
    const auto* node = graph.GetNode(producer);
    if (node == nullptr || node->removed || (!op_type.empty() && node->op_type != op_type)) {
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

bool HasVectorAttribute(const StaticGraph& graph, const StaticNode& node, const std::string& key,
                        const std::vector<int64_t>& expected) {
    const auto* model_node = ModelNode(graph, node.name);
    if (model_node == nullptr) {
        return false;
    }
    const auto it = model_node->attributes.find(key);
    if (it == model_node->attributes.end()) {
        return false;
    }
    const auto* value = std::get_if<std::vector<int64_t>>(&it->second);
    return value != nullptr && *value == expected;
}

bool HasIntAttribute(const StaticGraph& graph, const StaticNode& node, const std::string& key, int64_t expected) {
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

bool HasUsers(const StaticGraph& graph, const std::string& value_name,
              std::vector<std::string> expected_users) {
    auto users = graph.GetUsers(value_name);
    std::sort(users.begin(), users.end());
    std::sort(expected_users.begin(), expected_users.end());
    return users == expected_users;
}

bool IsQwenModel(const StaticGraph& graph) {
    std::string model_name = graph.model().name + " " + graph.model().graph.name;
    std::transform(model_name.begin(), model_name.end(), model_name.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return model_name.find("qwen") != std::string::npos;
}

bool IsGraphOutput(const StaticGraph& graph, const std::string& value_name) {
    const auto& outputs = graph.model().graph.outputs;
    return std::find(outputs.begin(), outputs.end(), value_name) != outputs.end();
}

bool IsFp32(const std::shared_ptr<Tensor>& tensor) {
    return tensor != nullptr && tensor->data_type() == DataType::FP32;
}

bool MatchStatePattern(const StaticGraph& graph, const StaticNode& add, GatedDeltaPattern* pattern) {
    if (pattern == nullptr || add.op_type != "Add" || add.inputs.size() != 2 || add.outputs.size() != 1) {
        return false;
    }

    const StaticNode* state_decay = nullptr;
    std::string state_name;
    std::string decay_unsqueezed_name;
    std::string decay_broadcast_name;
    std::string decay_name;
    for (const auto& input : add.inputs) {
        const auto* candidate = ProducerNode(graph, input, "Mul");
        if (candidate == nullptr || candidate->inputs.size() != 2 || candidate->outputs.size() != 1) {
            continue;
        }
        for (size_t index = 0; index < 2; ++index) {
            const auto& candidate_state_name = candidate->inputs[index];
            const auto& candidate_decay_broadcast_name = candidate->inputs[1 - index];
            const auto state_tensor = graph.GetTensor(candidate_state_name);
            const auto* decay_broadcast = ProducerNode(graph, candidate_decay_broadcast_name, "Unsqueeze");
            const auto* decay_unsqueezed = decay_broadcast == nullptr || decay_broadcast->inputs.size() != 1
                                               ? nullptr
                                               : ProducerNode(graph, decay_broadcast->inputs[0], "Unsqueeze");
            if (state_tensor != nullptr && state_tensor->dims().size() == 4 && state_tensor->dims()[0] == 1 &&
                IsFp32(state_tensor) && decay_broadcast != nullptr && decay_unsqueezed != nullptr &&
                HasVectorAttribute(graph, *decay_broadcast, "axes", {3}) &&
                HasVectorAttribute(graph, *decay_unsqueezed, "axes", {2}) &&
                decay_unsqueezed->inputs.size() == 1 && IsFp32(graph.GetTensor(decay_unsqueezed->inputs[0]))) {
                state_decay = candidate;
                state_name = candidate_state_name;
                decay_broadcast_name = candidate_decay_broadcast_name;
                decay_unsqueezed_name = decay_broadcast->inputs[0];
                decay_name = decay_unsqueezed->inputs[0];
                break;
            }
        }
    }
    if (state_decay == nullptr) {
        return false;
    }
    const auto state_tensor = graph.GetTensor(state_name);
    const auto* decay_broadcast = ProducerNode(graph, decay_broadcast_name, "Unsqueeze");
    const auto* decay_unsqueezed = ProducerNode(graph, decay_unsqueezed_name, "Unsqueeze");
    if (state_tensor == nullptr || state_tensor->dims().size() != 4 || state_tensor->dims()[0] != 1 ||
        !IsFp32(state_tensor) || decay_broadcast == nullptr || decay_unsqueezed == nullptr ||
        !HasUsers(graph, decay_broadcast_name, {state_decay->name}) ||
        !HasUsers(graph, decay_unsqueezed_name, {decay_broadcast->name})) {
        return false;
    }
    const auto decay_tensor = graph.GetTensor(decay_name);
    if (!IsFp32(decay_tensor) || decay_tensor->numel() != state_tensor->dims()[1]) {
        return false;
    }

    const auto* update = ProducerNode(graph, add.inputs[0] == state_decay->outputs[0] ? add.inputs[1] : add.inputs[0],
                                      "Mul");
    if (update == nullptr || update->inputs.size() != 2 || update->outputs.size() != 1) {
        return false;
    }

    const StaticNode* k_col = nullptr;
    const StaticNode* delta_row = nullptr;
    for (const auto& input : update->inputs) {
        const auto* producer = ProducerNode(graph, input, "Unsqueeze");
        if (producer == nullptr || producer->inputs.size() != 1) {
            continue;
        }
        if (HasVectorAttribute(graph, *producer, "axes", {3})) {
            k_col = producer;
        } else if (HasVectorAttribute(graph, *producer, "axes", {2})) {
            delta_row = producer;
        }
    }
    if (k_col == nullptr || delta_row == nullptr || k_col == delta_row ||
        !HasVectorAttribute(graph, *k_col, "axes", {3}) || !HasVectorAttribute(graph, *delta_row, "axes", {2}) ||
        !HasUsers(graph, delta_row->outputs[0], {update->name})) {
        return false;
    }

    const auto k_name = k_col->inputs[0];
    const auto k_tensor = graph.GetTensor(k_name);
    if (!IsFp32(k_tensor) || k_tensor->dims().size() != 3 || k_tensor->dims()[0] != 1 ||
        k_tensor->dims()[1] != state_tensor->dims()[1] || k_tensor->dims()[2] != state_tensor->dims()[2]) {
        return false;
    }

    const auto k_col_users = graph.GetUsers(k_col->outputs[0]);
    if (k_col_users.size() != 2) {
        return false;
    }
    std::string kv_mem_full_name;
    for (const auto& user_name : k_col_users) {
        if (user_name == update->name) {
            continue;
        }
        const auto* candidate = graph.GetNode(user_name);
        if (candidate == nullptr || candidate->op_type != "Mul" || candidate->inputs.size() != 2 ||
            candidate->outputs.size() != 1) {
            return false;
        }
        kv_mem_full_name = candidate->name;
    }
    const auto* kv_mem_full = graph.GetNode(kv_mem_full_name);
    if (kv_mem_full == nullptr ||
        !((kv_mem_full->inputs[0] == state_decay->outputs[0] && kv_mem_full->inputs[1] == k_col->outputs[0]) ||
          (kv_mem_full->inputs[1] == state_decay->outputs[0] && kv_mem_full->inputs[0] == k_col->outputs[0]))) {
        return false;
    }
    const auto kv_mem_users = graph.GetUsers(kv_mem_full->outputs[0]);
    if (kv_mem_users.size() != 1) {
        return false;
    }
    const auto* kv_mem = graph.GetNode(kv_mem_users.front());
    if (kv_mem == nullptr || kv_mem->op_type != "ReduceSum" || kv_mem->inputs.size() != 1 ||
        kv_mem->inputs[0] != kv_mem_full->outputs[0] ||
        !HasVectorAttribute(graph, *kv_mem, "axes", {2}) || !HasIntAttribute(graph, *kv_mem, "keepdims", 0) ||
        graph.GetUsers(kv_mem->outputs[0]).size() != 1) {
        return false;
    }
    const auto* v_delta = graph.GetNode(graph.GetUsers(kv_mem->outputs[0]).front());
    if (v_delta == nullptr || v_delta->op_type != "Sub" || v_delta->inputs.size() != 2 || v_delta->outputs.size() != 1) {
        return false;
    }
    // The fused state kernel implements v - (state_decay * k). Subtraction is
    // intentionally order-sensitive; accepting the reversed form would
    // silently change the recurrent state update.
    if (v_delta->inputs[1] != kv_mem->outputs[0]) {
        return false;
    }
    const std::string v_name = v_delta->inputs[0];
    const auto v_tensor = graph.GetTensor(v_name);
    if (!IsFp32(v_tensor) || v_tensor->dims().size() != 3 || v_tensor->dims()[0] != 1 ||
        v_tensor->dims()[1] != state_tensor->dims()[1] || v_tensor->dims()[2] != state_tensor->dims()[3]) {
        return false;
    }

    const auto* delta = ProducerNode(graph, delta_row->inputs[0], "Mul");
    if (delta == nullptr || delta->inputs.size() != 2 || !HasUsers(graph, delta->outputs[0], {delta_row->name})) {
        return false;
    }
    std::string beta_broadcast_name;
    if (delta->inputs[0] == v_delta->outputs[0]) {
        beta_broadcast_name = delta->inputs[1];
    } else if (delta->inputs[1] == v_delta->outputs[0]) {
        beta_broadcast_name = delta->inputs[0];
    } else {
        return false;
    }
    const auto* beta_broadcast = ProducerNode(graph, beta_broadcast_name, "Unsqueeze");
    if (beta_broadcast == nullptr || beta_broadcast->inputs.size() != 1 ||
        !HasVectorAttribute(graph, *beta_broadcast, "axes", {2}) ||
        !HasUsers(graph, beta_broadcast_name, {delta->name})) {
        return false;
    }
    const auto beta_name = beta_broadcast->inputs[0];
    const auto beta_tensor = graph.GetTensor(beta_name);
    if (!IsFp32(beta_tensor) || beta_tensor->numel() != state_tensor->dims()[1]) {
        return false;
    }
    if (!HasUsers(graph, kv_mem_full->outputs[0], {kv_mem->name}) ||
        !HasUsers(graph, update->outputs[0], {add.name}) ||
        !HasUsers(graph, state_decay->outputs[0], {kv_mem_full->name, add.name})) {
        return false;
    }

    pattern->state_update = add.name;
    pattern->next_state = add.outputs[0];
    pattern->state = state_name;
    pattern->decay = decay_name;
    pattern->k = k_name;
    pattern->v = v_name;
    pattern->beta = beta_name;
    pattern->state_decay = state_decay->name;
    pattern->decay_unsqueezed = decay_unsqueezed->name;
    pattern->decay_broadcast = decay_broadcast->name;
    pattern->k_col = k_col->name;
    pattern->kv_mem_full = kv_mem_full->name;
    pattern->kv_mem = kv_mem->name;
    pattern->v_delta = v_delta->name;
    pattern->beta_broadcast = beta_broadcast->name;
    pattern->delta = delta->name;
    pattern->delta_row = delta_row->name;
    pattern->update = update->name;
    return true;
}

bool MatchOutputPattern(const StaticGraph& graph, GatedDeltaPattern* pattern) {
    if (pattern == nullptr) {
        return false;
    }
    const auto* state_update = graph.GetNode(pattern->state_update);
    if (state_update == nullptr || state_update->outputs.size() != 1) {
        return false;
    }
    for (const auto& user_name : graph.GetUsers(pattern->next_state)) {
        const auto* output_full = graph.GetNode(user_name);
        if (output_full == nullptr || output_full->op_type != "Mul" || output_full->inputs.size() != 2 ||
            output_full->outputs.size() != 1) {
            continue;
        }
        std::string q_col_name;
        if (output_full->inputs[0] == pattern->next_state) {
            q_col_name = output_full->inputs[1];
        } else if (output_full->inputs[1] == pattern->next_state) {
            q_col_name = output_full->inputs[0];
        } else {
            continue;
        }
        const auto* q_col = ProducerNode(graph, q_col_name, "Unsqueeze");
        if (q_col == nullptr || q_col->inputs.size() != 1 ||
            !HasVectorAttribute(graph, *q_col, "axes", {3}) ||
            !HasUsers(graph, q_col_name, {output_full->name}) ||
            graph.GetUsers(output_full->outputs[0]).size() != 1) {
            continue;
        }
        const auto* output = graph.GetNode(graph.GetUsers(output_full->outputs[0]).front());
        if (output == nullptr || output->op_type != "ReduceSum" || output->inputs.size() != 1 ||
            output->inputs[0] != output_full->outputs[0] ||
            !HasVectorAttribute(graph, *output, "axes", {2}) || !HasIntAttribute(graph, *output, "keepdims", 0)) {
            continue;
        }
        const auto q_tensor = graph.GetTensor(q_col->inputs[0]);
        const auto state_tensor = graph.GetTensor(pattern->state);
        if (!IsFp32(q_tensor) || !IsFp32(state_tensor) || q_tensor->dims().size() != 3 ||
            state_tensor->dims().size() != 4 || q_tensor->dims()[0] != 1 || q_tensor->dims()[1] != state_tensor->dims()[1] ||
            q_tensor->dims()[2] != state_tensor->dims()[2]) {
            continue;
        }
        pattern->output = output->name;
        pattern->output_full = output_full->name;
        pattern->q_col = q_col->name;
        pattern->q = q_col->inputs[0];
        return true;
    }
    return false;
}

bool RemoveMatchedNodes(StaticGraph* graph, const GatedDeltaPattern& pattern) {
    const std::vector<std::string> nodes = {
        pattern.output_full, pattern.q_col, pattern.update, pattern.delta_row, pattern.delta,
        pattern.v_delta, pattern.kv_mem, pattern.kv_mem_full, pattern.k_col, pattern.beta_broadcast,
        pattern.state_decay, pattern.decay_broadcast, pattern.decay_unsqueezed,
    };
    for (const auto& node_name : nodes) {
        if (graph->GetNode(node_name) != nullptr && !graph->RemoveNode(node_name)) {
            return false;
        }
    }
    return true;
}

bool CanRemoveMatchedNodes(const StaticGraph& graph, const GatedDeltaPattern& pattern) {
    const std::vector<std::string> nodes = {
        pattern.output_full, pattern.q_col, pattern.update, pattern.delta_row, pattern.delta,
        pattern.v_delta, pattern.kv_mem, pattern.kv_mem_full, pattern.k_col, pattern.beta_broadcast,
        pattern.state_decay, pattern.decay_broadcast, pattern.decay_unsqueezed,
    };
    for (const auto& node_name : nodes) {
        const auto* node = graph.GetNode(node_name);
        if (node == nullptr) {
            return false;
        }
        for (const auto& output_name : node->outputs) {
            if (IsGraphOutput(graph, output_name)) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace

const std::string& QwenGatedDeltaFusionPass::name() const { return kQwenGatedDeltaFusionPassName; }

int32_t QwenGatedDeltaFusionPass::Run(StaticGraph* graph) {
    if (graph == nullptr) {
        return -1;
    }
    if (graph->KernelDevice() != DeviceType::X86 || !IsQwenModel(*graph)) {
        return 0;
    }

    std::vector<GatedDeltaPattern> patterns;
    for (const auto& node : graph->nodes()) {
        if (node.removed || node.op_type != "Add") {
            continue;
        }
        GatedDeltaPattern pattern;
        if (MatchStatePattern(*graph, node, &pattern) && MatchOutputPattern(*graph, &pattern)) {
            patterns.push_back(std::move(pattern));
        }
    }

    for (const auto& pattern : patterns) {
        const auto* state_node = graph->GetNode(pattern.state_update);
        if (state_node == nullptr || !CanRemoveMatchedNodes(*graph, pattern)) {
            continue;
        }
        if (!graph->ReplaceNodeOp(pattern.state_update, "QwenGatedDeltaState",
                                  {pattern.state, pattern.k, pattern.v, pattern.beta, pattern.decay}) ||
            !graph->ReplaceNodeOp(pattern.output, "QwenGatedDeltaOutput", {pattern.next_state, pattern.q}) ||
            !RemoveMatchedNodes(graph, pattern)) {
            return -1;
        }
    }
    return 0;
}

}  // namespace feather
