#include "pass/qwen_matmul_add_fusion_pass.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

#include "core/static_graph.h"
#include "core/tensor.h"

namespace feather {

namespace {

const std::string kQwenMatMulAddFusionPassName = "QwenMatMulAddFusionPass";

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

bool IsVectorBias(const Tensor* bias, int64_t n) {
    if (bias == nullptr || !bias->IsInitialized() || bias->dims().empty() ||
        bias->dims()[bias->dims().size() - 1] != n) {
        return false;
    }
    for (size_t index = 0; index + 1 < bias->dims().size(); ++index) {
        if (bias->dims()[index] != 1) {
            return false;
        }
    }
    return bias->numel() == n;
}

bool MatchQwenMatMulAddPattern(const StaticGraph& graph, const StaticNode& add_node,
                               std::string* matmul_node_name, std::string* a_name, std::string* b_name,
                               std::string* bias_name) {
    if (matmul_node_name == nullptr || a_name == nullptr || b_name == nullptr || bias_name == nullptr ||
        add_node.removed || add_node.op_type != "Add" || add_node.inputs.size() != 2 || add_node.outputs.size() != 1) {
        return false;
    }

    for (size_t matmul_input_index = 0; matmul_input_index < 2; ++matmul_input_index) {
        const auto& matmul_output = add_node.inputs[matmul_input_index];
        const auto& candidate_bias = add_node.inputs[1 - matmul_input_index];
        const auto producer_name = graph.GetProducer(matmul_output);
        if (producer_name.empty()) {
            continue;
        }
        const auto* matmul_node = graph.GetNode(producer_name);
        if (matmul_node == nullptr || matmul_node->op_type != "MatMul" || matmul_node->inputs.size() != 2 ||
            matmul_node->outputs.size() != 1 || IsGraphOutput(graph, matmul_output)) {
            continue;
        }
        const auto users = graph.GetUsers(matmul_output);
        if (users.size() != 1 || users[0] != add_node.name) {
            continue;
        }

        const auto a = graph.GetTensor(matmul_node->inputs[0]);
        const auto b = graph.GetTensor(matmul_node->inputs[1]);
        const auto bias = graph.GetTensor(candidate_bias);
        if (a == nullptr || b == nullptr || bias == nullptr || a->dims().size() < 2 || b->dims().size() != 2) {
            continue;
        }
        const int64_t k = a->dims()[a->dims().size() - 1];
        if (k <= 0 || k != b->dims()[0] || !IsVectorBias(bias.get(), b->dims()[1])) {
            continue;
        }

        *matmul_node_name = matmul_node->name;
        *a_name = matmul_node->inputs[0];
        *b_name = matmul_node->inputs[1];
        *bias_name = candidate_bias;
        return true;
    }
    return false;
}

}  // namespace

const std::string& QwenMatMulAddFusionPass::name() const { return kQwenMatMulAddFusionPassName; }

int32_t QwenMatMulAddFusionPass::Run(StaticGraph* graph) {
    if (graph == nullptr) {
        return -1;
    }
    if (!IsQwenModel(*graph)) {
        return 0;
    }

    std::vector<std::string> add_node_names;
    for (const auto& node : graph->nodes()) {
        if (!node.removed && node.op_type == "Add") {
            add_node_names.push_back(node.name);
        }
    }

    for (const auto& add_node_name : add_node_names) {
        const auto* add_node = graph->GetNode(add_node_name);
        if (add_node == nullptr) {
            continue;
        }

        std::string matmul_node_name;
        std::string a_name;
        std::string b_name;
        std::string bias_name;
        if (!MatchQwenMatMulAddPattern(*graph, *add_node, &matmul_node_name, &a_name, &b_name, &bias_name)) {
            continue;
        }
        if (!graph->ReplaceNodeOp(add_node_name, "Gemm", {a_name, b_name, bias_name}) ||
            !graph->RemoveNode(matmul_node_name)) {
            return -1;
        }
    }
    return 0;
}

}  // namespace feather
