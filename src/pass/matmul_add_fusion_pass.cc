#include "pass/matmul_add_fusion_pass.h"

#include <algorithm>
#include <string>
#include <vector>

#include "core/static_graph.h"
#include "core/tensor.h"

namespace feather {

namespace {

const std::string kMatMulAddFusionPassName = "MatMulAddFusionPass";

bool IsGraphOutput(const StaticGraph& graph, const std::string& value_name) {
    const auto& outputs = graph.model().graph.outputs;
    return std::find(outputs.begin(), outputs.end(), value_name) != outputs.end();
}

bool IsValidGemmBias(const Tensor* bias, int64_t m, int64_t n) {
    if (bias == nullptr || !bias->IsInitialized()) {
        return false;
    }
    if (bias->dims().size() == 1) {
        return bias->dims()[0] == n;
    }
    if (bias->dims().size() == 2) {
        return bias->dims()[0] == m && bias->dims()[1] == n;
    }
    return false;
}

bool MatchMatMulAddPattern(const StaticGraph& graph, const StaticNode& add_node, std::string* matmul_node_name,
                           std::string* a_name, std::string* b_name, std::string* bias_name) {
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
            matmul_node->outputs.size() != 1) {
            continue;
        }
        if (IsGraphOutput(graph, matmul_output)) {
            continue;
        }
        const auto users = graph.GetUsers(matmul_output);
        if (users.size() != 1 || users[0] != add_node.name) {
            continue;
        }
        const auto a = graph.GetTensor(matmul_node->inputs[0]);
        const auto b = graph.GetTensor(matmul_node->inputs[1]);
        const auto bias = graph.GetTensor(candidate_bias);
        if (a == nullptr || b == nullptr || bias == nullptr || a->dims().size() != 2 || b->dims().size() != 2) {
            continue;
        }
        if (a->dims()[1] != b->dims()[0]) {
            continue;
        }
        if (!IsValidGemmBias(bias.get(), a->dims()[0], b->dims()[1])) {
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

const std::string& MatMulAddFusionPass::name() const { return kMatMulAddFusionPassName; }

int32_t MatMulAddFusionPass::Run(StaticGraph* graph) {
    if (graph == nullptr) {
        return -1;
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
        if (!MatchMatMulAddPattern(*graph, *add_node, &matmul_node_name, &a_name, &b_name, &bias_name)) {
            continue;
        }
        if (!graph->ReplaceNodeOp(add_node_name, "Gemm", {a_name, b_name, bias_name})) {
            return -1;
        }
        if (!graph->RemoveNode(matmul_node_name)) {
            return -1;
        }
    }

    return 0;
}

}  // namespace feather
