#include "pass/qwen_gemm_argmax_fusion_pass.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>

#include "core/static_graph.h"
#include "util/types.h"

namespace feather {
namespace {

const std::string kPassName = "QwenGemmArgmaxFusionPass";

bool IsQwenModel(const StaticGraph& graph) {
    std::string name = graph.model().name + " " + graph.model().graph.name;
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return name.find("qwen") != std::string::npos;
}

const model::NodeDesc* FindModelNode(const StaticGraph& graph, const std::string& name) {
    for (const auto& node : graph.model().graph.nodes) {
        if (node.name == name) {
            return &node;
        }
    }
    return nullptr;
}

bool IsIntAttribute(const model::NodeDesc& node, const std::string& key, int64_t expected, bool absent_is_default) {
    const auto it = node.attributes.find(key);
    if (it == node.attributes.end()) {
        return absent_is_default;
    }
    const auto* value = std::get_if<int64_t>(&it->second);
    return value != nullptr && *value == expected;
}

bool IsFloatAttribute(const model::NodeDesc& node, const std::string& key, float expected, bool absent_is_default) {
    const auto it = node.attributes.find(key);
    if (it == node.attributes.end()) {
        return absent_is_default;
    }
    const auto* value = std::get_if<float>(&it->second);
    return value != nullptr && std::fabs(*value - expected) <= 1.0e-6f;
}

bool IsFinalQwenLogitsGemm(const StaticGraph& graph, const StaticNode& node) {
    if (node.removed || node.op_type != "Gemm" || node.inputs.size() != 2 || node.outputs.size() != 1 ||
        node.outputs[0] != "logits" || !graph.IsGraphOutputValue("logits") || !graph.GetUsers("logits").empty()) {
        return false;
    }
    const auto* model_node = FindModelNode(graph, node.name);
    if (model_node == nullptr || !IsIntAttribute(*model_node, "transA", 0, true) ||
        !IsIntAttribute(*model_node, "transB", 1, false) || !IsFloatAttribute(*model_node, "alpha", 1.0f, true) ||
        !IsFloatAttribute(*model_node, "beta", 1.0f, true)) {
        return false;
    }
    const auto a = graph.GetTensor(node.inputs[0]);
    const auto b = graph.GetTensor(node.inputs[1]);
    const auto out = graph.GetTensor(node.outputs[0]);
    if (a == nullptr || b == nullptr || out == nullptr || a->data_type() != DataType::BF16 ||
        b->data_type() != DataType::BF16 || a->dims().size() < 2 || b->dims().size() != 2 ||
        a->dims()[a->dims().size() - 1] != b->dims()[1] ||
        a->numel() != a->dims()[a->dims().size() - 1] || b->dims()[0] <= 0 || !b->is_immutable()) {
        return false;
    }
    return true;
}

bool IsFp8Type(DataType dtype) { return dtype == DataType::FP8E4M3 || dtype == DataType::FP8E5M2; }

bool IsFp8LogitsCast(const StaticGraph& graph, const StaticNode& cast, const StaticNode** gemm,
                     float* output_scale) {
    if (gemm == nullptr || output_scale == nullptr || cast.removed || cast.op_type != "Cast" ||
        cast.inputs.size() != 1 || cast.outputs.size() != 1 || cast.outputs[0] != "logits" ||
        !graph.IsGraphOutputValue("logits") || !graph.GetUsers("logits").empty()) {
        return false;
    }
    const auto* cast_model_node = FindModelNode(graph, cast.name);
    if (cast_model_node == nullptr || !IsIntAttribute(*cast_model_node, "to", 16, false)) {
        return false;
    }
    const auto producer_name = graph.GetProducer(cast.inputs[0]);
    const auto* producer = producer_name.empty() ? nullptr : graph.GetNode(producer_name);
    const auto projected_users = graph.GetUsers(cast.inputs[0]);
    if (producer == nullptr || producer->op_type != "Gemm" || producer->inputs.size() != 2 ||
        producer->outputs.size() != 1 || producer->outputs[0] != cast.inputs[0] || projected_users.size() != 1 ||
        projected_users[0] != cast.name) {
        return false;
    }
    const auto* model_node = FindModelNode(graph, producer->name);
    if (model_node == nullptr || !IsIntAttribute(*model_node, "transA", 0, true) ||
        !IsIntAttribute(*model_node, "transB", 1, false) ||
        !IsFloatAttribute(*model_node, "alpha", 1.0f, true) ||
        !IsFloatAttribute(*model_node, "beta", 1.0f, true)) {
        return false;
    }
    const auto a = graph.GetTensor(producer->inputs[0]);
    const auto b = graph.GetTensor(producer->inputs[1]);
    const auto projected = graph.GetTensor(producer->outputs[0]);
    const auto logits = graph.GetTensor(cast.outputs[0]);
    if (a == nullptr || b == nullptr || projected == nullptr || logits == nullptr || !a->IsInitialized() ||
        !b->IsInitialized() || !projected->IsInitialized() || a->data_type() != b->data_type() ||
        !IsFp8Type(a->data_type()) || projected->data_type() != a->data_type() ||
        logits->data_type() != DataType::BF16 || !b->is_immutable() ||
        !HasCompatiblePerTensorQuantization(a->quantization()) ||
        !HasCompatiblePerTensorQuantization(b->quantization()) ||
        !HasCompatiblePerTensorQuantization(projected->quantization()) ||
        !std::isfinite(projected->quantization_scale()) || projected->quantization_scale() <= 0.0f ||
        a->dims().size() < 2 || b->dims().size() != 2 ||
        a->dims()[a->dims().size() - 1] != b->dims()[1] ||
        a->numel() != a->dims()[a->dims().size() - 1] || b->dims()[0] <= 0) {
        return false;
    }
    *gemm = producer;
    *output_scale = projected->quantization_scale();
    return true;
}

}  // namespace

const std::string& QwenGemmArgmaxFusionPass::name() const { return kPassName; }

int32_t QwenGemmArgmaxFusionPass::Run(StaticGraph* graph) {
    if (graph == nullptr) {
        return -1;
    }
    if (!IsQwenModel(*graph) ||
        (graph->KernelDevice() != DeviceType::X86 && graph->KernelDevice() != DeviceType::CUDA)) {
        return 0;
    }
    for (const auto& node : graph->nodes()) {
        if (IsFinalQwenLogitsGemm(*graph, node)) {
            if (!graph->SetValueDataType("logits", DataType::INT64) ||
                !graph->ReplaceNodeOp(node.name, "QwenGemmArgmax", {node.inputs[0], node.inputs[1]})) {
                return -1;
            }
            continue;
        }

        const StaticNode* gemm = nullptr;
        float output_scale = 1.0f;
        if (!IsFp8LogitsCast(*graph, node, &gemm, &output_scale)) {
            continue;
        }
        if (graph->KernelDevice() == DeviceType::CUDA) {
            // CUDA only has a BF16 QwenGemmArgmax kernel. Keep the supported
            // FP8 Gemm + Cast path intact until an FP8 fused kernel exists.
            continue;
        }
        model::NodeDesc replacement;
        replacement.name = node.name;
        replacement.op_type = "QwenGemmArgmax";
        replacement.inputs = {gemm->inputs[0], gemm->inputs[1]};
        replacement.outputs = node.outputs;
        replacement.attributes["output_scale"] = output_scale;
        if (!graph->SetValueDataType("logits", DataType::INT64) ||
            !graph->ReplaceNodeDesc(replacement) || !graph->RemoveNode(gemm->name)) {
            return -1;
        }
    }
    return 0;
}

}  // namespace feather
