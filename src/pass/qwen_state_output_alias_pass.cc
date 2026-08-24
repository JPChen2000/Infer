#include "pass/qwen_state_output_alias_pass.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

#include "core/static_graph.h"

namespace feather {

namespace {

const std::string kQwenStateOutputAliasPassName = "QwenStateOutputAliasPass";
constexpr char kNextStatePrefix[] = "next_";

bool IsQwenModel(const StaticGraph& graph) {
    std::string model_name = graph.model().name + " " + graph.model().graph.name;
    std::transform(model_name.begin(), model_name.end(), model_name.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return model_name.find("qwen") != std::string::npos;
}

bool IsGraphOutput(const StaticGraph& graph, const std::string& value_name) {
    return graph.IsGraphOutputValue(value_name);
}

bool IsExplicitStateOutput(const StaticGraph& graph, const StaticNode& node) {
    if (node.removed || node.op_type != "Identity" || node.inputs.size() != 1 || node.outputs.size() != 1 ||
        !IsGraphOutput(graph, node.outputs[0]) || !graph.GetUsers(node.outputs[0]).empty()) {
        return false;
    }
    const std::string& output_name = node.outputs[0];
    if (output_name.rfind(kNextStatePrefix, 0) != 0) {
        return false;
    }
    const std::string state_input_name = output_name.substr(sizeof(kNextStatePrefix) - 1);
    return std::find(graph.model().graph.inputs.begin(), graph.model().graph.inputs.end(), state_input_name) !=
           graph.model().graph.inputs.end();
}

}  // namespace

const std::string& QwenStateOutputAliasPass::name() const { return kQwenStateOutputAliasPassName; }

int32_t QwenStateOutputAliasPass::Run(StaticGraph* graph) {
    if (graph == nullptr) {
        return -1;
    }
    // CUDA state buffers are swapped by Tensor object in the runner after the
    // graph finishes. Keep the public output name aliased to the producer
    // tensor so the relay itself does not become a runtime launch.
    if ((graph->KernelDevice() != DeviceType::X86 && graph->KernelDevice() != DeviceType::CUDA) ||
        !IsQwenModel(*graph)) {
        return 0;
    }

    std::vector<std::string> aliases;
    for (const auto& node : graph->nodes()) {
        if (IsExplicitStateOutput(*graph, node)) {
            aliases.push_back(node.name);
        }
    }
    for (const auto& node_name : aliases) {
        if (graph->GetNode(node_name) != nullptr && !graph->RemoveNodeWithOutputAlias(node_name)) {
            return -1;
        }
    }
    return 0;
}

}  // namespace feather
