#include "core/graph_lowering.h"

namespace feather {

int32_t GraphLowering::Lower(StaticGraph& static_graph, RuntimeGraph* runtime_graph) const {
    if (runtime_graph == nullptr) {
        return -1;
    }

    runtime_graph->Clear();
    for (const auto& item : static_graph.tensors()) {
        if (runtime_graph->SetTensor(item.first, item.second) != 0) {
            return -1;
        }
    }

    for (const auto& static_node : static_graph.nodes()) {
        if (static_node.removed) {
            continue;
        }
        const auto& op = static_node.op;
        if (op == nullptr) {
            return -1;
        }
        RuntimeNode node;
        node.name = op->name();
        node.op_type = op->type();
        node.inputs = static_node.inputs;
        node.outputs = static_node.outputs;
        node.owner = op;
        node.kernel = op->DetachKernel();
        if (node.kernel == nullptr) {
            return -1;
        }
        runtime_graph->AddNode(std::move(node));
    }

    if (runtime_graph->Finalize() != 0) {
        return -1;
    }
    return runtime_graph->Check();
}

}  // namespace feather
