#include "core/graph_lowering.h"

#ifdef FEATHER_WITH_CUDA
#include "src/kernel/cuda/runtime.h"
#endif

namespace feather {

int32_t GraphLowering::Lower(StaticGraph& static_graph, RuntimeGraph* runtime_graph) const {
    if (runtime_graph == nullptr) {
        return -1;
    }

    const auto requested_thread_mode = runtime_graph->ThreadMode();
    runtime_graph->Clear();
    runtime_graph->SetThreadMode(static_graph.KernelDevice() == DeviceType::CUDA ? RuntimeThreadMode::kSerialGraph
                                                                                : requested_thread_mode);
    runtime_graph->SetOutputNames(static_graph.model().graph.outputs);
    for (const auto& item : static_graph.tensors()) {
        if (runtime_graph->SetTensor(item.first, item.second) != 0) {
            return -1;
        }
    }

#ifdef FEATHER_WITH_CUDA
    for (const auto& value : static_graph.model().graph.values) {
        if (!value.constant) {
            continue;
        }
        auto tensor = static_graph.GetTensor(value.tensor.name);
        if (tensor != nullptr) {
            kernel::cuda_detail::MarkTensorDevicePersistent(tensor.get(), true);
        }
    }
#endif

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
        node.kernel_device = node.kernel->device();
        runtime_graph->AddNode(std::move(node));
    }

    if (runtime_graph->Finalize() != 0) {
        return -1;
    }
    return runtime_graph->Check();
}

}  // namespace feather
