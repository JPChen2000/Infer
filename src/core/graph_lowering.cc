#include "core/graph_lowering.h"

#ifdef FEATHER_WITH_CUDA
#include "src/kernel/cuda/runtime.h"
#endif

namespace feather {

Status GraphLowering::LowerStatus(StaticGraph& static_graph, RuntimeGraph* runtime_graph) const {
    const auto fail = [](const std::string& message) {
        return Status::Error(StatusCode::kBuildFailed, message);
    };
    if (runtime_graph == nullptr) {
        return fail("runtime graph is null");
    }

    const auto requested_thread_mode = runtime_graph->ThreadMode();
    runtime_graph->Clear();
    runtime_graph->SetThreadMode(static_graph.KernelDevice() == DeviceType::CUDA ? RuntimeThreadMode::kSerialGraph
                                                                                : requested_thread_mode);
    runtime_graph->SetOutputNames(static_graph.model().graph.outputs);
    for (const auto& item : static_graph.tensors()) {
        if (runtime_graph->SetTensor(item.first, item.second) != 0) {
            return fail("failed to bind tensor: " + item.first);
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
            return fail("null static operator: " + static_node.name);
        }
        RuntimeNode node;
        node.name = op->name();
        node.op_type = op->type();
        node.inputs = static_node.inputs;
        node.outputs = static_node.outputs;
        node.kernel = op->DetachKernel();
        if (node.kernel == nullptr) {
            return fail("operator has no kernel: " + static_node.name);
        }
        node.kernel_device = node.kernel->device();
        runtime_graph->AddNode(std::move(node));
    }

    const auto finalize_status = runtime_graph->FinalizeStatus();
    if (!finalize_status.ok()) {
        return finalize_status;
    }
    const auto check_status = runtime_graph->CheckStatus();
    return check_status.ok() ? Status::Ok() : check_status;
}

int32_t GraphLowering::Lower(StaticGraph& static_graph, RuntimeGraph* runtime_graph) const {
    return LowerStatus(static_graph, runtime_graph).ok() ? 0 : -1;
}

}  // namespace feather
