#include "core/operator_registry.h"

namespace feather {

OperatorRegistry& OperatorRegistry::instance() {
    static OperatorRegistry registry;
    static bool initializing = false;
    static bool initialized = false;
    if (!initialized && !initializing) {
        initializing = true;
        operators::RegisterBuiltinOperators();
        initialized = true;
        initializing = false;
    }
    return registry;
}

void OperatorRegistry::Register(const std::string& op_type, Builder builder) { registry_[op_type] = std::move(builder); }

std::shared_ptr<OpBase> OperatorRegistry::Create(const model::NodeDesc& node, TensorMap& tensors) const {
    return Create(node, tensors, BuildContext{ActiveKernelDevice()});
}

std::shared_ptr<OpBase> OperatorRegistry::Create(const model::NodeDesc& node, TensorMap& tensors,
                                                 const BuildContext& context) const {
    auto it = registry_.find(node.op_type);
    if (it == registry_.end()) {
        return nullptr;
    }
    auto op = it->second(node, tensors, context);
    if (op != nullptr) {
        op->SetExecutionDevice(context.device);
    }
    return op;
}

}  // namespace feather
