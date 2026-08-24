#include "core/operator_registry.h"

namespace feather {

OperatorRegistry& OperatorRegistry::instance() {
    static OperatorRegistry registry;
    static bool initializing = false;
    static bool initialized = false;
    if (!initialized && !initializing) {
        initializing = true;
        operators::EnsureBuiltinOperatorsRegistered();
        initialized = true;
        initializing = false;
    }
    return registry;
}

void OperatorRegistry::Register(const std::string& op_type, Builder builder) { registry_[op_type] = std::move(builder); }

std::shared_ptr<OpBase> OperatorRegistry::Create(const model::NodeDesc& node, TensorMap& tensors) const {
    auto it = registry_.find(node.op_type);
    if (it == registry_.end()) {
        return nullptr;
    }
    return it->second(node, tensors);
}

}  // namespace feather
