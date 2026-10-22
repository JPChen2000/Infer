// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "core/operator_registry.h"

namespace feather {

OperatorRegistry& OperatorRegistry::instance() {
    static OperatorRegistry registry;
    static std::recursive_mutex initialization_mutex;
    std::lock_guard<std::recursive_mutex> lock(initialization_mutex);
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
    KernelDeviceScope device_scope(context.device);
    KernelSelectionScope selection_scope(context.fallback, &node, context.defer_kernels, context.workspace_limit);
    auto op = it->second(node, tensors, context);
    if (op != nullptr) {
        op->SetExecutionDevice(context.device);
    }
    return op;
}

}  // namespace feather
