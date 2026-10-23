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

void OperatorRegistry::Register(const std::string& op_type, Builder builder) {
    auto replacement = std::make_shared<const Builder>(std::move(builder));
    {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        registry_[op_type].swap(replacement);
    }
    // Retire captured resources after unlocking: their destructors can invoke
    // user code, including another registration.
}

std::shared_ptr<OpBase> OperatorRegistry::Create(const model::NodeDesc& node, TensorMap& tensors) const {
    return Create(node, tensors, BuildContext{ActiveKernelDevice()});
}

std::shared_ptr<OpBase> OperatorRegistry::Create(const model::NodeDesc& node, TensorMap& tensors,
                                                 const BuildContext& context) const {
    last_create_status_ = Status::Ok();
    KernelDispatcher::ResetSelectionStatus();
    std::shared_ptr<const Builder> builder;
    {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        const auto it = registry_.find(node.op_type);
        if (it == registry_.end()) {
            last_create_status_ = Status::Error(StatusCode::kNotFound, "unregistered operator: " + node.op_type);
            return nullptr;
        }
        builder = it->second;
    }
    KernelDeviceScope device_scope(context.device);
    KernelSelectionScope selection_scope(context.fallback, &node, context.defer_kernels, context.workspace_limit);
    // User builders may register replacements or recursively create operators.
    // Keep this invocation alive independently and never hold the map lock here.
    auto op = (*builder)(node, tensors, context);
    if (op != nullptr) {
        op->SetExecutionDevice(context.device);
        auto provisional = op->DetachKernel();
        if (provisional) {
            last_create_status_ = KernelBase::FinalizeSelection(&provisional);
            if (!last_create_status_.ok()) return nullptr;
            op->AttachKernel(std::move(provisional));
        }
    } else {
        last_create_status_ = KernelDispatcher::LastSelectionStatus();
        if (last_create_status_.ok())
            last_create_status_ = Status::Error(StatusCode::kBuildFailed, "operator construction failed: " + node.op_type);
    }
    return op;
}

}  // namespace feather
