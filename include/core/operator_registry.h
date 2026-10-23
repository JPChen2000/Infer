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

#ifndef FEATHER_CORE_OPERATOR_REGISTRY_H
#define FEATHER_CORE_OPERATOR_REGISTRY_H

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/operator.h"
#include "model/model_format.h"

namespace feather {

namespace operators {
void RegisterBuiltinOperators();
void EnsureBuiltinOperatorsRegistered();
}

class OperatorRegistry {
   public:
    using TensorMap = std::unordered_map<std::string, std::shared_ptr<Tensor>>;
    struct BuildContext {
        DeviceType device{GetHostRuntimeDevice()};
        KernelFallbackPolicy fallback{KernelFallbackPolicy::kAllowCommon};
        DataType data_type{DataType::UNKNOWN};
        DataLayout layout{DataLayout::ND};
        bool defer_kernels{false};
        size_t workspace_limit{static_cast<size_t>(-1)};
    };
    using Builder = std::function<std::shared_ptr<OpBase>(const model::NodeDesc&, TensorMap&, const BuildContext&)>;

    static OperatorRegistry& instance();

    void Register(const std::string& op_type, Builder builder);
    std::shared_ptr<OpBase> Create(const model::NodeDesc& node, TensorMap& tensors) const;
    std::shared_ptr<OpBase> Create(const model::NodeDesc& node, TensorMap& tensors, const BuildContext& context) const;
    const Status& LastCreateStatus() const { return last_create_status_; }

   private:
    mutable std::mutex registry_mutex_;
    inline static thread_local Status last_create_status_;
    std::unordered_map<std::string, std::shared_ptr<const Builder>> registry_;
};

inline DeviceType NormalizeKernelDevice(DeviceType device) {
    return device == DeviceType::UNKNOWN ? GetHostRuntimeDevice() : device;
}

namespace detail {
inline thread_local DeviceType g_active_kernel_device = GetHostRuntimeDevice();
}

inline DeviceType ActiveKernelDevice() {
    return NormalizeKernelDevice(detail::g_active_kernel_device);
}

class KernelDeviceScope {
   public:
    explicit KernelDeviceScope(DeviceType device) : previous_(detail::g_active_kernel_device) {
        detail::g_active_kernel_device = NormalizeKernelDevice(device);
    }

    KernelDeviceScope(const KernelDeviceScope&) = delete;
    KernelDeviceScope& operator=(const KernelDeviceScope&) = delete;

    ~KernelDeviceScope() { detail::g_active_kernel_device = previous_; }

   private:
    DeviceType previous_;
};

inline DataType ResolveExecutionDataType(const std::vector<std::shared_ptr<Tensor>>& tensors,
                                         DataType fallback = DataType::FP32) {
    for (const auto& tensor : tensors) {
        if (tensor != nullptr && tensor->data_type() != DataType::UNKNOWN) {
            return tensor->data_type();
        }
    }
    return fallback;
}

inline DataLayout ResolveExecutionLayout(const std::vector<std::shared_ptr<Tensor>>& tensors,
                                         DataLayout fallback = DataLayout::ND) {
    for (const auto& tensor : tensors) {
        if (tensor != nullptr && tensor->layout() != DataLayout::ND) {
            return tensor->layout();
        }
    }
    return fallback;
}

inline std::unique_ptr<KernelBase> CreateKernelForTensor(DeviceType device, const std::string& op_type,
                                                         const std::vector<std::shared_ptr<Tensor>>& tensors,
                                                         DataType fallback = DataType::FP32) {
    return KernelDispatcher::instance().create(KernelRequest{device, ResolveExecutionDataType(tensors, fallback),
        ResolveExecutionLayout(tensors), op_type, tensors, KernelSelectionScope::Node(), KernelSelectionScope::WorkspaceLimit()}, KernelSelectionScope::Policy());
}

inline std::unique_ptr<KernelBase> CreateHostKernelForTensor(const std::string& op_type,
                                                             const std::vector<std::shared_ptr<Tensor>>& tensors,
                                                             DataType fallback = DataType::FP32) {
    return CreateKernelForTensor(ActiveKernelDevice(), op_type, tensors, fallback);
}

}  // namespace feather

#endif  // FEATHER_CORE_OPERATOR_REGISTRY_H
