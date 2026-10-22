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

#ifndef FEATHER_CORE_KERNEL_H
#define FEATHER_CORE_KERNEL_H
#include <functional>
#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <typeindex>
#include <mutex>
#include "core/status.h"

#include "core/tensor.h"
using feather::Tensor;

namespace feather {
namespace kernel {
void RegisterBuiltinKernels();
void EnsureBuiltinKernelsRegistered();
}

namespace model { struct NodeDesc; }
enum class KernelFallbackPolicy { kAllowCommon, kRequireRequested };
struct KernelRequest {
    DeviceType device{DeviceType::UNKNOWN};
    DataType dtype{DataType::UNKNOWN};
    DataLayout layout{DataLayout::ND};
    std::string op_type;
    std::vector<std::shared_ptr<Tensor>> tensors;
    const model::NodeDesc* node{nullptr};
    size_t workspace_limit{static_cast<size_t>(-1)};
};
class KernelSelectionScope {
 public:
    KernelSelectionScope(KernelFallbackPolicy policy, const model::NodeDesc* node, bool defer = false,
                         size_t workspace_limit = static_cast<size_t>(-1))
        : previous_defer_(defer_), previous_policy_(policy_), previous_node_(node_), previous_workspace_(workspace_limit_) {
        policy_ = policy; node_ = node; defer_ = defer; workspace_limit_ = workspace_limit;
    }
    ~KernelSelectionScope() { policy_ = previous_policy_; node_ = previous_node_; defer_ = previous_defer_; workspace_limit_ = previous_workspace_; }
    static bool Defer() { return defer_; }
    static KernelFallbackPolicy Policy() { return policy_; }
    static const model::NodeDesc* Node() { return node_; }
    static size_t WorkspaceLimit() { return workspace_limit_; }
 private:
    bool previous_defer_{false};
    inline static thread_local bool defer_{false};
    KernelFallbackPolicy previous_policy_;
    const model::NodeDesc* previous_node_;
    inline static thread_local KernelFallbackPolicy policy_{KernelFallbackPolicy::kAllowCommon};
    inline static thread_local const model::NodeDesc* node_{nullptr};
    size_t previous_workspace_;
    inline static thread_local size_t workspace_limit_{static_cast<size_t>(-1)};
};

namespace detail {
template <typename T, typename = void> struct HasTensorVisitor : std::false_type {};
struct TensorVisitorProbe { void operator()(std::shared_ptr<Tensor>&) const; };
template <typename T> struct HasTensorVisitor<T, std::void_t<decltype(std::declval<T&>().VisitTensors(TensorVisitorProbe{}))>> : std::true_type {};
}

class KernelBase {
   public:
    using Factory = std::function<std::unique_ptr<KernelBase>()>;
    using TensorBindings = std::unordered_map<const Tensor*, std::shared_ptr<Tensor>>;
    using ParameterBinder = std::function<Status(KernelBase&, const TensorBindings&)>;
    KernelBase() = default;
    virtual ~KernelBase() = default;
    virtual void SetParam(void* param) {
        clone_param_ = {};
        bind_param_ = {};
        param_type_ = typeid(void);
        param_owner_.reset();
        param_ = param;
    };
    // Own the parameter structure while retaining shared Tensor handles; this
    // keeps kernel execution independent from the operator object's lifetime
    // without copying model data.
    template <typename Param>
    void SetParamOwner(std::shared_ptr<Param> param) {
        const auto snapshot = *param;
        clone_param_ = [snapshot](KernelBase& target) {
            target.SetParamOwner(std::make_shared<Param>(snapshot));
        };
        bind_param_ = [snapshot](KernelBase& target, const TensorBindings& bindings) {
            if constexpr (detail::HasTensorVisitor<Param>::value) {
                auto rebound = std::make_shared<Param>(snapshot);
                bool complete = true;
                rebound->VisitTensors([&](std::shared_ptr<Tensor>& tensor) {
                    if (!tensor) return;
                    const auto found = bindings.find(tensor.get());
                    if (found == bindings.end()) complete = false;
                    else tensor = found->second;
                });
                if (!complete) return Status::Error(StatusCode::kBuildFailed, "unbound tensor in kernel parameters");
                target.SetParamOwner(std::move(rebound));
                return Status::Ok();
            } else {
                return Status::Error(StatusCode::kUnsupported, "kernel parameter schema has no Tensor visitor");
            }
        };
        param_type_ = typeid(Param);
        param_owner_ = std::move(param);
        param_ = param_owner_.get();
    }
    template <typename Param> Param* ParamAs() const {
        if (param_ == nullptr) return nullptr;
        if (param_type_ != typeid(void) && param_type_ != typeid(Param))
            throw std::logic_error("kernel parameter type mismatch: " + op_type_);
        return static_cast<Param*>(param_);
    }
    void SetFactory(Factory factory) { factory_ = std::move(factory); }
    Factory InstanceFactory() const {
        if (!factory_) return {};
        const auto factory = factory_;
        const auto device = device_; const auto dtype = data_type_; const auto layout = layout_; const auto op = op_type_;
        return [factory, device, dtype, layout, op] {
            auto result = factory();
            if (result) { result->SetMetadata(device, dtype, layout, op); result->SetFactory(factory); }
            return result;
        };
    }
    std::unique_ptr<KernelBase> Clone() const;
    ParameterBinder SnapshotParameters() const { return bind_param_; }
    virtual Status Supports() const { return Status::Ok(); }
    bool OwnsParam() const { return param_owner_ != nullptr; }
    // Optional one-time preparation hook for immutable weights and other
    // runtime-owned resources. Kernels may keep the default no-op behavior.
    virtual int32_t Prepare() { return 0; }
    virtual int32_t compute() = 0;
    void SetMetadata(DeviceType device, DataType data_type, std::string op_type) {
        device_ = device;
        data_type_ = data_type;
        op_type_ = std::move(op_type);
    }
    void SetMetadata(DeviceType device, DataType data_type, DataLayout layout, std::string op_type) {
        device_ = device;
        data_type_ = data_type;
        layout_ = layout;
        op_type_ = std::move(op_type);
    }
    DeviceType device() const { return device_; }
    DataType data_type() const { return data_type_; }
    DataLayout layout() const { return layout_; }
    const std::string& op_type() const { return op_type_; }

   protected:
    Factory factory_;
    ParameterBinder bind_param_;
    std::function<void(KernelBase&)> clone_param_;
    std::type_index param_type_{typeid(void)};
    void* param_{nullptr};
    std::shared_ptr<void> param_owner_;
    DeviceType device_{DeviceType::UNKNOWN};
    DataType data_type_{DataType::UNKNOWN};
    DataLayout layout_{DataLayout::ND};
    std::string op_type_;
};

// Static operators retain a parameter recipe. Concrete backend instances are
// created only when lowering or instantiating an execution session.
class DeferredKernel final : public KernelBase {
 public:
    int32_t compute() override {
        auto concrete = Clone();
        return concrete && concrete->Supports().ok() && concrete->Prepare() == 0 ? concrete->compute() : -1;
    }
};

class KernelDispatcher {
   public:
    using KernelFactory = std::function<std::unique_ptr<KernelBase>()>;

    static KernelDispatcher& instance() {
        static KernelDispatcher inst;
        static std::recursive_mutex initialization_mutex;
        std::lock_guard<std::recursive_mutex> lock(initialization_mutex);
        static bool initializing = false;
        static bool initialized = false;
        if (!initialized && !initializing) {
            initializing = true;
            kernel::RegisterBuiltinKernels();
            initialized = true;
            initializing = false;
        }
        return inst;
    }

    using SupportPredicate = std::function<Status(const KernelRequest&)>;
    void RegisterCandidate(DeviceType dev, DataType dtype, DataLayout layout, const std::string& op_type,
                           std::string id, KernelFactory factory, SupportPredicate supports = {}, int priority = 0) {
        std::lock_guard<std::recursive_mutex> lock(registry_mutex_);
        auto& candidates = registry_[dev][dtype][layout][op_type];
        candidates.erase(std::remove_if(candidates.begin(), candidates.end(), [&](const Candidate& candidate) {
            return candidate.id == id;
        }), candidates.end());
        candidates.push_back({std::move(id), std::move(factory), std::move(supports), priority});
        std::stable_sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) { return a.priority > b.priority; });
    }
    void registerKernel(DeviceType dev, DataType dtype, DataLayout layout, const std::string& op_type, KernelFactory factory) {
        RegisterCandidate(dev, dtype, layout, op_type, "", std::move(factory));
    }
    void registerKernel(DeviceType dev, DataType dtype, const std::string& op_type, KernelFactory factory) {
        registerKernel(dev, dtype, DataLayout::ND, op_type, std::move(factory));
    }
    std::unique_ptr<KernelBase> create(const KernelRequest& request, KernelFallbackPolicy policy = KernelFallbackPolicy::kAllowCommon) const {
        auto dev = request.device;
        if (dev == DeviceType::X86 && !HostSupportsX86Kernels()) {
            if (policy == KernelFallbackPolicy::kRequireRequested) return nullptr;
            dev = DeviceType::COMMON;
        }
        if (auto kernel = createExact(dev, request.layout, request)) return kernel;
        if (request.layout != DataLayout::ND)
            if (auto kernel = createExact(dev, DataLayout::ND, request)) return kernel;
        if (policy == KernelFallbackPolicy::kRequireRequested ||
            (request.device == DeviceType::CUDA && request.dtype == DataType::INT8)) return nullptr;
        if (dev != DeviceType::COMMON) {
            if (auto kernel = createExact(DeviceType::COMMON, request.layout, request)) return kernel;
            if (request.layout != DataLayout::ND)
                if (auto kernel = createExact(DeviceType::COMMON, DataLayout::ND, request)) return kernel;
        }
        return nullptr;
    }
    std::unique_ptr<KernelBase> create(DeviceType dev, DataType dtype, DataLayout layout, const std::string& op_type) const {
        return create(KernelRequest{dev, dtype, layout, op_type, {}, KernelSelectionScope::Node(), KernelSelectionScope::WorkspaceLimit()}, KernelSelectionScope::Policy());
    }
    std::unique_ptr<KernelBase> create(DeviceType dev, DataType dtype, const std::string& op_type) const {
        return create(dev, dtype, DataLayout::ND, op_type);
    }
 private:
    struct Candidate { std::string id; KernelFactory factory; SupportPredicate supports; int priority; };
    std::unique_ptr<KernelBase> createExact(DeviceType dev, DataLayout layout, const KernelRequest& request) const {
        std::lock_guard<std::recursive_mutex> lock(registry_mutex_);
        const auto dev_it = registry_.find(dev); if (dev_it == registry_.end()) return nullptr;
        const auto dtype_it = dev_it->second.find(request.dtype); if (dtype_it == dev_it->second.end()) return nullptr;
        const auto layout_it = dtype_it->second.find(layout); if (layout_it == dtype_it->second.end()) return nullptr;
        const auto op_it = layout_it->second.find(request.op_type); if (op_it == layout_it->second.end()) return nullptr;
        auto actual = request; actual.device = dev;
        for (const auto& candidate : op_it->second) {
            if (candidate.supports && !candidate.supports(actual).ok()) continue;
            std::unique_ptr<KernelBase> kernel;
            if (KernelSelectionScope::Defer()) kernel = std::make_unique<DeferredKernel>();
            else kernel = candidate.factory();
            if (kernel) {
                kernel->SetMetadata(dev, request.dtype, request.layout, request.op_type);
                kernel->SetFactory(candidate.factory);
                return kernel;
            }
        }
        return nullptr;
    }
    mutable std::recursive_mutex registry_mutex_;
    std::map<DeviceType, std::map<DataType, std::map<DataLayout, std::map<std::string, std::vector<Candidate>>>>> registry_;
};

inline std::unique_ptr<KernelBase> KernelBase::Clone() const {
    if (!clone_param_) return nullptr;
    auto make = InstanceFactory();
    auto result = make ? make() : KernelDispatcher::instance().create(device_, data_type_, layout_, op_type_);
    if (result) clone_param_(*result);
    return result;
}

#define REGISTER_LAYOUT_KERNEL(dev, dtype, layout, op, klass)                                                \
    class klass##layout##Registerer {                                                                         \
       public:                                                                                                \
        klass##layout##Registerer() {                                                                         \
            KernelDispatcher::instance().registerKernel(DeviceType::dev, DataType::dtype, DataLayout::layout,\
                                                        #op, []() { return std::make_unique<klass>(); });    \
        }                                                                                                     \
    };                                                                                                        \
    static klass##layout##Registerer global_##klass##layout##Registerer;

#define REGISTER_KERNEL(dev, dtype, op, klass)                                                       \
    class klass##Registerer {                                                                        \
       public:                                                                                       \
        klass##Registerer() {                                                                        \
            KernelDispatcher::instance().registerKernel(DeviceType::dev, DataType::dtype, DataLayout::ND, #op, \
                                                        []() { return std::make_unique<klass>(); }); \
        }                                                                                            \
    };                                                                                               \
    static klass##Registerer global_##klass##Registerer;

}  // namespace feather
#endif  // FEATHER_CORE_KERNEL_H
