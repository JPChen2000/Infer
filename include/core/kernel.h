#ifndef FEATHER_CORE_KERNEL_H
#define FEATHER_CORE_KERNEL_H
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "core/tensor.h"
using feather::Tensor;

namespace feather {
namespace kernel {
void EnsureBuiltinKernelsRegistered();
}

class KernelBase {
   public:
    KernelBase() = default;
    virtual ~KernelBase() = default;
    virtual void SetParam(void* param) { param_ = param; };
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
    void* param_{nullptr};
    DeviceType device_{DeviceType::UNKNOWN};
    DataType data_type_{DataType::UNKNOWN};
    DataLayout layout_{DataLayout::ND};
    std::string op_type_;
};

class KernelDispatcher {
   public:
    using KernelFactory = std::function<std::unique_ptr<KernelBase>()>;

    static KernelDispatcher& instance() {
        static KernelDispatcher inst;
        static bool initializing = false;
        static bool initialized = false;
        if (!initialized && !initializing) {
            initializing = true;
            kernel::EnsureBuiltinKernelsRegistered();
            initialized = true;
            initializing = false;
        }
        return inst;
    }

    void registerKernel(DeviceType dev, DataType dtype, DataLayout layout, const std::string& op_type, KernelFactory factory) {
        registry_[dev][dtype][layout][op_type] = std::move(factory);
    }

    void registerKernel(DeviceType dev, DataType dtype, const std::string& op_type, KernelFactory factory) {
        registerKernel(dev, dtype, DataLayout::ND, op_type, std::move(factory));
    }

    std::unique_ptr<KernelBase> create(DeviceType dev, DataType dtype, DataLayout layout, const std::string& op_type) const {
        if (auto kernel = createExact(dev, dtype, layout, op_type); kernel != nullptr) {
            return kernel;
        }
        if (layout != DataLayout::ND) {
            if (auto kernel = createExact(dev, dtype, DataLayout::ND, op_type); kernel != nullptr) {
                kernel->SetMetadata(dev, dtype, layout, op_type);
                return kernel;
            }
        }
        if (dev != DeviceType::COMMON) {
            if (auto kernel = createExact(DeviceType::COMMON, dtype, layout, op_type); kernel != nullptr) {
                kernel->SetMetadata(DeviceType::COMMON, dtype, layout, op_type);
                return kernel;
            }
            if (layout != DataLayout::ND) {
                if (auto kernel = createExact(DeviceType::COMMON, dtype, DataLayout::ND, op_type); kernel != nullptr) {
                    kernel->SetMetadata(DeviceType::COMMON, dtype, layout, op_type);
                    return kernel;
                }
            }
        }
        return nullptr;
    }

    std::unique_ptr<KernelBase> create(DeviceType dev, DataType dtype, const std::string& op_type) const {
        return create(dev, dtype, DataLayout::ND, op_type);
    }

   private:
    std::unique_ptr<KernelBase> createExact(DeviceType dev, DataType dtype, DataLayout layout,
                                            const std::string& op_type) const {
        if (auto dev_it = registry_.find(dev); dev_it != registry_.end()) {
            if (auto dtype_it = dev_it->second.find(dtype); dtype_it != dev_it->second.end()) {
                if (auto layout_it = dtype_it->second.find(layout); layout_it != dtype_it->second.end()) {
                    if (auto op_it = layout_it->second.find(op_type); op_it != layout_it->second.end()) {
                        auto kernel = op_it->second();
                        if (kernel != nullptr) {
                            kernel->SetMetadata(dev, dtype, layout, op_type);
                        }
                        return kernel;
                    }
                }
            }
        }
        return nullptr;
    }

    std::map<DeviceType, std::map<DataType, std::map<DataLayout, std::map<std::string, KernelFactory>>>> registry_;

   private:
    std::unique_ptr<KernelBase> createExact(DeviceType dev, DataType dtype, const std::string& op_type) const {
        return createExact(dev, dtype, DataLayout::ND, op_type);
    }
};

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
