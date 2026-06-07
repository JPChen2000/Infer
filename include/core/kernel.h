#ifndef FEATHER_CORE_KERNEL_H
#define FEATHER_CORE_KERNEL_H
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>

#include "core/tensor.h"
using feather::Tensor;

namespace feather {
namespace kernel {
void EnsureBuiltinKernelsRegistered();
}

class KernelBase {
   public:
    KernelBase() = default;
    explicit KernelBase(const KernelBase& base);
    virtual void SetParam(void* param) { param_ = param; };
    virtual int32_t compute() = 0;

   protected:
    void* param_;
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

    void registerKernel(DeviceType dev, DataType dtype, const std::string& op_type, KernelFactory factory) {
        registry_[dev][dtype][op_type] = std::move(factory);
    }

    std::unique_ptr<KernelBase> create(DeviceType dev, DataType dtype, const std::string& op_type) const {
        if (auto kernel = createExact(dev, dtype, op_type); kernel != nullptr) {
            return kernel;
        }
        if (dev != DeviceType::COMMON) {
            if (auto kernel = createExact(DeviceType::COMMON, dtype, op_type); kernel != nullptr) {
                return kernel;
            }
        }
        return nullptr;
    }

   private:
    std::unique_ptr<KernelBase> createExact(DeviceType dev, DataType dtype, const std::string& op_type) const {
        if (auto dev_it = registry_.find(dev); dev_it != registry_.end()) {
            if (auto dtype_it = dev_it->second.find(dtype); dtype_it != dev_it->second.end()) {
                if (auto op_it = dtype_it->second.find(op_type); op_it != dtype_it->second.end()) {
                    return op_it->second();
                }
            }
        }
        return nullptr;
    }

    std::map<DeviceType, std::map<DataType, std::map<std::string, KernelFactory>>> registry_;
};

#define REGISTER_KERNEL(dev, dtype, op, klass)                                                       \
    class klass##Registerer {                                                                        \
       public:                                                                                       \
        klass##Registerer() {                                                                        \
            KernelDispatcher::instance().registerKernel(DeviceType::dev, DataType::dtype, #op,       \
                                                        []() { return std::make_unique<klass>(); }); \
        }                                                                                            \
    };                                                                                               \
    static klass##Registerer global_##klass##Registerer;

}  // namespace feather
#endif  // FEATHER_CORE_KERNEL_H
