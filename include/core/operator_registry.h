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
void EnsureBuiltinOperatorsRegistered();
}

class OperatorRegistry {
   public:
    using TensorMap = std::unordered_map<std::string, std::shared_ptr<Tensor>>;
    using Builder = std::function<std::shared_ptr<OpBase>(const model::NodeDesc&, TensorMap&)>;

    static OperatorRegistry& instance();

    void Register(const std::string& op_type, Builder builder);
    std::shared_ptr<OpBase> Create(const model::NodeDesc& node, TensorMap& tensors) const;
    const Builder* Find(const std::string& op_type) const;

   private:
    std::unordered_map<std::string, Builder> registry_;
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

inline std::unique_ptr<KernelBase> CreateKernelForTensor(DeviceType device, const std::string& op_type,
                                                         const std::vector<std::shared_ptr<Tensor>>& tensors,
                                                         DataType fallback = DataType::FP32) {
    return KernelDispatcher::instance().create(device, ResolveExecutionDataType(tensors, fallback), op_type);
}

inline std::unique_ptr<KernelBase> CreateHostKernelForTensor(const std::string& op_type,
                                                             const std::vector<std::shared_ptr<Tensor>>& tensors,
                                                             DataType fallback = DataType::FP32) {
    return CreateKernelForTensor(GetHostRuntimeDevice(), op_type, tensors, fallback);
}

}  // namespace feather

#endif  // FEATHER_CORE_OPERATOR_REGISTRY_H
