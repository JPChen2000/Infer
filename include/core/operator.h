#ifndef FEAHER_CORE_OPERATOR_H
#define FEAHER_CORE_OPERATOR_H

#include <memory>
#include <string>
#include <vector>

#include "core/kernel.h"
#include "core/tensor.h"

namespace feather {

class OpBase {
   public:
    OpBase() = default;
    OpBase(std::string name, std::string type) : name_(std::move(name)), type_(std::move(type)) {}
    virtual ~OpBase() = default;

    const std::string& name() const { return name_; }
    const std::string& type() const { return type_; }
    void SetExecutionDevice(DeviceType device) {
        execution_device_ = NormalizeExecutionDevice(device);
        execution_device_explicit_ = true;
    }
    DeviceType execution_device() const { return execution_device_; }

    const std::vector<std::shared_ptr<Tensor>>& inputs() const { return inputs_; }
    const std::vector<std::shared_ptr<Tensor>>& outputs() const { return outputs_; }

    virtual void AttachKernel(std::unique_ptr<KernelBase> kernel) = 0;
    virtual std::unique_ptr<KernelBase> DetachKernel() = 0;
    virtual bool HasKernel() const = 0;

    virtual int32_t CheckShape() const { return 0; }
    virtual int32_t InferOutputShapes() { return 0; }
    virtual void RefreshKernelParams() {}
    virtual int32_t Run() = 0;

   protected:
    void SetInputs(std::vector<std::shared_ptr<Tensor>> inputs) { inputs_ = std::move(inputs); }
    void SetOutputs(std::vector<std::shared_ptr<Tensor>> outputs) { outputs_ = std::move(outputs); }

    std::string name_;
    std::string type_;
    std::vector<std::shared_ptr<Tensor>> inputs_;
    std::vector<std::shared_ptr<Tensor>> outputs_;
    DeviceType execution_device_{GetHostRuntimeDevice()};
    bool execution_device_explicit_{false};

   private:
    static DeviceType NormalizeExecutionDevice(DeviceType device) {
        return device == DeviceType::UNKNOWN ? GetHostRuntimeDevice() : device;
    }
};

}  // namespace feather

#endif  // FEAHER_CORE_OPERATOR_H
