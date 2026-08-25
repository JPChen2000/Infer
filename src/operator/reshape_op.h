#ifndef FEATHER_OPERATOR_RESHAPE_OP_H
#define FEATHER_OPERATOR_RESHAPE_OP_H

#include <memory>
#include <string>

#include "core/operator.h"
#include "src/kernel/reshape.h"
#include "src/operator/params.h"

namespace feather {
namespace operators {

class ReshapeOp : public OpBase {
   public:
    ReshapeOp();
    explicit ReshapeOp(const ReshapeParam& param);
    ReshapeOp(std::string name, const ReshapeParam& param);

    int32_t CheckShape() const override;
    int32_t InferOutputShapes() override;
    void AttachKernel(std::unique_ptr<KernelBase> kernel) override {
        kernel_ = std::move(kernel);
        if (kernel_ != nullptr) {
            kernel_->SetParamOwner(std::make_shared<std::decay_t<decltype(param_)>>(param_));
        }
    }
    std::unique_ptr<KernelBase> DetachKernel() override { return std::move(kernel_); }
    bool HasKernel() const override { return kernel_ != nullptr; }
    int32_t Run() override;
    void RefreshKernelParams() override;

   private:
    void SyncIO();

    ReshapeParam param_;
    std::unique_ptr<KernelBase> kernel_;
};

}  // namespace operators
}  // namespace feather

#endif  // FEATHER_OPERATOR_RESHAPE_OP_H
