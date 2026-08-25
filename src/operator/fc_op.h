#ifndef FEATHER_OPERATOR_FC_OP_H
#define FEATHER_OPERATOR_FC_OP_H

#include <memory>
#include <string>

#include "core/dim.h"
#include "core/operator.h"
#include "core/tensor.h"
#include "src/kernel/fc.h"
#include "src/operator/params.h"
using feather::kernel::FcKernel;
namespace feather {
namespace operators {
class FcOp : public OpBase {
   public:
    FcOp();
    explicit FcOp(const FcParam& param);
    FcOp(std::string name, const FcParam& param);

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

   private:
    void SyncIO();

    FcParam param_;
    std::unique_ptr<KernelBase> kernel_;
};

}  // namespace operators
}  // namespace feather

#endif  // FEATHER_OPERATOR_FC_OP_H
