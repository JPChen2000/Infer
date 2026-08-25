#ifndef FEATHER_OPERATOR_CONV2D_OP_H
#define FEATHER_OPERATOR_CONV2D_OP_H

#include <memory>
#include <string>

#include "core/dim.h"
#include "core/operator.h"
#include "core/tensor.h"
#include "src/kernel/conv2d.h"
#include "src/operator/params.h"

using feather::kernel::Conv2DKernel;
namespace feather {
namespace operators {
class Conv2dOp : public OpBase {
   public:
    Conv2dOp();
    explicit Conv2dOp(const Conv2dParam& param);
    Conv2dOp(std::string name, const Conv2dParam& param);

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

    Conv2dParam param_;
    std::unique_ptr<KernelBase> kernel_;
};

}  // namespace operators
}  // namespace feather
#endif  // FEATHER_OPERATOR_CONV2D_OP_H
