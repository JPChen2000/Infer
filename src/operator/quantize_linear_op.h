#ifndef FEATHER_OPERATOR_QUANTIZE_LINEAR_OP_H
#define FEATHER_OPERATOR_QUANTIZE_LINEAR_OP_H

#include <memory>
#include <string>

#include "core/operator.h"
#include "src/kernel/quantize_linear.h"
#include "src/operator/params.h"

namespace feather {
namespace operators {

class QuantizeLinearOp : public OpBase {
   public:
    QuantizeLinearOp(std::string name, const QuantizeLinearParam& param);

    int32_t CheckShape() const override;
    int32_t InferOutputShapes() override;
    void AttachKernel(std::unique_ptr<KernelBase> kernel) override;
    std::unique_ptr<KernelBase> DetachKernel() override { return std::move(kernel_); }
    bool HasKernel() const override { return kernel_ != nullptr; }
    int32_t Run() override;
    void RefreshKernelParams() override;

   private:
    void SyncIO();

    QuantizeLinearParam param_;
    std::unique_ptr<KernelBase> kernel_;
};

void EnsureQuantizeLinearOperatorRegistered();

}  // namespace operators
}  // namespace feather

#endif  // FEATHER_OPERATOR_QUANTIZE_LINEAR_OP_H
