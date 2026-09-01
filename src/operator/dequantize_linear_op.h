#ifndef FEATHER_OPERATOR_DEQUANTIZE_LINEAR_OP_H
#define FEATHER_OPERATOR_DEQUANTIZE_LINEAR_OP_H

#include <memory>
#include <string>

#include "core/operator.h"
#include "src/kernel/dequantize_linear.h"
#include "src/operator/params.h"

namespace feather {
namespace operators {

class DequantizeLinearOp : public OpBase {
   public:
    DequantizeLinearOp(std::string name, const DequantizeLinearParam& param);

    int32_t CheckShape() const override;
    int32_t InferOutputShapes() override;
    void AttachKernel(std::unique_ptr<KernelBase> kernel) override;
    std::unique_ptr<KernelBase> DetachKernel() override { return std::move(kernel_); }
    bool HasKernel() const override { return kernel_ != nullptr; }
    int32_t Run() override;
    void RefreshKernelParams() override;

   private:
    void SyncIO();

    DequantizeLinearParam param_;
    std::unique_ptr<KernelBase> kernel_;
};

void EnsureDequantizeLinearOperatorRegistered();

}  // namespace operators
}  // namespace feather

#endif  // FEATHER_OPERATOR_DEQUANTIZE_LINEAR_OP_H
