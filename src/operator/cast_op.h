#ifndef FEATHER_OPERATOR_CAST_OP_H
#define FEATHER_OPERATOR_CAST_OP_H

#include <memory>
#include <string>

#include "core/operator.h"
#include "src/kernel/cast.h"
#include "src/operator/params.h"

namespace feather {
namespace operators {

class CastOp : public OpBase {
   public:
    CastOp(std::string name, const CastParam& param);

    int32_t CheckShape() const override;
    int32_t InferOutputShapes() override;
    void AttachKernel(std::unique_ptr<KernelBase> kernel) override;
    std::unique_ptr<KernelBase> DetachKernel() override { return std::move(kernel_); }
    bool HasKernel() const override { return kernel_ != nullptr; }
    int32_t Run() override;

   private:
    void SyncIO();

    CastParam param_;
    std::unique_ptr<KernelBase> kernel_;
};

void EnsureCastOperatorRegistered();

}  // namespace operators
}  // namespace feather

#endif  // FEATHER_OPERATOR_CAST_OP_H
