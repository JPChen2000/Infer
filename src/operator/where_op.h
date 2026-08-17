#ifndef FEATHER_OPERATOR_WHERE_OP_H
#define FEATHER_OPERATOR_WHERE_OP_H

#include <memory>
#include <string>

#include "core/operator.h"
#include "src/kernel/where.h"
#include "src/operator/params.h"

namespace feather {
namespace operators {

class WhereOp : public OpBase {
   public:
    WhereOp(std::string name, const WhereParam& param);

    int32_t CheckShape() const override;
    int32_t InferOutputShapes() override;
    void AttachKernel(std::unique_ptr<KernelBase> kernel) override;
    std::unique_ptr<KernelBase> DetachKernel() override { return std::move(kernel_); }
    bool HasKernel() const override { return kernel_ != nullptr; }
    int32_t Run() override;

   private:
    void SyncIO();

    WhereParam param_;
    std::unique_ptr<KernelBase> kernel_;
};

void EnsureWhereOperatorRegistered();

}  // namespace operators
}  // namespace feather

#endif  // FEATHER_OPERATOR_WHERE_OP_H
