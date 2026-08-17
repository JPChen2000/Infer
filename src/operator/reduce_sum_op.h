#ifndef FEATHER_OPERATOR_REDUCE_SUM_OP_H
#define FEATHER_OPERATOR_REDUCE_SUM_OP_H

#include <memory>
#include <string>

#include "core/operator.h"
#include "src/kernel/reduce_sum.h"
#include "src/operator/params.h"

namespace feather {
namespace operators {

class ReduceSumOp : public OpBase {
   public:
    ReduceSumOp(std::string name, const ReduceSumParam& param);
    int32_t CheckShape() const override;
    int32_t InferOutputShapes() override;
    void AttachKernel(std::unique_ptr<KernelBase> kernel) override;
    std::unique_ptr<KernelBase> DetachKernel() override { return std::move(kernel_); }
    bool HasKernel() const override { return kernel_ != nullptr; }
    int32_t Run() override;

   private:
    void SyncIO();
    ReduceSumParam param_;
    std::unique_ptr<KernelBase> kernel_;
};

void EnsureReduceSumOperatorRegistered();

}  // namespace operators
}  // namespace feather

#endif  // FEATHER_OPERATOR_REDUCE_SUM_OP_H
