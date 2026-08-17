#ifndef FEATHER_OPERATOR_SUB_OP_H
#define FEATHER_OPERATOR_SUB_OP_H

#include <memory>
#include <string>

#include "core/operator.h"
#include "src/kernel/sub.h"
#include "src/operator/params.h"

namespace feather {
namespace operators {

class SubOp : public OpBase {
   public:
    SubOp(std::string name, const BinaryParam& param);

    int32_t CheckShape() const override;
    int32_t InferOutputShapes() override;
    void AttachKernel(std::unique_ptr<KernelBase> kernel) override;
    std::unique_ptr<KernelBase> DetachKernel() override { return std::move(kernel_); }
    bool HasKernel() const override { return kernel_ != nullptr; }
    int32_t Run() override;

   private:
    void SyncIO();

    BinaryParam param_;
    std::unique_ptr<KernelBase> kernel_;
};

}  // namespace operators
}  // namespace feather

#endif  // FEATHER_OPERATOR_SUB_OP_H
