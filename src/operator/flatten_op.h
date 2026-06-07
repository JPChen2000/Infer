#ifndef FEATHER_OPERATOR_FLATTEN_OP_H
#define FEATHER_OPERATOR_FLATTEN_OP_H

#include <memory>
#include <string>

#include "core/operator.h"
#include "src/kernel/flatten.h"
#include "src/operator/params.h"

namespace feather {
namespace operators {

class FlattenOp : public OpBase {
   public:
    FlattenOp();
    explicit FlattenOp(const FlattenParam& param);
    FlattenOp(std::string name, const FlattenParam& param);

    int32_t CheckShape() const override;
    int32_t InferOutputShapes() override;
    void AttachKernel(std::unique_ptr<KernelBase> kernel) override;
    std::unique_ptr<KernelBase> DetachKernel() override { return std::move(kernel_); }
    bool HasKernel() const override { return kernel_ != nullptr; }
    int32_t Run() override;

   private:
    void SyncIO();

    FlattenParam param_;
    std::unique_ptr<KernelBase> kernel_;
};

}  // namespace operators
}  // namespace feather

#endif  // FEATHER_OPERATOR_FLATTEN_OP_H
