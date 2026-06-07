#ifndef FEATHER_OPERATOR_SPLIT_OP_H
#define FEATHER_OPERATOR_SPLIT_OP_H

#include <memory>
#include <string>

#include "core/operator.h"
#include "src/kernel/split.h"
#include "src/operator/params.h"

namespace feather {
namespace operators {

class SplitOp : public OpBase {
   public:
    SplitOp();
    explicit SplitOp(const SplitParam& param);
    SplitOp(std::string name, const SplitParam& param);

    int32_t CheckShape() const override;
    int32_t InferOutputShapes() override;
    void AttachKernel(std::unique_ptr<KernelBase> kernel) override;
    std::unique_ptr<KernelBase> DetachKernel() override { return std::move(kernel_); }
    bool HasKernel() const override { return kernel_ != nullptr; }
    int32_t Run() override;

   private:
    void SyncIO();

    SplitParam param_;
    std::unique_ptr<KernelBase> kernel_;
};

}  // namespace operators
}  // namespace feather

#endif  // FEATHER_OPERATOR_SPLIT_OP_H
