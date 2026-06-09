#ifndef FEATHER_OPERATOR_RESIZE_CONCAT_OP_H
#define FEATHER_OPERATOR_RESIZE_CONCAT_OP_H

#include <memory>
#include <string>

#include "core/operator.h"
#include "src/kernel/resize_concat.h"
#include "src/operator/params.h"

namespace feather {
namespace operators {

class ResizeConcatOp : public OpBase {
   public:
    ResizeConcatOp();
    explicit ResizeConcatOp(const ResizeConcatParam& param);
    ResizeConcatOp(std::string name, const ResizeConcatParam& param);

    int32_t CheckShape() const override;
    int32_t InferOutputShapes() override;
    void AttachKernel(std::unique_ptr<KernelBase> kernel) override;
    std::unique_ptr<KernelBase> DetachKernel() override { return std::move(kernel_); }
    bool HasKernel() const override { return kernel_ != nullptr; }
    int32_t Run() override;

   private:
    void SyncIO();

    ResizeConcatParam param_;
    std::unique_ptr<KernelBase> kernel_;
};

void EnsureResizeConcatOperatorRegistered();

}  // namespace operators
}  // namespace feather

#endif  // FEATHER_OPERATOR_RESIZE_CONCAT_OP_H
