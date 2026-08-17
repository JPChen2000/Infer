#ifndef FEATHER_OPERATOR_CONSTANT_OF_SHAPE_OP_H
#define FEATHER_OPERATOR_CONSTANT_OF_SHAPE_OP_H

#include <memory>
#include <string>

#include "core/operator.h"
#include "src/kernel/constant_of_shape.h"
#include "src/operator/params.h"

namespace feather {
namespace operators {

class ConstantOfShapeOp : public OpBase {
   public:
    ConstantOfShapeOp(std::string name, const ConstantOfShapeParam& param);

    int32_t CheckShape() const override;
    int32_t InferOutputShapes() override;
    void AttachKernel(std::unique_ptr<KernelBase> kernel) override;
    std::unique_ptr<KernelBase> DetachKernel() override { return std::move(kernel_); }
    bool HasKernel() const override { return kernel_ != nullptr; }
    int32_t Run() override;

   private:
    void SyncIO();

    ConstantOfShapeParam param_;
    std::unique_ptr<KernelBase> kernel_;
};

void EnsureConstantOfShapeOperatorRegistered();

}  // namespace operators
}  // namespace feather

#endif  // FEATHER_OPERATOR_CONSTANT_OF_SHAPE_OP_H
