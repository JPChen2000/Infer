#ifndef FEATHER_OPERATOR_GEMM_OP_H
#define FEATHER_OPERATOR_GEMM_OP_H

#include <memory>
#include <string>

#include "core/operator.h"
#include "src/kernel/gemm.h"
#include "src/operator/params.h"

namespace feather {
namespace operators {

class GemmOp : public OpBase {
   public:
    GemmOp();
    explicit GemmOp(const GemmParam& param);
    GemmOp(std::string name, const GemmParam& param);

    int32_t CheckShape() const override;
    int32_t InferOutputShapes() override;
    void AttachKernel(std::unique_ptr<KernelBase> kernel) override {
        kernel_ = std::move(kernel);
        if (kernel_ != nullptr) {
            kernel_->SetParam((void*)&param_);
        }
    }
    std::unique_ptr<KernelBase> DetachKernel() override { return std::move(kernel_); }
    bool HasKernel() const override { return kernel_ != nullptr; }
    int32_t Run() override;

   private:
    void SyncIO();

    GemmParam param_;
    std::unique_ptr<KernelBase> kernel_;
};

}  // namespace operators
}  // namespace feather

#endif  // FEATHER_OPERATOR_GEMM_OP_H
