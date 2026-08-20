#ifndef FEATHER_OPERATOR_QWEN_RMS_NORM_OP_H
#define FEATHER_OPERATOR_QWEN_RMS_NORM_OP_H

#include <memory>
#include <string>

#include "core/operator.h"
#include "src/kernel/qwen_rms_norm.h"
#include "src/operator/params.h"

namespace feather {
namespace operators {

class QwenRmsNormOp : public OpBase {
   public:
    QwenRmsNormOp(std::string name, const QwenRmsNormParam& param);

    int32_t CheckShape() const override;
    int32_t InferOutputShapes() override;
    void AttachKernel(std::unique_ptr<KernelBase> kernel) override;
    std::unique_ptr<KernelBase> DetachKernel() override { return std::move(kernel_); }
    bool HasKernel() const override { return kernel_ != nullptr; }
    int32_t Run() override;

   private:
    void SyncIO();

    QwenRmsNormParam param_;
    std::unique_ptr<KernelBase> kernel_;
};

void EnsureQwenRmsNormOperatorRegistered();

}  // namespace operators
}  // namespace feather

#endif  // FEATHER_OPERATOR_QWEN_RMS_NORM_OP_H
