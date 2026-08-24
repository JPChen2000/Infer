#ifndef FEATHER_OPERATOR_QWEN_GEMM_ARGMAX_OP_H
#define FEATHER_OPERATOR_QWEN_GEMM_ARGMAX_OP_H

#include <memory>
#include <string>
#include <vector>

#include "core/operator.h"
#include "src/kernel/qwen_gemm_argmax.h"
#include "src/operator/params.h"

namespace feather {
namespace operators {

class QwenGemmArgmaxOp : public OpBase {
   public:
    QwenGemmArgmaxOp(std::string name, const QwenGemmArgmaxParam& param);

    int32_t CheckShape() const override;
    int32_t InferOutputShapes() override;
    void AttachKernel(std::unique_ptr<KernelBase> kernel) override;
    std::unique_ptr<KernelBase> DetachKernel() override { return std::move(kernel_); }
    bool HasKernel() const override { return kernel_ != nullptr; }
    int32_t Run() override;

   private:
    void SyncIO();

    QwenGemmArgmaxParam param_;
    std::unique_ptr<KernelBase> kernel_;
};

void EnsureQwenGemmArgmaxOperatorRegistered();

}  // namespace operators
}  // namespace feather

#endif  // FEATHER_OPERATOR_QWEN_GEMM_ARGMAX_OP_H
