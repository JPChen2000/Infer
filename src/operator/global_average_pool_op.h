#ifndef FEATHER_OPERATOR_GLOBAL_AVERAGE_POOL_OP_H
#define FEATHER_OPERATOR_GLOBAL_AVERAGE_POOL_OP_H

#include <memory>
#include <string>

#include "core/operator.h"
#include "src/kernel/global_average_pool.h"
#include "src/operator/params.h"

namespace feather {
namespace operators {

class GlobalAveragePoolOp : public OpBase {
   public:
    GlobalAveragePoolOp();
    explicit GlobalAveragePoolOp(const GlobalAveragePoolParam& param);
    GlobalAveragePoolOp(std::string name, const GlobalAveragePoolParam& param);

    int32_t CheckShape() const override;
    int32_t InferOutputShapes() override;
    void AttachKernel(std::unique_ptr<KernelBase> kernel) override;
    std::unique_ptr<KernelBase> DetachKernel() override { return std::move(kernel_); }
    bool HasKernel() const override { return kernel_ != nullptr; }
    int32_t Run() override;

   private:
    void SyncIO();

    GlobalAveragePoolParam param_;
    std::unique_ptr<KernelBase> kernel_;
};

void EnsureGlobalAveragePoolOperatorRegistered();

}  // namespace operators
}  // namespace feather

#endif  // FEATHER_OPERATOR_GLOBAL_AVERAGE_POOL_OP_H
