#ifndef FEATHER_OPERATOR_POOL_OP_H
#define FEATHER_OPERATOR_POOL_OP_H

#include <memory>
#include <string>

#include "core/operator.h"
#include "src/kernel/pool.h"
#include "src/operator/params.h"

namespace feather {
namespace operators {

class AvgPoolOp : public OpBase {
   public:
    AvgPoolOp();
    explicit AvgPoolOp(const PoolParam& param);
    AvgPoolOp(std::string name, const PoolParam& param);

    int32_t CheckShape() const override;
    int32_t InferOutputShapes() override;
    void AttachKernel(std::unique_ptr<KernelBase> kernel) override;
    std::unique_ptr<KernelBase> DetachKernel() override { return std::move(kernel_); }
    bool HasKernel() const override { return kernel_ != nullptr; }
    int32_t Run() override;

   private:
    void SyncIO();

    PoolParam param_;
    std::unique_ptr<KernelBase> kernel_;
};

class MaxPoolOp : public OpBase {
   public:
    MaxPoolOp();
    explicit MaxPoolOp(const PoolParam& param);
    MaxPoolOp(std::string name, const PoolParam& param);

    int32_t CheckShape() const override;
    int32_t InferOutputShapes() override;
    void AttachKernel(std::unique_ptr<KernelBase> kernel) override;
    std::unique_ptr<KernelBase> DetachKernel() override { return std::move(kernel_); }
    bool HasKernel() const override { return kernel_ != nullptr; }
    int32_t Run() override;

   private:
    void SyncIO();

    PoolParam param_;
    std::unique_ptr<KernelBase> kernel_;
};

}  // namespace operators
}  // namespace feather

#endif  // FEATHER_OPERATOR_POOL_OP_H
