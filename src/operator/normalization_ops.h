// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#ifndef FEATHER_OPERATOR_NORMALIZATION_OPS_H
#define FEATHER_OPERATOR_NORMALIZATION_OPS_H

#include "core/operator.h"
#include "src/operator/normalization_param.h"

namespace feather {
namespace operators {

class NormalizationOp : public OpBase {
 public:
    NormalizationOp(std::string name, const NormalizationParam& param);
    int32_t CheckShape() const override;
    int32_t InferOutputShapes() override;
    void RefreshKernelParams() override;
    void AttachKernel(std::unique_ptr<KernelBase> kernel) override;
    std::unique_ptr<KernelBase> DetachKernel() override { return std::move(kernel_); }
    bool HasKernel() const override { return kernel_ != nullptr; }
    const KernelBase* kernel() const override { return kernel_.get(); }
    int32_t Run() override;

 private:
    void SyncIO();
    NormalizationParam param_;
    std::unique_ptr<KernelBase> kernel_;
};

void EnsureNormalizationOperatorsRegistered();

}  // namespace operators
}  // namespace feather
#endif
