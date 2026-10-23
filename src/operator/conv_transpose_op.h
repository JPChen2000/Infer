// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#ifndef FEATHER_OPERATOR_CONV_TRANSPOSE_OP_H
#define FEATHER_OPERATOR_CONV_TRANSPOSE_OP_H

#include "core/operator.h"
#include "src/operator/conv_transpose_param.h"

namespace feather {
namespace operators {
class ConvTransposeOp final : public OpBase {
 public:
    ConvTransposeOp(std::string name, const ConvTransposeParam& param);
    int32_t CheckShape() const override;
    int32_t InferOutputShapes() override;
    int32_t Run() override;
    void AttachKernel(std::unique_ptr<KernelBase> kernel) override;
    std::unique_ptr<KernelBase> DetachKernel() override { return std::move(kernel_); }
    bool HasKernel() const override { return kernel_ != nullptr; }
    const KernelBase* kernel() const override { return kernel_.get(); }

 private:
    ConvTransposeParam param_;
    std::unique_ptr<KernelBase> kernel_;
};
}  // namespace operators
}  // namespace feather
#endif
