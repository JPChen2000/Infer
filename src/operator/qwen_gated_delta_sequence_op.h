// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0

#ifndef FEATHER_OPERATOR_QWEN_GATED_DELTA_SEQUENCE_OP_H
#define FEATHER_OPERATOR_QWEN_GATED_DELTA_SEQUENCE_OP_H

#include "core/operator.h"
#include "src/kernel/qwen_gated_delta_sequence.h"

namespace feather {
namespace operators {
class QwenGatedDeltaSequenceOp : public OpBase {
 public:
    QwenGatedDeltaSequenceOp(std::string name, const QwenGatedDeltaSequenceParam& param);
    int32_t CheckShape() const override;
    int32_t InferOutputShapes() override;
    void AttachKernel(std::unique_ptr<KernelBase> kernel) override;
    std::unique_ptr<KernelBase> DetachKernel() override { return std::move(kernel_); }
    bool HasKernel() const override { return kernel_ != nullptr; }
    const KernelBase* kernel() const override { return kernel_.get(); }
    int32_t Run() override;
 private:
    QwenGatedDeltaSequenceParam param_;
    std::unique_ptr<KernelBase> kernel_;
};
void EnsureQwenGatedDeltaSequenceOperatorsRegistered();
} // namespace operators
} // namespace feather
#endif
