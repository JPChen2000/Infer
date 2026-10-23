// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FEATHER_OPERATOR_SCALED_DOT_PRODUCT_ATTENTION_OP_H
#define FEATHER_OPERATOR_SCALED_DOT_PRODUCT_ATTENTION_OP_H

#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "core/operator.h"

namespace feather {
namespace operators {

// Pure attention: Q [B,Hq,Q,D], K [B,Hkv,K,D], V [B,Hkv,K,Dv].
// Optional FP32 additive mask is rank 4, broadcasting over [B,Hq,Q,K].
// Causal attention uses top-left alignment (key index <= query index).
// Fully masked rows produce zero; accumulation and softmax use FP32.
struct ScaledDotProductAttentionParam {
    std::shared_ptr<Tensor> q;
    std::shared_ptr<Tensor> k;
    std::shared_ptr<Tensor> v;
    std::shared_ptr<Tensor> mask;
    std::shared_ptr<Tensor> out;
    // NaN denotes the default 1 / sqrt(D), resolved from the actual shape.
    float scale{std::numeric_limits<float>::quiet_NaN()};
    bool causal{false};
    template <typename Visitor>
    void VisitTensors(Visitor&& visit) {
        visit(q);
        visit(k);
        visit(v);
        visit(mask);
        visit(out);
    }
};

bool ValidateScaledDotProductAttention(const ScaledDotProductAttentionParam& param, bool require_storage);
float ScaledDotProductAttentionScale(const ScaledDotProductAttentionParam& param);
void EnsureScaledDotProductAttentionOperatorRegistered();

class ScaledDotProductAttentionOp : public OpBase {
   public:
    ScaledDotProductAttentionOp();
    explicit ScaledDotProductAttentionOp(const ScaledDotProductAttentionParam& param);
    ScaledDotProductAttentionOp(std::string name, const ScaledDotProductAttentionParam& param);
    int32_t CheckShape() const override;
    int32_t InferOutputShapes() override;
    void AttachKernel(std::unique_ptr<KernelBase> kernel) override;
    std::unique_ptr<KernelBase> DetachKernel() override { return std::move(kernel_); }
    bool HasKernel() const override { return kernel_ != nullptr; }
    const KernelBase* kernel() const override { return kernel_.get(); }
    void RefreshKernelParams() override;
    int32_t Run() override;

   private:
    void SyncIO();
    ScaledDotProductAttentionParam param_;
    std::unique_ptr<KernelBase> kernel_;
};

}  // namespace operators
namespace kernel {
void EnsureScaledDotProductAttentionKernelsRegistered();
void EnsureCudaScaledDotProductAttentionKernelsRegistered();
template <DeviceType dev, DataType dtype>
class ScaledDotProductAttentionKernel : public KernelBase {
   public:
    Status Supports() const override {
        const auto* param = ParamAs<operators::ScaledDotProductAttentionParam>();
        if (param == nullptr || !operators::ValidateScaledDotProductAttention(*param, false) ||
            param->q->data_type() != dtype)
            return Status::Error(StatusCode::kUnsupported,
                                 "invalid ScaledDotProductAttention shape, dtype, or storage");
        return Status::Ok();
    }
    int32_t compute() override;
};
}  // namespace kernel
}  // namespace feather
#endif
