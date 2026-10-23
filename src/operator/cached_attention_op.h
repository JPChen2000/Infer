// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#ifndef FEATHER_OPERATOR_CACHED_ATTENTION_OP_H
#define FEATHER_OPERATOR_CACHED_ATTENTION_OP_H

#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "core/operator.h"

namespace feather {
namespace operators {

// Caches are mutable inputs, not outputs: positions [0, position) must already
// contain the valid prefix. position is a device-readable INT64 scalar/[1].
// A chunk writes [position, position + T), then attends causally within it.
// Deferred/captured CUDA callers must validate 0 <= position <= capacity - T
// before every execution/replay and preserve cache device allocations. Invalid
// device controls skip all cache writes and produce NaNs, without synchronizing
// the stream or poisoning the CUDA context. Common returns an error instead.
struct CachedAttentionParam {
    std::shared_ptr<Tensor> q;
    std::shared_ptr<Tensor> k;
    std::shared_ptr<Tensor> v;
    std::shared_ptr<Tensor> cache_k;
    std::shared_ptr<Tensor> cache_v;
    std::shared_ptr<Tensor> position;
    std::shared_ptr<Tensor> out;
    // NaN denotes the default 1 / sqrt(D), resolved from the actual shape.
    float scale{std::numeric_limits<float>::quiet_NaN()};
    bool causal{true};
    bool round_intermediates{false};
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(q); visit(k); visit(v); visit(cache_k); visit(cache_v);
        visit(position); visit(out);
    }
};

// Shape-only validation never reads position; CUDA execution must remain
// capturable. With require_storage, also validate all input/output capacities.
bool ValidateCachedAttention(const CachedAttentionParam& param, bool require_storage);
float CachedAttentionScale(const CachedAttentionParam& param);
void EnsureCachedAttentionOperatorRegistered();

class CachedAttentionOp : public OpBase {
 public:
    CachedAttentionOp();
    explicit CachedAttentionOp(const CachedAttentionParam& param);
    CachedAttentionOp(std::string name, const CachedAttentionParam& param);
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
    CachedAttentionParam param_;
    std::unique_ptr<KernelBase> kernel_;
};

}  // namespace operators
namespace kernel {
void EnsureCachedAttentionKernelsRegistered();
void EnsureCudaCachedAttentionKernelsRegistered();
template <DeviceType dev, DataType dtype>
class CachedAttentionKernel : public KernelBase {
 public:
    Status Supports() const override {
        const auto* param = ParamAs<operators::CachedAttentionParam>();
        if (param == nullptr || !operators::ValidateCachedAttention(*param, false) || param->q->data_type() != dtype)
            return Status::Error(StatusCode::kUnsupported, "invalid CachedAttention shape, dtype, or cache storage");
        return Status::Ok();
    }
    int32_t compute() override;
};
}  // namespace kernel
}  // namespace feather
#endif
