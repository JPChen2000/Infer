// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#include "src/operator/cached_attention_op.h"

#include <cmath>
#include <vector>

#include "core/operator_registry.h"
#include "core/operator_semantics.h"
#include "src/operator/tensor_op_utils.h"

namespace feather {
namespace operators {
namespace {

bool PositiveShape(const Tensor& tensor, size_t rank) {
    if (tensor.dims().size() != rank) return false;
    for (auto dim : tensor.dims().data())
        if (dim <= 0) return false;
    size_t bytes = 0;
    return CheckedTensorBytes(tensor.dims().data(), tensor.data_type(), &bytes);
}

bool StorageReady(const Tensor& tensor) {
    size_t bytes = 0;
    return CheckedTensorBytes(tensor.dims().data(), tensor.data_type(), &bytes) && tensor.IsInitialized() &&
           tensor.memory_size() >= bytes;
}

std::shared_ptr<OpBase> BuildCachedAttentionOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors,
                                               const OperatorRegistry::BuildContext& context) {
    if (node.inputs.size() != 6 || node.outputs.size() != 1) return nullptr;
    CachedAttentionParam param;
    param.q = tensors[node.inputs[0]];
    param.k = tensors[node.inputs[1]];
    param.v = tensors[node.inputs[2]];
    param.cache_k = tensors[node.inputs[3]];
    param.cache_v = tensors[node.inputs[4]];
    param.position = tensors[node.inputs[5]];
    param.out = tensors[node.outputs[0]];
    for (const auto& attribute : node.attributes) {
        if (attribute.first == "scale") {
            const auto* value = std::get_if<float>(&attribute.second);
            if (value == nullptr || !std::isfinite(*value)) return nullptr;
            param.scale = *value;
        } else if (attribute.first == "causal") {
            const auto* value = std::get_if<int64_t>(&attribute.second);
            if (value == nullptr || *value != 1) return nullptr;
        } else if (attribute.first == "round_intermediates") {
            const auto* value = std::get_if<int64_t>(&attribute.second);
            if (value == nullptr || (*value != 0 && *value != 1)) return nullptr;
            param.round_intermediates = *value != 0;
        } else {
            return nullptr;
        }
    }
    auto op = std::make_shared<CachedAttentionOp>(node.name.empty() ? "cached_attention" : node.name, param);
    if (op->InferOutputShapes() != 0) return nullptr;
    kernel::EnsureCachedAttentionKernelsRegistered();
    auto kernel = CreateKernelForTensor(context.device, "CachedAttention", {param.q, param.out}, DataType::FP32);
    if (kernel == nullptr) return nullptr;
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_cached_attention_registered = []() {
    OperatorSemantics semantics;
    semantics.input_count = 6;
    semantics.mutable_inputs = {3, 4};
    semantics.external_inputs = {3, 4, 5};
    semantics.exclusive_inputs = {3, 4};
    semantics.serial_execution = true;
    semantics.validate_runtime = [](const std::vector<std::shared_ptr<Tensor>>& inputs) {
        const auto fail = [] {
            return Status::Error(StatusCode::kInvalidArgument, "invalid CachedAttention position or state descriptors");
        };
        if (inputs.size() != 6 || !inputs[0] || !inputs[3] || !inputs[5]) return fail();
        const auto& position = *inputs[5];
        const auto& cache = *inputs[3];
        const auto& query = *inputs[0];
        if (!position.IsInitialized() || position.data_type() != DataType::INT64 || position.numel() != 1 ||
            position.memory_size() < sizeof(int64_t) || cache.dims().size() != 4 || query.dims().size() != 4)
            return fail();
        const auto start = position.data<int64_t>()[0];
        if (start < 0 || query.dims()[2] > cache.dims()[2] || start > cache.dims()[2] - query.dims()[2]) return fail();
        return Status::Ok();
    };
    RegisterOperatorSemantics("CachedAttention", std::move(semantics));
    OperatorRegistry::instance().Register("CachedAttention", BuildCachedAttentionOp);
    return true;
}();
}  // namespace

bool ValidateCachedAttention(const CachedAttentionParam& p, bool require_storage) {
    if (!p.q || !p.k || !p.v || !p.cache_k || !p.cache_v || !p.position || !p.out || !p.causal ||
        (!std::isnan(p.scale) && !std::isfinite(p.scale)))
        return false;
    const auto dtype = p.q->data_type();
    if (dtype != DataType::FP32 && dtype != DataType::FP16 && dtype != DataType::BF16) return false;
    for (const auto& tensor : {p.q, p.k, p.v, p.cache_k, p.cache_v}) {
        if (tensor->data_type() != dtype || !PositiveShape(*tensor, 4) || tensor->quantization().enabled) return false;
    }
    if (p.cache_k->is_immutable() || p.cache_v->is_immutable() || p.out->is_immutable() ||
        (p.out->data_type() != DataType::UNKNOWN && p.out->data_type() != dtype) || p.out->quantization().enabled ||
        p.position->data_type() != DataType::INT64 || p.position->quantization().enabled ||
        !(p.position->dims().empty() || (p.position->dims().size() == 1 && p.position->dims()[0] == 1)))
        return false;
    const auto& q = p.q->dims();
    const auto& k = p.k->dims();
    const auto& v = p.v->dims();
    const auto& ck = p.cache_k->dims();
    const auto& cv = p.cache_v->dims();
    if (q[0] != k[0] || q[0] != v[0] || q[0] != ck[0] || q[0] != cv[0] || k[1] != v[1] || k[1] != ck[1] ||
        k[1] != cv[1] || q[1] % k[1] != 0 || q[2] != k[2] || q[2] != v[2] || q[3] != k[3] || q[3] != ck[3] ||
        v[3] != cv[3] || ck[2] != cv[2] || q[2] > ck[2])
        return false;
    // Storage aliasing would introduce races between chunk writes and reads.
    const std::shared_ptr<Tensor> tensors[] = {p.q, p.k, p.v, p.cache_k, p.cache_v, p.position, p.out};
    for (size_t i = 0; i < 7; ++i) {
        for (size_t j = i + 1; j < 7; ++j) {
            if (i < 3 && j < 3) continue;  // Read-only q/k/v may alias.
            if (tensors[i].get() == tensors[j].get()) return false;
            if (require_storage && tensors[i]->storage() && tensors[i]->storage() == tensors[j]->storage()) {
                const size_t a = tensors[i]->offset(), b = tensors[j]->offset();
                // Distinct tensors can occupy disjoint ranges of one arena.
                if ((a <= b && b - a < tensors[i]->memory_size()) || (b < a && a - b < tensors[j]->memory_size()))
                    return false;
            }
        }
    }
    if (require_storage) {
        if (p.out->dims().data() != std::vector<int64_t>{q[0], q[1], q[2], v[3]}) return false;
        for (const auto& tensor : tensors)
            if (!StorageReady(*tensor)) return false;
    }
    return true;
}

float CachedAttentionScale(const CachedAttentionParam& p) {
    return std::isnan(p.scale) ? 1.0f / std::sqrt(static_cast<float>(p.q->dims()[3])) : p.scale;
}

void EnsureCachedAttentionOperatorRegistered() { (void)g_cached_attention_registered; }
CachedAttentionOp::CachedAttentionOp() : OpBase("cached_attention", "CachedAttention") {}
CachedAttentionOp::CachedAttentionOp(const CachedAttentionParam& p) : CachedAttentionOp("cached_attention", p) {}
CachedAttentionOp::CachedAttentionOp(std::string name, const CachedAttentionParam& p)
    : OpBase(std::move(name), "CachedAttention"), param_(p) {
    SyncIO();
}
void CachedAttentionOp::SyncIO() {
    SetInputs({param_.q, param_.k, param_.v, param_.cache_k, param_.cache_v, param_.position});
    SetOutputs({param_.out});
}
int32_t CachedAttentionOp::CheckShape() const { return ValidateCachedAttention(param_, false) ? 0 : -1; }
int32_t CachedAttentionOp::InferOutputShapes() {
    if (CheckShape() != 0) return -1;
    const auto& q = param_.q->dims();
    const std::vector<int64_t> shape{q[0], q[1], q[2], param_.v->dims()[3]};
    size_t bytes = 0;
    if (!CheckedTensorBytes(shape, param_.q->data_type(), &bytes)) return -1;
    param_.out = tensor_op_detail::AllocateOutput(param_.out, shape, param_.q->data_type());
    if (!param_.out) return -1;
    SyncIO();
    return 0;
}
void CachedAttentionOp::RefreshKernelParams() {
    if (kernel_) kernel_->SetParamOwner(std::make_shared<CachedAttentionParam>(param_));
}
void CachedAttentionOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    RefreshKernelParams();
}
int32_t CachedAttentionOp::Run() {
    if (!kernel_ || InferOutputShapes() != 0) return -1;
    RefreshKernelParams();
    return kernel_->compute();
}
}  // namespace operators
}  // namespace feather
