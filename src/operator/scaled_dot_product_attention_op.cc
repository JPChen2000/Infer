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

#include "src/operator/scaled_dot_product_attention_op.h"

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

std::shared_ptr<OpBase> BuildScaledDotProductAttentionOp(const model::NodeDesc& node,
                                                         OperatorRegistry::TensorMap& tensors,
                                                         const OperatorRegistry::BuildContext& context) {
    if ((node.inputs.size() != 3 && node.inputs.size() != 4) || node.outputs.size() != 1) return nullptr;
    ScaledDotProductAttentionParam param;
    param.q = tensors[node.inputs[0]];
    param.k = tensors[node.inputs[1]];
    param.v = tensors[node.inputs[2]];
    if (node.inputs.size() == 4) param.mask = tensors[node.inputs[3]];
    param.out = tensors[node.outputs[0]];
    for (const auto& attribute : node.attributes) {
        if (attribute.first == "scale") {
            const auto* value = std::get_if<float>(&attribute.second);
            if (value == nullptr || !std::isfinite(*value)) return nullptr;
            param.scale = *value;
        } else if (attribute.first == "causal") {
            const auto* value = std::get_if<int64_t>(&attribute.second);
            if (value == nullptr || (*value != 0 && *value != 1)) return nullptr;
            param.causal = *value != 0;
        } else {
            return nullptr;
        }
    }
    auto op = std::make_shared<ScaledDotProductAttentionOp>(
        node.name.empty() ? "scaled_dot_product_attention" : node.name, param);
    if (op->InferOutputShapes() != 0) return nullptr;
    kernel::EnsureScaledDotProductAttentionKernelsRegistered();
    auto kernel =
        CreateKernelForTensor(context.device, "ScaledDotProductAttention", {param.q, param.out}, DataType::FP32);
    if (kernel == nullptr) return nullptr;
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_scaled_dot_product_attention_registered = []() {
    OperatorRegistry::instance().Register("ScaledDotProductAttention", BuildScaledDotProductAttentionOp);
    return true;
}();
}  // namespace

bool ValidateScaledDotProductAttention(const ScaledDotProductAttentionParam& p, bool require_storage) {
    if (!p.q || !p.k || !p.v || !p.out || (!std::isnan(p.scale) && !std::isfinite(p.scale))) return false;
    const auto dtype = p.q->data_type();
    if (dtype != DataType::FP32 && dtype != DataType::FP16 && dtype != DataType::BF16) return false;
    for (const auto& t : {p.q, p.k, p.v})
        if (t->data_type() != dtype || !PositiveShape(*t, 4) || t->quantization().enabled) return false;
    if (p.out->is_immutable() || p.out->quantization().enabled ||
        (p.out->data_type() != DataType::UNKNOWN && p.out->data_type() != dtype))
        return false;
    const auto& q = p.q->dims();
    const auto& k = p.k->dims();
    const auto& v = p.v->dims();
    if (q[0] != k[0] || q[0] != v[0] || k[1] != v[1] || q[1] % k[1] != 0 || k[2] != v[2] || q[3] != k[3]) return false;
    const std::vector<int64_t> shape{q[0], q[1], q[2], v[3]};
    size_t bytes = 0;
    if (!CheckedTensorBytes(shape, dtype, &bytes)) return false;
    if (p.mask) {
        if (p.mask->data_type() != DataType::FP32 || p.mask->quantization().enabled || !PositiveShape(*p.mask, 4))
            return false;
        const int64_t target[] = {q[0], q[1], q[2], k[2]};
        for (int i = 0; i < 4; ++i)
            if (p.mask->dims()[i] != 1 && p.mask->dims()[i] != target[i]) return false;
    }
    for (const auto& t : {p.q, p.k, p.v, p.mask}) {
        if (!t) continue;
        if (t.get() == p.out.get()) return false;
        if (t->storage() && t->storage() == p.out->storage()) {
            const size_t a = t->offset(), b = p.out->offset();
            if ((a <= b && b - a < t->memory_size()) || (b < a && a - b < p.out->memory_size())) return false;
        }
        if (require_storage && !StorageReady(*t)) return false;
    }
    if (require_storage && (p.out->data_type() != dtype || p.out->dims().data() != shape || !StorageReady(*p.out)))
        return false;
    return true;
}

float ScaledDotProductAttentionScale(const ScaledDotProductAttentionParam& p) {
    return std::isnan(p.scale) ? 1.0f / std::sqrt(static_cast<float>(p.q->dims()[3])) : p.scale;
}

void EnsureScaledDotProductAttentionOperatorRegistered() { (void)g_scaled_dot_product_attention_registered; }
ScaledDotProductAttentionOp::ScaledDotProductAttentionOp()
    : OpBase("scaled_dot_product_attention", "ScaledDotProductAttention") {}
ScaledDotProductAttentionOp::ScaledDotProductAttentionOp(const ScaledDotProductAttentionParam& p)
    : ScaledDotProductAttentionOp("scaled_dot_product_attention", p) {}
ScaledDotProductAttentionOp::ScaledDotProductAttentionOp(std::string name, const ScaledDotProductAttentionParam& p)
    : OpBase(std::move(name), "ScaledDotProductAttention"), param_(p) {
    SyncIO();
}
void ScaledDotProductAttentionOp::SyncIO() {
    if (param_.mask)
        SetInputs({param_.q, param_.k, param_.v, param_.mask});
    else
        SetInputs({param_.q, param_.k, param_.v});
    SetOutputs({param_.out});
}
int32_t ScaledDotProductAttentionOp::CheckShape() const {
    return ValidateScaledDotProductAttention(param_, false) ? 0 : -1;
}
int32_t ScaledDotProductAttentionOp::InferOutputShapes() {
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
void ScaledDotProductAttentionOp::RefreshKernelParams() {
    if (kernel_) kernel_->SetParamOwner(std::make_shared<ScaledDotProductAttentionParam>(param_));
}
void ScaledDotProductAttentionOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    RefreshKernelParams();
}
int32_t ScaledDotProductAttentionOp::Run() {
    if (!kernel_ || InferOutputShapes() != 0) return -1;
    RefreshKernelParams();
    return kernel_->compute();
}
}  // namespace operators
}  // namespace feather
