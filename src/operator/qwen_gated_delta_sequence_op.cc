// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0

#include "src/operator/qwen_gated_delta_sequence_op.h"

#include "core/operator_registry.h"

namespace feather {
namespace operators {
namespace {
std::shared_ptr<OpBase> BuildSequence(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors,
                                     const OperatorRegistry::BuildContext& context) {
    if (node.inputs.size() != 6 || node.outputs.size() != 2) return nullptr;
    auto find = [&](const std::string& name) -> std::shared_ptr<Tensor> {
        const auto entry = tensors.find(name);
        return entry == tensors.end() ? nullptr : entry->second;
    };
    QwenGatedDeltaSequenceParam p{};
    p.state = find(node.inputs[0]); p.k = find(node.inputs[1]); p.v = find(node.inputs[2]);
    p.beta = find(node.inputs[3]); p.decay = find(node.inputs[4]); p.q = find(node.inputs[5]);
    p.next_state = find(node.outputs[0]); p.out = find(node.outputs[1]);
    auto op = std::make_shared<QwenGatedDeltaSequenceOp>(
        node.name.empty() ? "qwen_gated_delta_sequence" : node.name, p);
    if (op->InferOutputShapes() != 0) return nullptr;
    auto compute = CreateKernelForTensor(context.device, "QwenGatedDeltaSequence",
        {p.state, p.k, p.v, p.beta, p.decay, p.q, p.next_state, p.out}, DataType::FP32);
    if (!compute) return nullptr;
    op->AttachKernel(std::move(compute));
    return op;
}

void ResizeOutput(const std::shared_ptr<Tensor>& output, const std::vector<int64_t>& shape) {
    size_t bytes = 0;
    CheckedByteCount(shape, sizeof(float), &bytes); // Validated before allocation.
    if (!qwen_sequence_detail::HasCapacity(*output, bytes)) {
        Tensor allocation(bytes);
        output->ResetStorage(allocation.storage(), bytes);
    }
    output->Resize(shape);
    output->set_data_type(DataType::FP32);
}
} // namespace

void EnsureQwenGatedDeltaSequenceOperatorsRegistered() {
    static const bool registered = [] {
        OperatorRegistry::instance().Register("QwenGatedDeltaSequence", BuildSequence);
        return true;
    }();
    (void)registered;
}

QwenGatedDeltaSequenceOp::QwenGatedDeltaSequenceOp(std::string name, const QwenGatedDeltaSequenceParam& p)
    : OpBase(std::move(name), "QwenGatedDeltaSequence"), param_(p) {
    SetInputs({p.state, p.k, p.v, p.beta, p.decay, p.q});
    SetOutputs({p.next_state, p.out});
}

int32_t QwenGatedDeltaSequenceOp::CheckShape() const {
    return qwen_sequence_detail::Validate(&param_, false) ? 0 : -1;
}

int32_t QwenGatedDeltaSequenceOp::InferOutputShapes() {
    if (CheckShape() != 0) return -1;
    ResizeOutput(param_.next_state, param_.state->dims().data());
    ResizeOutput(param_.out, {1, param_.k->dims()[1], param_.state->dims()[1], param_.state->dims()[3]});
    return 0;
}

void QwenGatedDeltaSequenceOp::AttachKernel(std::unique_ptr<KernelBase> compute) {
    kernel_ = std::move(compute);
    if (kernel_) kernel_->SetParamOwner(std::make_shared<QwenGatedDeltaSequenceParam>(param_));
}

int32_t QwenGatedDeltaSequenceOp::Run() {
    return InferOutputShapes() == 0 && kernel_ ? kernel_->compute() : -1;
}
} // namespace operators
} // namespace feather
