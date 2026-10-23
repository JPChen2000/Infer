// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#include "src/operator/normalization_ops.h"

#include "core/operator_registry.h"
#include "src/kernel/normalization.h"

namespace feather {
namespace operators {
namespace {

bool ReadInt(const model::NodeDesc& node, const char* key, int64_t* value) {
    const auto it = node.attributes.find(key);
    if (it == node.attributes.end()) return true;
    const auto* number = std::get_if<int64_t>(&it->second);
    if (!number) return false;
    *value = *number;
    return true;
}

bool ReadFloat(const model::NodeDesc& node, const char* key, float* value) {
    const auto it = node.attributes.find(key);
    if (it == node.attributes.end()) return true;
    if (const auto* number = std::get_if<float>(&it->second)) { *value = *number; return true; }
    if (const auto* number = std::get_if<int64_t>(&it->second)) {
        *value = static_cast<float>(*number); return true;
    }
    return false;
}

std::shared_ptr<OpBase> BuildNormalization(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors,
                                         const OperatorRegistry::BuildContext& context, bool layer_norm) {
    if (node.inputs.size() < 2 || node.inputs.size() > 3 || node.outputs.empty() ||
        node.outputs.size() > (layer_norm ? 3u : 1u) || node.outputs[0].empty()) return nullptr;
    // StaticGraph requires a named Tensor for every output slot. Reject holes
    // rather than misinterpret an InvStdDev-only optional output as Mean.
    for (const auto& name : node.outputs) if (name.empty()) return nullptr;
    for (const auto& entry : node.attributes) {
        if (entry.first != "axis" && entry.first != "epsilon" && entry.first != "stash_type" &&
            (layer_norm || entry.first != "weight_offset")) return nullptr;
    }
    const auto tensor = [&tensors](const std::string& name) -> std::shared_ptr<Tensor> {
        const auto it = tensors.find(name);
        return it == tensors.end() ? nullptr : it->second;
    };
    NormalizationParam param;
    param.layer_norm = layer_norm;
    param.input = tensor(node.inputs[0]);
    param.weight = tensor(node.inputs[1]);
    param.out = tensor(node.outputs[0]);
    if (node.inputs.size() == 3 && !node.inputs[2].empty()) {
        param.bias = tensor(node.inputs[2]);
        if (!param.bias) return nullptr;
    }
    if (node.outputs.size() > 1 && !node.outputs[1].empty()) {
        param.mean = tensor(node.outputs[1]);
        if (!param.mean) return nullptr;
    }
    if (node.outputs.size() > 2 && !node.outputs[2].empty()) {
        param.inv_std_dev = tensor(node.outputs[2]);
        if (!param.inv_std_dev) return nullptr;
    }
    if (!ReadInt(node, "axis", &param.axis) || !ReadInt(node, "stash_type", &param.stash_type) ||
        !ReadFloat(node, "epsilon", &param.epsilon) || !ReadFloat(node, "weight_offset", &param.weight_offset))
        return nullptr;
    auto op = std::make_shared<NormalizationOp>(node.name.empty() ? node.op_type : node.name, param);
    if (op->InferOutputShapes() != 0) return nullptr;
    kernel::EnsureNormalizationKernelsRegistered();
    auto selected = CreateKernelForTensor(context.device, layer_norm ? "LayerNormalization" : "RMSNormalization",
                                          {param.input, param.weight, param.bias, param.out});
    if (!selected) return nullptr;
    op->AttachKernel(std::move(selected));
    return op;
}

bool registered = [] {
    OperatorRegistry::instance().Register("RMSNormalization", [](const auto& node, auto& tensors, const auto& context) {
        return BuildNormalization(node, tensors, context, false);
    });
    OperatorRegistry::instance().Register("LayerNormalization", [](const auto& node, auto& tensors, const auto& context) {
        return BuildNormalization(node, tensors, context, true);
    });
    return true;
}();

int32_t Allocate(const std::shared_ptr<Tensor>& output, const std::vector<int64_t>& shape,
                 DataType dtype, DataLayout layout) {
    size_t bytes = 0;
    if (!output || !CheckedTensorBytes(shape, dtype, &bytes)) return -1;
    if (!output->IsInitialized() || output->memory_size() < bytes)
        output->ResetBuffer(std::make_shared<Buffer>(bytes), bytes);
    output->Resize(shape);
    output->set_data_type(dtype);
    output->set_layout(layout);
    return 0;
}

}  // namespace

void EnsureNormalizationOperatorsRegistered() { (void)registered; }

NormalizationOp::NormalizationOp(std::string name, const NormalizationParam& param)
    : OpBase(std::move(name), param.layer_norm ? "LayerNormalization" : "RMSNormalization"), param_(param) {
    SyncIO();
}

void NormalizationOp::SyncIO() {
    std::vector<std::shared_ptr<Tensor>> inputs{param_.input, param_.weight};
    if (param_.bias) inputs.push_back(param_.bias);
    SetInputs(std::move(inputs));
    std::vector<std::shared_ptr<Tensor>> outputs{param_.out};
    if (param_.mean) outputs.push_back(param_.mean);
    if (param_.inv_std_dev) outputs.push_back(param_.inv_std_dev);
    SetOutputs(std::move(outputs));
}

int32_t NormalizationOp::CheckShape() const {
    normalization_detail::Shape shape;
    return normalization_detail::Describe(param_, &shape) ? 0 : -1;
}

int32_t NormalizationOp::InferOutputShapes() {
    normalization_detail::Shape shape;
    if (!normalization_detail::Describe(param_, &shape) ||
        Allocate(param_.out, param_.input->dims().data(), param_.input->data_type(), param_.input->layout()) != 0 ||
        (param_.mean && Allocate(param_.mean, shape.stats_dims, DataType::FP32, param_.input->layout()) != 0) ||
        (param_.inv_std_dev && Allocate(param_.inv_std_dev, shape.stats_dims, DataType::FP32, param_.input->layout()) != 0))
        return -1;
    RefreshKernelParams();
    return 0;
}

void NormalizationOp::RefreshKernelParams() {
    if (kernel_) kernel_->SetParamOwner(std::make_shared<NormalizationParam>(param_));
}

void NormalizationOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    RefreshKernelParams();
}

int32_t NormalizationOp::Run() {
    return InferOutputShapes() == 0 && kernel_ ? kernel_->compute() : -1;
}

}  // namespace operators
}  // namespace feather
