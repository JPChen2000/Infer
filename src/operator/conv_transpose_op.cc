// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#include "src/operator/conv_transpose_op.h"

#include "core/operator_registry.h"
#include "src/kernel/conv_transpose.h"

namespace feather {
namespace operators {
namespace {
template <size_t N>
bool ReadVector(const model::NodeDesc& node, const std::string& name, std::array<int64_t, N>* value) {
    const auto entry = node.attributes.find(name);
    if (entry == node.attributes.end()) return true;
    const auto* values = std::get_if<std::vector<int64_t>>(&entry->second);
    if (!values || values->size() != N) return false;
    std::copy(values->begin(), values->end(), value->begin());
    return true;
}

std::shared_ptr<OpBase> BuildConvTranspose(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors,
                                         const OperatorRegistry::BuildContext& context) {
    if ((node.inputs.size() != 2 && node.inputs.size() != 3) || node.outputs.size() != 1 ||
        node.attributes.count("auto_pad") || node.attributes.count("output_shape")) return nullptr;
    const auto tensor = [&tensors](const std::string& name) -> std::shared_ptr<Tensor> {
        auto it = tensors.find(name);
        return it == tensors.end() ? nullptr : it->second;
    };
    ConvTransposeParam param;
    param.input = tensor(node.inputs[0]);
    param.weight = tensor(node.inputs[1]);
    param.out = tensor(node.outputs[0]);
    if (node.inputs.size() == 3 && !node.inputs[2].empty()) {
        param.bias = tensor(node.inputs[2]);
        if (!param.bias) return nullptr;
    }
    if (!ReadVector(node, "strides", &param.strides) || !ReadVector(node, "pads", &param.pads) ||
        !ReadVector(node, "dilations", &param.dilations) ||
        !ReadVector(node, "output_padding", &param.output_padding)) return nullptr;
    auto group = node.attributes.find("group");
    if (group != node.attributes.end()) {
        const auto* value = std::get_if<int64_t>(&group->second);
        if (!value) return nullptr;
        param.group = *value;
    }
    auto op = std::make_shared<ConvTransposeOp>(node.name.empty() ? "ConvTranspose" : node.name, param);
    if (op->InferOutputShapes() != 0) return nullptr;
    kernel::EnsureCommonConvTransposeKernelsRegistered();
    auto selected = CreateKernelForTensor(context.device, "ConvTranspose",
                                          {param.input, param.weight, param.bias, param.out}, DataType::FP32);
    if (!selected) return nullptr;
    op->AttachKernel(std::move(selected));
    return op;
}
bool registered = [] {
    OperatorRegistry::instance().Register("ConvTranspose", BuildConvTranspose);
    return true;
}();
}  // namespace

void EnsureConvTransposeOperatorRegistered() { (void)registered; }

ConvTransposeOp::ConvTransposeOp(std::string name, const ConvTransposeParam& param)
    : OpBase(std::move(name), "ConvTranspose"), param_(param) {
    std::vector<std::shared_ptr<Tensor>> inputs{param_.input, param_.weight};
    if (param_.bias) inputs.push_back(param_.bias);
    SetInputs(std::move(inputs));
    SetOutputs({param_.out});
}

int32_t ConvTransposeOp::CheckShape() const {
    std::vector<int64_t> shape;
    return ConvTransposeOutputShape(param_, &shape) ? 0 : -1;
}

int32_t ConvTransposeOp::InferOutputShapes() {
    std::vector<int64_t> shape;
    size_t bytes = 0;
    if (!ConvTransposeOutputShape(param_, &shape) || !CheckedTensorBytes(shape, DataType::FP32, &bytes)) return -1;
    if (!param_.out->IsInitialized() || param_.out->memory_size() < bytes)
        param_.out->ResetBuffer(std::make_shared<Buffer>(bytes), bytes);
    param_.out->Resize(shape);
    param_.out->set_data_type(DataType::FP32);
    param_.out->set_layout(DataLayout::NCHW);
    return 0;
}

void ConvTransposeOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_) kernel_->SetParamOwner(std::make_shared<ConvTransposeParam>(param_));
}

int32_t ConvTransposeOp::Run() {
    return InferOutputShapes() == 0 && kernel_ ? kernel_->compute() : -1;
}
}  // namespace operators
}  // namespace feather
