#include "src/operator/dequantize_linear_op.h"

#include <utility>

#include "core/operator_registry.h"
#include "src/operator/tensor_op_utils.h"

namespace feather {
namespace operators {
namespace {

DataType ResolveDequantizeDataType(int64_t onnx_type) {
    switch (onnx_type) {
        case 1: return DataType::FP32;
        case 10: return DataType::FP16;
        case 16: return DataType::BF16;
        default: return DataType::UNKNOWN;
    }
}

std::shared_ptr<OpBase> BuildDequantizeLinearOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors,
                                                const OperatorRegistry::BuildContext& context) {
    if (node.inputs.size() != 3 || node.outputs.size() != 1) return nullptr;
    DequantizeLinearParam param{};
    param.input = tensors[node.inputs[0]];
    param.scale = tensors[node.inputs[1]];
    param.zero_point = tensors[node.inputs[2]];
    param.out = tensors[node.outputs[0]];
    param.axis = tensor_op_detail::GetIntAttribute(node.attributes, "axis", -1);
    param.to = ResolveDequantizeDataType(tensor_op_detail::GetIntAttribute(node.attributes, "to", 1));
    if (param.to == DataType::UNKNOWN) return nullptr;
    auto op = std::make_shared<DequantizeLinearOp>(node.name.empty() ? "DequantizeLinear" : node.name, param);
    op->SetExecutionDevice(context.device);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) return nullptr;
    kernel::EnsureDequantizeLinearKernelsRegistered();
    auto kernel = CreateKernelForTensor(context.device, "DequantizeLinear", {param.input}, param.input->data_type());
    if (kernel == nullptr) return nullptr;
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_dequantize_linear_op_registered = []() {
    OperatorRegistry::instance().Register("DequantizeLinear", BuildDequantizeLinearOp);
    return true;
}();

}  // namespace

void EnsureDequantizeLinearOperatorRegistered() { (void)g_dequantize_linear_op_registered; }

DequantizeLinearOp::DequantizeLinearOp(std::string name, const DequantizeLinearParam& param)
    : OpBase(std::move(name), "DequantizeLinear"), param_(param) {
    SyncIO();
}

void DequantizeLinearOp::SyncIO() {
    SetInputs({param_.input, param_.scale, param_.zero_point});
    SetOutputs({param_.out});
}

int32_t DequantizeLinearOp::CheckShape() const {
    if (param_.input == nullptr || param_.scale == nullptr || param_.zero_point == nullptr || param_.out == nullptr ||
        !param_.input->IsInitialized() || param_.input->data_type() != DataType::INT8 ||
        (param_.to != DataType::FP32 && param_.to != DataType::FP16 && param_.to != DataType::BF16)) return -1;
    return 0;
}

int32_t DequantizeLinearOp::InferOutputShapes() {
    if (CheckShape() != 0 || param_.input->numel() <= 0) return -1;
    const size_t required_bytes = static_cast<size_t>(param_.input->numel()) * DataTypeBytes(param_.to);
    if (!param_.out->IsInitialized() || param_.out->memory_size() < required_bytes) {
        param_.out = std::make_shared<Tensor>(required_bytes);
    }
    param_.out->Resize(param_.input->dims().data());
    param_.out->set_data_type(param_.to);
    param_.out->set_layout(param_.input->layout());
    param_.out->set_quantization(QuantizationParams{});
    SyncIO();
    return 0;
}

void DequantizeLinearOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) kernel_->SetParamOwner(std::make_shared<DequantizeLinearParam>(param_));
}

int32_t DequantizeLinearOp::Run() {
    if (InferOutputShapes() != 0 || kernel_ == nullptr) return -1;
    RefreshKernelParams();
    return kernel_->compute();
}

void DequantizeLinearOp::RefreshKernelParams() {
    if (kernel_ != nullptr) kernel_->SetParamOwner(std::make_shared<DequantizeLinearParam>(param_));
}

}  // namespace operators
}  // namespace feather
