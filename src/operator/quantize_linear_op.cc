#include "src/operator/quantize_linear_op.h"

#include <utility>

#include "core/operator_registry.h"
#include "src/operator/tensor_op_utils.h"

namespace feather {
namespace operators {
namespace {

std::shared_ptr<OpBase> BuildQuantizeLinearOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors,
                                              const OperatorRegistry::BuildContext& context) {
    if (node.inputs.size() != 3 || node.outputs.size() != 1) return nullptr;
    QuantizeLinearParam param{};
    param.input = tensors[node.inputs[0]];
    param.scale = tensors[node.inputs[1]];
    param.zero_point = tensors[node.inputs[2]];
    param.out = tensors[node.outputs[0]];
    param.axis = tensor_op_detail::GetIntAttribute(node.attributes, "axis", -1);
    auto op = std::make_shared<QuantizeLinearOp>(node.name.empty() ? "QuantizeLinear" : node.name, param);
    op->SetExecutionDevice(context.device);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) return nullptr;
    kernel::EnsureQuantizeLinearKernelsRegistered();
    auto kernel = CreateKernelForTensor(context.device, "QuantizeLinear", {param.input}, param.input->data_type());
    if (kernel == nullptr) return nullptr;
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_quantize_linear_op_registered = []() {
    OperatorRegistry::instance().Register("QuantizeLinear", BuildQuantizeLinearOp);
    return true;
}();

}  // namespace

void EnsureQuantizeLinearOperatorRegistered() { (void)g_quantize_linear_op_registered; }

QuantizeLinearOp::QuantizeLinearOp(std::string name, const QuantizeLinearParam& param)
    : OpBase(std::move(name), "QuantizeLinear"), param_(param) {
    SyncIO();
}

void QuantizeLinearOp::SyncIO() {
    SetInputs({param_.input, param_.scale, param_.zero_point});
    SetOutputs({param_.out});
}

int32_t QuantizeLinearOp::CheckShape() const {
    if (param_.input == nullptr || param_.scale == nullptr || param_.zero_point == nullptr || param_.out == nullptr ||
        !param_.input->IsInitialized() ||
        (param_.input->data_type() != DataType::FP32 && param_.input->data_type() != DataType::FP16 &&
         param_.input->data_type() != DataType::BF16)) return -1;
    return 0;
}

int32_t QuantizeLinearOp::InferOutputShapes() {
    if (CheckShape() != 0 || param_.input->numel() <= 0) return -1;
    const size_t required_bytes = static_cast<size_t>(param_.input->numel());
    if (!param_.out->IsInitialized() || param_.out->memory_size() < required_bytes) {
        param_.out = std::make_shared<Tensor>(required_bytes);
    }
    param_.out->Resize(param_.input->dims().data());
    param_.out->set_data_type(DataType::INT8);
    param_.out->set_layout(param_.input->layout());
    SyncIO();
    return 0;
}

void QuantizeLinearOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) kernel_->SetParamOwner(std::make_shared<QuantizeLinearParam>(param_));
}

int32_t QuantizeLinearOp::Run() {
    if (InferOutputShapes() != 0 || kernel_ == nullptr) return -1;
    RefreshKernelParams();
    return kernel_->compute();
}

void QuantizeLinearOp::RefreshKernelParams() {
    if (kernel_ != nullptr) kernel_->SetParamOwner(std::make_shared<QuantizeLinearParam>(param_));
}

}  // namespace operators
}  // namespace feather
