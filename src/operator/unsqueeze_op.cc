#include "src/operator/unsqueeze_op.h"

#include "core/operator_registry.h"
#include "src/operator/tensor_op_utils.h"

namespace feather {
namespace operators {

namespace {

std::shared_ptr<OpBase> BuildUnsqueezeOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if ((node.inputs.size() != 1 && node.inputs.size() != 2) || node.outputs.size() != 1) {
        return nullptr;
    }

    AxesParam param{};
    param.input = tensors[node.inputs[0]];
    if (node.inputs.size() == 2) {
        param.axes_tensor = tensors[node.inputs[1]];
    }
    param.out = tensors[node.outputs[0]];
    param.axes = tensor_op_detail::GetIntVectorAttribute(node.attributes, "axes");
    if (param.input == nullptr || param.out == nullptr ||
        (param.axes_tensor == nullptr && param.axes.empty())) {
        return nullptr;
    }

    auto op = std::make_shared<UnsqueezeOp>(node.name.empty() ? "Unsqueeze" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    kernel::EnsureUnsqueezeKernelsRegistered();
    auto kernel = CreateKernelForTensor(context.device, "Unsqueeze", {param.input, param.out}, DataType::FP32);
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_unsqueeze_op_registered = []() {
    OperatorRegistry::instance().Register("Unsqueeze", BuildUnsqueezeOp);
    return true;
}();

}  // namespace

void EnsureUnsqueezeOperatorRegistered() { (void)g_unsqueeze_op_registered; }

UnsqueezeOp::UnsqueezeOp(std::string name, const AxesParam& param)
    : OpBase(std::move(name), "Unsqueeze"), param_(param) {
    SyncIO();
}

void UnsqueezeOp::SyncIO() {
    std::vector<std::shared_ptr<Tensor>> inputs = {param_.input};
    if (param_.axes_tensor != nullptr) {
        inputs.push_back(param_.axes_tensor);
    }
    SetInputs(std::move(inputs));
    SetOutputs({param_.out});
}

int32_t UnsqueezeOp::CheckShape() const {
    return param_.input != nullptr && param_.out != nullptr ? 0 : -1;
}

int32_t UnsqueezeOp::InferOutputShapes() {
    if (CheckShape() != 0 || tensor_op_detail::InferAxesOutputShape(&param_, true) != 0) {
        return -1;
    }
    SyncIO();
    return 0;
}

void UnsqueezeOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) {
        kernel_->SetParam(static_cast<void*>(&param_));
    }
}

int32_t UnsqueezeOp::Run() {
    if (InferOutputShapes() != 0 || kernel_ == nullptr) {
        return -1;
    }
    return kernel_->compute();
}

}  // namespace operators
}  // namespace feather
