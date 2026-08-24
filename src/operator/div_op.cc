#include "src/operator/div_op.h"

#include <utility>

#include "core/operator_registry.h"
#include "src/operator/elementwise_utils.h"

namespace feather {
namespace operators {

namespace {

std::shared_ptr<OpBase> BuildDivOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if (node.inputs.size() != 2 || node.outputs.size() != 1) {
        return nullptr;
    }
    BinaryParam param{};
    param.lhs = tensors[node.inputs[0]];
    param.rhs = tensors[node.inputs[1]];
    param.out = tensors[node.outputs[0]];

    auto op = std::make_shared<DivOp>(node.name.empty() ? "div" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    kernel::EnsureDivKernelsRegistered();
    auto kernel = CreateKernelForTensor(context.device, "Div", {param.lhs, param.rhs, op->outputs().front()});
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_div_op_registered = []() {
    OperatorRegistry::instance().Register("Div", BuildDivOp);
    return true;
}();

}  // namespace

void EnsureDivOperatorRegistered() { (void)g_div_op_registered; }

DivOp::DivOp(std::string name, const BinaryParam& param) : OpBase(std::move(name), "Div"), param_(param) { SyncIO(); }

void DivOp::SyncIO() {
    SetInputs({param_.lhs, param_.rhs});
    SetOutputs({param_.out});
}

int32_t DivOp::CheckShape() const { return elementwise_detail::CheckBinaryParam(param_); }

int32_t DivOp::InferOutputShapes() {
    const auto status = elementwise_detail::InferBinaryOutput(&param_);
    if (status == 0) {
        SyncIO();
    }
    return status;
}

void DivOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) {
        kernel_->SetParam(&param_);
    }
}

int32_t DivOp::Run() {
    if (InferOutputShapes() != 0 || kernel_ == nullptr) {
        return -1;
    }
    return kernel_->compute();
}

}  // namespace operators
}  // namespace feather
