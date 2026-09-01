#include "src/operator/neg_op.h"

#include <utility>

#include "core/operator_registry.h"
#include "src/operator/elementwise_utils.h"
#include "src/operator/elementwise_utils.h"

namespace feather {
namespace operators {
namespace {

std::shared_ptr<OpBase> BuildNegOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if (node.inputs.size() != 1 || node.outputs.size() != 1) return nullptr;
    UnaryParam param{};
    param.input = tensors[node.inputs[0]];
    param.out = tensors[node.outputs[0]];
    auto op = std::make_shared<NegOp>(node.name.empty() ? "neg" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) return nullptr;
    kernel::EnsureNegKernelsRegistered();
    auto kernel = CreateKernelForTensor(context.device, "Neg", {param.input, op->outputs().front()}, DataType::FP32);
    if (kernel == nullptr) return nullptr;
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_neg_op_registered = []() {
    OperatorRegistry::instance().Register("Neg", BuildNegOp);
    return true;
}();

}  // namespace

void EnsureNegOperatorRegistered() { (void)g_neg_op_registered; }

NegOp::NegOp(std::string name, const UnaryParam& param) : OpBase(std::move(name), "Neg"), param_(param) { SyncIO(); }

void NegOp::SyncIO() { SetInputs({param_.input}); SetOutputs({param_.out}); }

int32_t NegOp::CheckShape() const { return elementwise_detail::CheckUnaryParam(param_); }

int32_t NegOp::InferOutputShapes() {
    const auto status = elementwise_detail::InferUnaryOutput(&param_);
    if (status == 0) SyncIO();
    return status;
}

void NegOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) kernel_->SetParamOwner(std::make_shared<std::decay_t<decltype(param_)>>(param_));
}

int32_t NegOp::Run() { return InferOutputShapes() == 0 && kernel_ != nullptr ? kernel_->compute() : -1; }

}  // namespace operators
}  // namespace feather
