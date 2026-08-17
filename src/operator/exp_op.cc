#include "src/operator/exp_op.h"

#include <utility>

#include "core/operator_registry.h"
#include "src/operator/elementwise_utils.h"

namespace feather {
namespace operators {
namespace {

std::shared_ptr<OpBase> BuildExpOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors) {
    if (node.inputs.size() != 1 || node.outputs.size() != 1) return nullptr;
    UnaryParam param{};
    param.input = tensors[node.inputs[0]];
    param.out = tensors[node.outputs[0]];
    auto op = std::make_shared<ExpOp>(node.name.empty() ? "exp" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) return nullptr;
    kernel::EnsureExpKernelsRegistered();
    auto kernel = CreateHostKernelForTensor("Exp", {param.input, op->outputs().front()}, DataType::FP32);
    if (kernel == nullptr) return nullptr;
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_exp_op_registered = []() {
    OperatorRegistry::instance().Register("Exp", BuildExpOp);
    return true;
}();

}  // namespace

void EnsureExpOperatorRegistered() { (void)g_exp_op_registered; }

ExpOp::ExpOp(std::string name, const UnaryParam& param) : OpBase(std::move(name), "Exp"), param_(param) { SyncIO(); }

void ExpOp::SyncIO() { SetInputs({param_.input}); SetOutputs({param_.out}); }

int32_t ExpOp::CheckShape() const { return elementwise_detail::CheckUnaryParam(param_); }

int32_t ExpOp::InferOutputShapes() {
    const auto status = elementwise_detail::InferUnaryOutput(&param_);
    if (status == 0) SyncIO();
    return status;
}

void ExpOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) kernel_->SetParam(&param_);
}

int32_t ExpOp::Run() { return InferOutputShapes() == 0 && kernel_ != nullptr ? kernel_->compute() : -1; }

}  // namespace operators
}  // namespace feather
