#include "src/operator/sin_op.h"

#include <utility>

#include "core/operator_registry.h"
#include "src/operator/elementwise_utils.h"

namespace feather {
namespace operators {
namespace {

std::shared_ptr<OpBase> BuildSinOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if (node.inputs.size() != 1 || node.outputs.size() != 1) return nullptr;
    UnaryParam param{};
    param.input = tensors[node.inputs[0]];
    param.out = tensors[node.outputs[0]];
    auto op = std::make_shared<SinOp>(node.name.empty() ? "sin" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) return nullptr;
    kernel::EnsureSinKernelsRegistered();
    auto kernel = CreateKernelForTensor(context.device, "Sin", {param.input, op->outputs().front()}, DataType::FP32);
    if (kernel == nullptr) return nullptr;
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_sin_op_registered = []() {
    OperatorRegistry::instance().Register("Sin", BuildSinOp);
    return true;
}();

}  // namespace

void EnsureSinOperatorRegistered() { (void)g_sin_op_registered; }

SinOp::SinOp(std::string name, const UnaryParam& param) : OpBase(std::move(name), "Sin"), param_(param) { SyncIO(); }

void SinOp::SyncIO() { SetInputs({param_.input}); SetOutputs({param_.out}); }

int32_t SinOp::CheckShape() const { return elementwise_detail::CheckUnaryParam(param_); }

int32_t SinOp::InferOutputShapes() {
    const auto status = elementwise_detail::InferUnaryOutput(&param_);
    if (status == 0) SyncIO();
    return status;
}

void SinOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) kernel_->SetParamOwner(std::make_shared<std::decay_t<decltype(param_)>>(param_));
}

int32_t SinOp::Run() { return InferOutputShapes() == 0 && kernel_ != nullptr ? kernel_->compute() : -1; }

}  // namespace operators
}  // namespace feather
