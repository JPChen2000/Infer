#include "src/operator/softplus_op.h"

#include <utility>

#include "core/operator_registry.h"
#include "src/operator/elementwise_utils.h"

namespace feather {
namespace operators {
namespace {

std::shared_ptr<OpBase> BuildSoftplusOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors) {
    if (node.inputs.size() != 1 || node.outputs.size() != 1) return nullptr;
    UnaryParam param{};
    param.input = tensors[node.inputs[0]];
    param.out = tensors[node.outputs[0]];
    auto op = std::make_shared<SoftplusOp>(node.name.empty() ? "softplus" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) return nullptr;
    kernel::EnsureSoftplusKernelsRegistered();
    auto kernel = CreateHostKernelForTensor("Softplus", {param.input, op->outputs().front()}, DataType::FP32);
    if (kernel == nullptr) return nullptr;
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_softplus_op_registered = []() {
    OperatorRegistry::instance().Register("Softplus", BuildSoftplusOp);
    return true;
}();

}  // namespace

void EnsureSoftplusOperatorRegistered() { (void)g_softplus_op_registered; }

SoftplusOp::SoftplusOp(std::string name, const UnaryParam& param)
    : OpBase(std::move(name), "Softplus"), param_(param) {
    SyncIO();
}

void SoftplusOp::SyncIO() { SetInputs({param_.input}); SetOutputs({param_.out}); }

int32_t SoftplusOp::CheckShape() const { return elementwise_detail::CheckUnaryParam(param_); }

int32_t SoftplusOp::InferOutputShapes() {
    const auto status = elementwise_detail::InferUnaryOutput(&param_);
    if (status == 0) SyncIO();
    return status;
}

void SoftplusOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) kernel_->SetParam(&param_);
}

int32_t SoftplusOp::Run() { return InferOutputShapes() == 0 && kernel_ != nullptr ? kernel_->compute() : -1; }

}  // namespace operators
}  // namespace feather
