#include "src/operator/erf_op.h"

#include <utility>

#include "core/operator_registry.h"
#include "src/operator/elementwise_utils.h"
#include "src/operator/elementwise_utils.h"

namespace feather {
namespace operators {

namespace {

std::shared_ptr<OpBase> BuildErfOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if (node.inputs.size() != 1 || node.outputs.size() != 1) {
        return nullptr;
    }
    UnaryParam param{};
    param.input = tensors[node.inputs[0]];
    param.out = tensors[node.outputs[0]];

    auto op = std::make_shared<ErfOp>(node.name.empty() ? "erf" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    kernel::EnsureErfKernelsRegistered();
    auto kernel = CreateKernelForTensor(context.device, "Erf", {param.input, op->outputs().front()}, DataType::FP32);
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_erf_op_registered = []() {
    OperatorRegistry::instance().Register("Erf", BuildErfOp);
    return true;
}();

}  // namespace

void EnsureErfOperatorRegistered() { (void)g_erf_op_registered; }

ErfOp::ErfOp(std::string name, const UnaryParam& param) : OpBase(std::move(name), "Erf"), param_(param) { SyncIO(); }

void ErfOp::SyncIO() {
    SetInputs({param_.input});
    SetOutputs({param_.out});
}

int32_t ErfOp::CheckShape() const { return elementwise_detail::CheckUnaryParam(param_); }

int32_t ErfOp::InferOutputShapes() {
    const auto status = elementwise_detail::InferUnaryOutput(&param_);
    if (status == 0) {
        SyncIO();
    }
    return status;
}

void ErfOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) {
        kernel_->SetParamOwner(std::make_shared<std::decay_t<decltype(param_)>>(param_));
    }
}

int32_t ErfOp::Run() {
    if (InferOutputShapes() != 0 || kernel_ == nullptr) {
        return -1;
    }
    return kernel_->compute();
}

}  // namespace operators
}  // namespace feather
