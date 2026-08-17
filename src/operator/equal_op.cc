#include "src/operator/equal_op.h"

#include "core/operator_registry.h"
#include "src/operator/tensor_op_utils.h"

namespace feather {
namespace operators {

namespace {

std::shared_ptr<OpBase> BuildEqualOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors) {
    if (node.inputs.size() != 2 || node.outputs.size() != 1) {
        return nullptr;
    }

    EqualParam param{};
    param.lhs = tensors[node.inputs[0]];
    param.rhs = tensors[node.inputs[1]];
    param.out = tensors[node.outputs[0]];

    auto op = std::make_shared<EqualOp>(node.name.empty() ? "Equal" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    kernel::EnsureEqualKernelsRegistered();
    auto kernel = CreateHostKernelForTensor("Equal", {param.lhs, param.rhs}, DataType::FP32);
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_equal_op_registered = []() {
    OperatorRegistry::instance().Register("Equal", BuildEqualOp);
    return true;
}();

}  // namespace

void EnsureEqualOperatorRegistered() { (void)g_equal_op_registered; }

EqualOp::EqualOp(std::string name, const EqualParam& param) : OpBase(std::move(name), "Equal"), param_(param) {
    SyncIO();
}

void EqualOp::SyncIO() {
    SetInputs({param_.lhs, param_.rhs});
    SetOutputs({param_.out});
}

int32_t EqualOp::CheckShape() const {
    if (param_.lhs == nullptr || param_.rhs == nullptr || param_.out == nullptr ||
        param_.lhs->data_type() != param_.rhs->data_type()) {
        return -1;
    }
    std::vector<int64_t> shape;
    return tensor_op_detail::InferBroadcastShape(param_.lhs->dims().data(), param_.rhs->dims().data(), &shape) ? 0 : -1;
}

int32_t EqualOp::InferOutputShapes() {
    if (CheckShape() != 0) {
        return -1;
    }
    std::vector<int64_t> shape;
    if (!tensor_op_detail::InferBroadcastShape(param_.lhs->dims().data(), param_.rhs->dims().data(), &shape)) {
        return -1;
    }
    param_.out = tensor_op_detail::AllocateOutput(param_.out, shape, DataType::BOOL);
    if (param_.out == nullptr) {
        return -1;
    }
    SyncIO();
    return 0;
}

void EqualOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) {
        kernel_->SetParam(static_cast<void*>(&param_));
    }
}

int32_t EqualOp::Run() {
    if (InferOutputShapes() != 0 || kernel_ == nullptr) {
        return -1;
    }
    return kernel_->compute();
}

}  // namespace operators
}  // namespace feather
