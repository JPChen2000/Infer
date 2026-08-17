#include "src/operator/where_op.h"

#include "core/operator_registry.h"
#include "src/operator/tensor_op_utils.h"

namespace feather {
namespace operators {

namespace {

std::shared_ptr<OpBase> BuildWhereOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors) {
    if (node.inputs.size() != 3 || node.outputs.size() != 1) {
        return nullptr;
    }

    WhereParam param{};
    param.condition = tensors[node.inputs[0]];
    param.x = tensors[node.inputs[1]];
    param.y = tensors[node.inputs[2]];
    param.out = tensors[node.outputs[0]];

    auto op = std::make_shared<WhereOp>(node.name.empty() ? "Where" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    kernel::EnsureWhereKernelsRegistered();
    auto kernel = CreateHostKernelForTensor("Where", {param.x}, DataType::FP32);
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_where_op_registered = []() {
    OperatorRegistry::instance().Register("Where", BuildWhereOp);
    return true;
}();

}  // namespace

void EnsureWhereOperatorRegistered() { (void)g_where_op_registered; }

WhereOp::WhereOp(std::string name, const WhereParam& param) : OpBase(std::move(name), "Where"), param_(param) {
    SyncIO();
}

void WhereOp::SyncIO() {
    SetInputs({param_.condition, param_.x, param_.y});
    SetOutputs({param_.out});
}

int32_t WhereOp::CheckShape() const {
    if (param_.condition == nullptr || param_.x == nullptr || param_.y == nullptr || param_.out == nullptr ||
        param_.condition->data_type() != DataType::BOOL || param_.x->data_type() != param_.y->data_type()) {
        return -1;
    }
    std::vector<int64_t> shape;
    if (!tensor_op_detail::InferBroadcastShape(param_.condition->dims().data(), param_.x->dims().data(), &shape)) {
        return -1;
    }
    return tensor_op_detail::InferBroadcastShape(shape, param_.y->dims().data(), &shape) ? 0 : -1;
}

int32_t WhereOp::InferOutputShapes() {
    if (CheckShape() != 0) {
        return -1;
    }
    std::vector<int64_t> shape;
    if (!tensor_op_detail::InferBroadcastShape(param_.condition->dims().data(), param_.x->dims().data(), &shape) ||
        !tensor_op_detail::InferBroadcastShape(shape, param_.y->dims().data(), &shape)) {
        return -1;
    }
    param_.out = tensor_op_detail::AllocateOutput(param_.out, shape, param_.x->data_type());
    if (param_.out == nullptr) {
        return -1;
    }
    param_.out->set_layout(param_.x->layout());
    SyncIO();
    return 0;
}

void WhereOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) {
        kernel_->SetParam(static_cast<void*>(&param_));
    }
}

int32_t WhereOp::Run() {
    if (InferOutputShapes() != 0 || kernel_ == nullptr) {
        return -1;
    }
    return kernel_->compute();
}

}  // namespace operators
}  // namespace feather
