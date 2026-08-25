#include "src/operator/shape_op.h"

#include <algorithm>
#include <limits>

#include "core/operator_registry.h"
#include "src/operator/tensor_op_utils.h"

namespace feather {
namespace operators {

namespace {

std::shared_ptr<OpBase> BuildShapeOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if (node.inputs.size() != 1 || node.outputs.size() != 1) {
        return nullptr;
    }

    ShapeParam param{};
    param.input = tensors[node.inputs[0]];
    param.out = tensors[node.outputs[0]];
    param.start = tensor_op_detail::GetIntAttribute(node.attributes, "start", 0);
    param.end = tensor_op_detail::GetIntAttribute(node.attributes, "end", std::numeric_limits<int64_t>::max());

    auto op = std::make_shared<ShapeOp>(node.name.empty() ? "Shape" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    kernel::EnsureShapeKernelsRegistered();
    auto kernel = CreateKernelForTensor(context.device, "Shape", {param.out}, DataType::INT64);
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_shape_op_registered = []() {
    OperatorRegistry::instance().Register("Shape", BuildShapeOp);
    return true;
}();

}  // namespace

void EnsureShapeOperatorRegistered() { (void)g_shape_op_registered; }

ShapeOp::ShapeOp(std::string name, const ShapeParam& param) : OpBase(std::move(name), "Shape"), param_(param) {
    SyncIO();
}

void ShapeOp::SyncIO() {
    SetInputs({param_.input});
    SetOutputs({param_.out});
}

int32_t ShapeOp::CheckShape() const {
    return param_.input != nullptr && param_.out != nullptr ? 0 : -1;
}

int32_t ShapeOp::InferOutputShapes() {
    if (CheckShape() != 0) {
        return -1;
    }
    const int64_t rank = static_cast<int64_t>(param_.input->dims().size());
    int64_t start = param_.start < 0 ? param_.start + rank : param_.start;
    int64_t end = param_.end < 0 ? param_.end + rank : param_.end;
    start = std::max<int64_t>(0, std::min<int64_t>(start, rank));
    end = std::max<int64_t>(0, std::min<int64_t>(end, rank));
    if (end < start) {
        return -1;
    }
    param_.out = tensor_op_detail::AllocateOutput(param_.out, {end - start}, DataType::INT64);
    if (param_.out == nullptr) {
        return -1;
    }
    SyncIO();
    return 0;
}

void ShapeOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) {
        kernel_->SetParamOwner(std::make_shared<std::decay_t<decltype(param_)>>(param_));
    }
}

int32_t ShapeOp::Run() {
    if (InferOutputShapes() != 0 || kernel_ == nullptr) {
        return -1;
    }
    return kernel_->compute();
}

}  // namespace operators
}  // namespace feather
