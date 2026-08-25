#include "src/operator/gather_op.h"

#include <functional>
#include <numeric>

#include "core/operator_registry.h"
#include "src/operator/tensor_op_utils.h"

namespace feather {
namespace operators {

namespace {

std::shared_ptr<OpBase> BuildGatherOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if (node.inputs.size() != 2 || node.outputs.size() != 1) {
        return nullptr;
    }

    GatherParam param{};
    param.data = tensors[node.inputs[0]];
    param.indices = tensors[node.inputs[1]];
    param.out = tensors[node.outputs[0]];
    param.axis = static_cast<int32_t>(tensor_op_detail::GetIntAttribute(node.attributes, "axis", 0));

    auto op = std::make_shared<GatherOp>(node.name.empty() ? "Gather" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    kernel::EnsureGatherKernelsRegistered();
    auto kernel = CreateKernelForTensor(context.device, "Gather", {param.data}, DataType::FP32);
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_gather_op_registered = []() {
    OperatorRegistry::instance().Register("Gather", BuildGatherOp);
    return true;
}();

}  // namespace

void EnsureGatherOperatorRegistered() { (void)g_gather_op_registered; }

GatherOp::GatherOp(std::string name, const GatherParam& param) : OpBase(std::move(name), "Gather"), param_(param) {
    SyncIO();
}

void GatherOp::SyncIO() {
    SetInputs({param_.data, param_.indices});
    SetOutputs({param_.out});
}

int32_t GatherOp::CheckShape() const {
    if (param_.data == nullptr || param_.indices == nullptr || param_.out == nullptr ||
        !tensor_op_detail::IsFloatingPointDataType(param_.data->data_type()) ||
        (param_.indices->data_type() != DataType::INT32 && param_.indices->data_type() != DataType::INT64)) {
        return -1;
    }
    const int32_t rank = static_cast<int32_t>(param_.data->dims().size());
    const int32_t axis = param_.axis < 0 ? param_.axis + rank : param_.axis;
    return axis >= 0 && axis < rank ? 0 : -1;
}

int32_t GatherOp::InferOutputShapes() {
    if (CheckShape() != 0) {
        return -1;
    }
    const auto data_shape = param_.data->dims().data();
    const auto indices_shape = param_.indices->dims().data();
    const int32_t rank = static_cast<int32_t>(data_shape.size());
    const int32_t axis = param_.axis < 0 ? param_.axis + rank : param_.axis;
    std::vector<int64_t> out_shape;
    out_shape.insert(out_shape.end(), data_shape.begin(), data_shape.begin() + axis);
    out_shape.insert(out_shape.end(), indices_shape.begin(), indices_shape.end());
    out_shape.insert(out_shape.end(), data_shape.begin() + axis + 1, data_shape.end());
    if (out_shape.empty()) {
        out_shape = {1};
    }
    param_.out = tensor_op_detail::AllocateOutput(param_.out, out_shape, param_.data->data_type());
    if (param_.out == nullptr) {
        return -1;
    }
    param_.out->set_layout(param_.data->layout());
    SyncIO();
    return 0;
}

void GatherOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) {
        kernel_->SetParamOwner(std::make_shared<std::decay_t<decltype(param_)>>(param_));
    }
}

int32_t GatherOp::Run() {
    if (InferOutputShapes() != 0 || kernel_ == nullptr) {
        return -1;
    }
    return kernel_->compute();
}

}  // namespace operators
}  // namespace feather
