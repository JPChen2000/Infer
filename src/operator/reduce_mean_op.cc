#include "src/operator/reduce_mean_op.h"

#include <functional>
#include <numeric>
#include <set>

#include "core/operator_registry.h"
#include "src/operator/tensor_op_utils.h"

namespace feather {
namespace operators {

namespace {

std::shared_ptr<OpBase> BuildReduceMeanOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if (node.inputs.size() != 1 || node.outputs.size() != 1) {
        return nullptr;
    }

    ReduceMeanParam param{};
    param.input = tensors[node.inputs[0]];
    param.out = tensors[node.outputs[0]];
    param.axes = tensor_op_detail::GetIntVectorAttribute(node.attributes, "axes");
    param.keepdims = tensor_op_detail::GetIntAttribute(node.attributes, "keepdims", 1) != 0;

    auto op = std::make_shared<ReduceMeanOp>(node.name.empty() ? "ReduceMean" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    kernel::EnsureReduceMeanKernelsRegistered();
    auto kernel = CreateKernelForTensor(context.device, "ReduceMean", {param.input}, DataType::FP32);
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_reduce_mean_op_registered = []() {
    OperatorRegistry::instance().Register("ReduceMean", BuildReduceMeanOp);
    return true;
}();

}  // namespace

void EnsureReduceMeanOperatorRegistered() { (void)g_reduce_mean_op_registered; }

ReduceMeanOp::ReduceMeanOp(std::string name, const ReduceMeanParam& param)
    : OpBase(std::move(name), "ReduceMean"), param_(param) {
    SyncIO();
}

void ReduceMeanOp::SyncIO() {
    SetInputs({param_.input});
    SetOutputs({param_.out});
}

int32_t ReduceMeanOp::CheckShape() const {
    return param_.input != nullptr && param_.out != nullptr &&
                   tensor_op_detail::IsFloatingPointDataType(param_.input->data_type())
               ? 0
               : -1;
}

int32_t ReduceMeanOp::InferOutputShapes() {
    if (CheckShape() != 0) {
        return -1;
    }
    const auto input_shape = param_.input->dims().data();
    auto axes = tensor_op_detail::NormalizeAxes(param_.axes, static_cast<int64_t>(input_shape.size()), false);
    if (axes.empty()) {
        axes.resize(input_shape.size());
        std::iota(axes.begin(), axes.end(), 0);
    }
    if (std::any_of(axes.begin(), axes.end(), [&input_shape](int64_t axis) {
            return axis < 0 || axis >= static_cast<int64_t>(input_shape.size());
        }) || std::set<int64_t>(axes.begin(), axes.end()).size() != axes.size()) {
        return -1;
    }

    std::set<int64_t> axis_set(axes.begin(), axes.end());
    std::vector<int64_t> out_shape;
    out_shape.reserve(input_shape.size());
    for (int64_t axis = 0; axis < static_cast<int64_t>(input_shape.size()); ++axis) {
        if (axis_set.count(axis) != 0) {
            if (param_.keepdims) {
                out_shape.push_back(1);
            }
        } else {
            out_shape.push_back(input_shape[static_cast<size_t>(axis)]);
        }
    }
    if (out_shape.empty()) {
        out_shape = {1};
    }
    param_.out = tensor_op_detail::AllocateOutput(param_.out, out_shape, param_.input->data_type());
    if (param_.out == nullptr) {
        return -1;
    }
    param_.out->set_layout(param_.input->layout());
    SyncIO();
    return 0;
}

void ReduceMeanOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) {
        kernel_->SetParamOwner(std::make_shared<std::decay_t<decltype(param_)>>(param_));
    }
}

int32_t ReduceMeanOp::Run() {
    if (InferOutputShapes() != 0 || kernel_ == nullptr) {
        return -1;
    }
    return kernel_->compute();
}

}  // namespace operators
}  // namespace feather
