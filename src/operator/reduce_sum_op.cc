#include "src/operator/reduce_sum_op.h"

#include <algorithm>
#include <numeric>
#include <set>
#include <utility>

#include "core/operator_registry.h"
#include "src/operator/tensor_op_utils.h"

namespace feather {
namespace operators {
namespace {

std::vector<int64_t> ResolveAxes(const ReduceSumParam& param) {
    const auto rank = static_cast<int64_t>(param.input->dims().size());
    auto axes = tensor_op_detail::NormalizeAxes(param.axes, rank, false);
    if (axes.empty()) {
        axes.resize(static_cast<size_t>(rank));
        std::iota(axes.begin(), axes.end(), 0);
    }
    return axes;
}

std::shared_ptr<OpBase> BuildReduceSumOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if (node.inputs.size() != 1 || node.outputs.size() != 1) return nullptr;
    ReduceSumParam param{};
    param.input = tensors[node.inputs[0]];
    param.out = tensors[node.outputs[0]];
    param.axes = tensor_op_detail::GetIntVectorAttribute(node.attributes, "axes");
    param.keepdims = tensor_op_detail::GetIntAttribute(node.attributes, "keepdims", 1) != 0;
    auto op = std::make_shared<ReduceSumOp>(node.name.empty() ? "reduce_sum" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) return nullptr;
    kernel::EnsureReduceSumKernelsRegistered();
    auto kernel = CreateKernelForTensor(context.device, "ReduceSum", {param.input, op->outputs().front()}, DataType::FP32);
    if (kernel == nullptr) return nullptr;
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_reduce_sum_op_registered = []() {
    OperatorRegistry::instance().Register("ReduceSum", BuildReduceSumOp);
    return true;
}();

}  // namespace

void EnsureReduceSumOperatorRegistered() { (void)g_reduce_sum_op_registered; }

ReduceSumOp::ReduceSumOp(std::string name, const ReduceSumParam& param)
    : OpBase(std::move(name), "ReduceSum"), param_(param) {
    SyncIO();
}

void ReduceSumOp::SyncIO() { SetInputs({param_.input}); SetOutputs({param_.out}); }

int32_t ReduceSumOp::CheckShape() const {
    if (param_.input == nullptr || param_.out == nullptr ||
        !tensor_op_detail::IsFloatingPointDataType(param_.input->data_type())) {
        return -1;
    }
    const auto axes = ResolveAxes(param_);
    const auto rank = static_cast<int64_t>(param_.input->dims().size());
    return std::all_of(axes.begin(), axes.end(), [rank](int64_t axis) { return axis >= 0 && axis < rank; }) &&
                   std::set<int64_t>(axes.begin(), axes.end()).size() == axes.size()
               ? 0
               : -1;
}

int32_t ReduceSumOp::InferOutputShapes() {
    if (CheckShape() != 0) return -1;
    const auto axes = ResolveAxes(param_);
    const std::set<int64_t> axis_set(axes.begin(), axes.end());
    const auto input_shape = param_.input->dims().data();
    std::vector<int64_t> output_shape;
    output_shape.reserve(input_shape.size());
    for (int64_t axis = 0; axis < static_cast<int64_t>(input_shape.size()); ++axis) {
        if (axis_set.count(axis) != 0) {
            if (param_.keepdims) output_shape.push_back(1);
        } else {
            output_shape.push_back(input_shape[static_cast<size_t>(axis)]);
        }
    }
    if (output_shape.empty()) output_shape = {1};
    param_.out = tensor_op_detail::AllocateOutput(param_.out, output_shape, param_.input->data_type());
    if (param_.out == nullptr) return -1;
    param_.out->set_layout(param_.input->layout());
    SyncIO();
    return 0;
}

void ReduceSumOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) kernel_->SetParamOwner(std::make_shared<std::decay_t<decltype(param_)>>(param_));
}

int32_t ReduceSumOp::Run() { return InferOutputShapes() == 0 && kernel_ != nullptr ? kernel_->compute() : -1; }

}  // namespace operators
}  // namespace feather
