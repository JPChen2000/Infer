#include "src/operator/expand_op.h"

#include "core/operator_registry.h"
#include "src/operator/tensor_op_utils.h"

namespace feather {
namespace operators {

namespace {

bool InferExpandedShape(const std::vector<int64_t>& input_shape, const std::vector<int64_t>& target_shape,
                        std::vector<int64_t>* output_shape) {
    if (output_shape == nullptr || target_shape.size() < input_shape.size()) {
        return false;
    }
    output_shape->assign(target_shape.begin(), target_shape.end());
    const size_t rank_gap = target_shape.size() - input_shape.size();
    for (size_t i = 0; i < target_shape.size(); ++i) {
        const int64_t input_dim = i < rank_gap ? 1 : input_shape[i - rank_gap];
        const int64_t target_dim = target_shape[i];
        if (input_dim <= 0 || target_dim <= 0 ||
            (input_dim != target_dim && input_dim != 1 && target_dim != 1)) {
            return false;
        }
        (*output_shape)[i] = std::max(input_dim, target_dim);
    }
    return true;
}

std::shared_ptr<OpBase> BuildExpandOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if (node.inputs.size() != 2 || node.outputs.size() != 1) {
        return nullptr;
    }

    ExpandParam param{};
    param.input = tensors[node.inputs[0]];
    param.shape = tensors[node.inputs[1]];
    param.out = tensors[node.outputs[0]];

    auto op = std::make_shared<ExpandOp>(node.name.empty() ? "Expand" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    kernel::EnsureExpandKernelsRegistered();
    auto kernel = CreateKernelForTensor(context.device, "Expand", {param.input}, DataType::FP32);
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_expand_op_registered = []() {
    OperatorRegistry::instance().Register("Expand", BuildExpandOp);
    return true;
}();

}  // namespace

void EnsureExpandOperatorRegistered() { (void)g_expand_op_registered; }

ExpandOp::ExpandOp(std::string name, const ExpandParam& param) : OpBase(std::move(name), "Expand"), param_(param) {
    SyncIO();
}

void ExpandOp::SyncIO() {
    SetInputs({param_.input, param_.shape});
    SetOutputs({param_.out});
}

int32_t ExpandOp::CheckShape() const {
    if (param_.input == nullptr || param_.shape == nullptr || param_.out == nullptr) {
        return -1;
    }
    std::vector<int64_t> target_shape;
    return tensor_op_detail::ReadShapeValues(param_.shape, &target_shape) ? 0 : -1;
}

int32_t ExpandOp::InferOutputShapes() {
    if (CheckShape() != 0) {
        return -1;
    }
    std::vector<int64_t> target_shape;
    if (!tensor_op_detail::ReadShapeValues(param_.shape, &target_shape) || target_shape.empty()) {
        return -1;
    }
    std::vector<int64_t> output_shape;
    if (!InferExpandedShape(param_.input->dims().data(), target_shape, &output_shape)) {
        return -1;
    }
    param_.out = tensor_op_detail::AllocateOutput(param_.out, output_shape, param_.input->data_type());
    if (param_.out == nullptr) {
        return -1;
    }
    param_.out->set_layout(param_.input->layout());
    SyncIO();
    return 0;
}

void ExpandOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) {
        kernel_->SetParamOwner(std::make_shared<std::decay_t<decltype(param_)>>(param_));
    }
}

int32_t ExpandOp::Run() {
    if (InferOutputShapes() != 0 || kernel_ == nullptr) {
        return -1;
    }
    return kernel_->compute();
}

}  // namespace operators
}  // namespace feather
