#include "src/operator/constant_of_shape_op.h"

#include <variant>

#include "core/operator_registry.h"
#include "src/operator/control_tensor.h"
#include "src/operator/tensor_op_utils.h"

namespace feather {
namespace operators {

namespace {

std::shared_ptr<OpBase> BuildConstantOfShapeOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if (node.inputs.size() != 1 || node.outputs.size() != 1) {
        return nullptr;
    }

    ConstantOfShapeParam param{};
    param.shape = tensors[node.inputs[0]];
    param.out = tensors[node.outputs[0]];
    if (param.out == nullptr) {
        return nullptr;
    }
    param.output_type = param.out->data_type();
    const auto float_it = node.attributes.find("value_float");
    if (float_it != node.attributes.end()) {
        const auto* value = std::get_if<float>(&float_it->second);
        if (value == nullptr) {
            return nullptr;
        }
        param.float_value = *value;
        param.use_float_value = true;
    } else {
        param.int_value = tensor_op_detail::GetIntAttribute(node.attributes, "value_int", 0);
    }

    auto op = std::make_shared<ConstantOfShapeOp>(node.name.empty() ? "ConstantOfShape" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    kernel::EnsureConstantOfShapeKernelsRegistered();
    auto kernel = CreateKernelForTensor(DeviceType::COMMON, "ConstantOfShape", {param.out}, param.output_type);
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_constant_of_shape_op_registered = []() {
    OperatorRegistry::instance().Register("ConstantOfShape", BuildConstantOfShapeOp);
    return true;
}();

}  // namespace

void EnsureConstantOfShapeOperatorRegistered() { (void)g_constant_of_shape_op_registered; }

ConstantOfShapeOp::ConstantOfShapeOp(std::string name, const ConstantOfShapeParam& param)
    : OpBase(std::move(name), "ConstantOfShape"), param_(param) {
    SyncIO();
}

void ConstantOfShapeOp::SyncIO() {
    SetInputs({param_.shape});
    SetOutputs({param_.out});
}

int32_t ConstantOfShapeOp::CheckShape() const {
    if (param_.shape == nullptr || param_.out == nullptr || param_.output_type == DataType::UNKNOWN) {
        return -1;
    }
    std::vector<int64_t> dims;
    return ReadIntegerTensor(param_.shape, &dims) && !dims.empty() ? 0 : -1;
}

int32_t ConstantOfShapeOp::InferOutputShapes() {
    if (CheckShape() != 0) {
        return -1;
    }
    std::vector<int64_t> dims;
    if (!ReadIntegerTensor(param_.shape, &dims) || dims.empty()) {
        return -1;
    }
    for (const auto dim : dims) {
        if (dim <= 0) {
            return -1;
        }
    }
    param_.out = tensor_op_detail::AllocateOutput(param_.out, dims, param_.output_type);
    if (param_.out == nullptr) {
        return -1;
    }
    SyncIO();
    return 0;
}

void ConstantOfShapeOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) {
        kernel_->SetParamOwner(std::make_shared<std::decay_t<decltype(param_)>>(param_));
    }
}

int32_t ConstantOfShapeOp::Run() {
    if (InferOutputShapes() != 0 || kernel_ == nullptr) {
        return -1;
    }
    return kernel_->compute();
}

}  // namespace operators
}  // namespace feather
