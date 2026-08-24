#include "src/operator/cast_op.h"

#include "core/operator_registry.h"
#include "src/operator/tensor_op_utils.h"

namespace feather {
namespace operators {

namespace {

bool ResolveCastDataType(int64_t onnx_type, DataType* data_type) {
    if (data_type == nullptr) {
        return false;
    }
    switch (onnx_type) {
        case 1:
            *data_type = DataType::FP32;
            return true;
        case 2:
            *data_type = DataType::UINT8;
            return true;
        case 3:
            *data_type = DataType::INT8;
            return true;
        case 6:
            *data_type = DataType::INT32;
            return true;
        case 7:
            *data_type = DataType::INT64;
            return true;
        case 9:
            *data_type = DataType::BOOL;
            return true;
        case 10:
            *data_type = DataType::FP16;
            return true;
        case 16:
            *data_type = DataType::BF16;
            return true;
        default:
            return false;
    }
}

std::shared_ptr<OpBase> BuildCastOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if (node.inputs.size() != 1 || node.outputs.size() != 1) {
        return nullptr;
    }

    CastParam param{};
    param.input = tensors[node.inputs[0]];
    param.out = tensors[node.outputs[0]];
    const int64_t onnx_type = tensor_op_detail::GetIntAttribute(node.attributes, "to", 1);
    if (!ResolveCastDataType(onnx_type, &param.to)) {
        return nullptr;
    }

    auto op = std::make_shared<CastOp>(node.name.empty() ? "Cast" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    kernel::EnsureCastKernelsRegistered();
    std::unique_ptr<KernelBase> kernel;
    const bool floating_cast = param.input != nullptr && tensor_op_detail::IsFloatingPointDataType(param.input->data_type()) &&
                               tensor_op_detail::IsFloatingPointDataType(param.to);
    const bool cuda_integer_to_float =
        context.device == DeviceType::CUDA && param.input != nullptr &&
        (param.input->data_type() == DataType::INT32 || param.input->data_type() == DataType::INT64) &&
        tensor_op_detail::IsFloatingPointDataType(param.to);
    if (floating_cast || cuda_integer_to_float) {
        kernel = CreateKernelForTensor(context.device, "Cast", {param.input}, DataType::FP32);
    } else {
        kernel = CreateKernelForTensor(DeviceType::COMMON, "Cast", {}, DataType::FP32);
    }
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_cast_op_registered = []() {
    OperatorRegistry::instance().Register("Cast", BuildCastOp);
    return true;
}();

}  // namespace

void EnsureCastOperatorRegistered() { (void)g_cast_op_registered; }

CastOp::CastOp(std::string name, const CastParam& param) : OpBase(std::move(name), "Cast"), param_(param) {
    SyncIO();
}

void CastOp::SyncIO() {
    SetInputs({param_.input});
    SetOutputs({param_.out});
}

int32_t CastOp::CheckShape() const {
    return param_.input != nullptr && param_.out != nullptr && DataTypeBytes(param_.to) != 0 ? 0 : -1;
}

int32_t CastOp::InferOutputShapes() {
    if (CheckShape() != 0) {
        return -1;
    }
    const auto out_shape = param_.input->dims().data();
    const size_t required_bytes = static_cast<size_t>(param_.input->numel()) * DataTypeBytes(param_.to);
    if (!param_.out->IsInitialized() || param_.out->memory_size() < required_bytes) {
        param_.out = std::make_shared<Tensor>(required_bytes);
    }
    param_.out->Resize(out_shape);
    param_.out->set_data_type(param_.to);
    param_.out->set_layout(param_.input->layout());
    SyncIO();
    return 0;
}

void CastOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) {
        kernel_->SetParam(static_cast<void*>(&param_));
    }
}

int32_t CastOp::Run() {
    if (InferOutputShapes() != 0 || kernel_ == nullptr) {
        return -1;
    }
    return kernel_->compute();
}

}  // namespace operators
}  // namespace feather
