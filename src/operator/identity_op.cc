#include "src/operator/identity_op.h"

#include <utility>

#include "core/operator_registry.h"
#include "src/operator/tensor_op_utils.h"
#include "util/types.h"

namespace feather {
namespace operators {

namespace {

std::shared_ptr<OpBase> BuildIdentityOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if (node.inputs.size() != 1 || node.outputs.size() != 1) {
        return nullptr;
    }

    UnaryParam param{};
    param.input = tensors[node.inputs[0]];
    param.out = tensors[node.outputs[0]];

    auto op = std::make_shared<IdentityOp>(node.name.empty() ? "identity" : node.name, param);
    op->SetExecutionDevice(context.device);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    auto kernel = CreateKernelForTensor(context.device, "Identity", {param.input, param.out});
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_identity_op_registered = []() {
    OperatorRegistry::instance().Register("Identity", BuildIdentityOp);
    return true;
}();

}  // namespace

void EnsureIdentityOperatorRegistered() { (void)g_identity_op_registered; }

IdentityOp::IdentityOp() : OpBase("identity", "Identity") {}

IdentityOp::IdentityOp(const UnaryParam& param) : IdentityOp("identity", param) {}

IdentityOp::IdentityOp(std::string name, const UnaryParam& param) : OpBase(std::move(name), "Identity"), param_(param) {
    SyncIO();
}

void IdentityOp::SyncIO() {
    SetInputs({param_.input});
    SetOutputs({param_.out});
}

int32_t IdentityOp::CheckShape() const {
    if (param_.input == nullptr || param_.out == nullptr) {
        return -1;
    }
    return 0;
}

int32_t IdentityOp::InferOutputShapes() {
    if (CheckShape() != 0) {
        return -1;
    }

    // Identity has view semantics on host backends. CUDA keeps separate host
    // Tensor storage but shares the device allocation in its view kernel.
    const auto device = execution_device_explicit_ ? execution_device_ : ActiveKernelDevice();
    if (device != DeviceType::CUDA && param_.input->data_type() != DataType::INT8) {
        param_.out->ShareDataWith(*param_.input);
        param_.out->Resize(param_.input->dims().data());
        param_.out->set_data_type(param_.input->data_type());
        param_.out->set_layout(param_.input->layout());
        SyncIO();
        return 0;
    }

    if (tensor_op_detail::InferSameTypeOutput(param_.input, &param_.out, param_.input->dims().data()) != 0) return -1;
    SyncIO();
    return 0;
}

void IdentityOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) {
        kernel_->SetParamOwner(std::make_shared<std::decay_t<decltype(param_)>>(param_));
    }
}

int32_t IdentityOp::Run() {
    if (InferOutputShapes() != 0 || kernel_ == nullptr) {
        return -1;
    }
    RefreshKernelParams();
    return kernel_->compute();
}

void IdentityOp::RefreshKernelParams() {
    if (kernel_ != nullptr) kernel_->SetParamOwner(std::make_shared<UnaryParam>(param_));
}

}  // namespace operators
}  // namespace feather
