#include "src/operator/identity_op.h"

#include <utility>

#include "core/operator_registry.h"
#include "util/types.h"

namespace feather {
namespace operators {

namespace {

std::unique_ptr<KernelBase> CreateIdentityKernel() {
    kernel::EnsureIdentityKernelsRegistered();
    return CreateHostKernelForTensor("Identity", {});
}

std::shared_ptr<OpBase> BuildIdentityOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors) {
    if (node.inputs.size() != 1 || node.outputs.size() != 1) {
        return nullptr;
    }

    UnaryParam param{};
    param.input = tensors[node.inputs[0]];
    param.out = tensors[node.outputs[0]];

    auto op = std::make_shared<IdentityOp>(node.name.empty() ? "identity" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    auto kernel = CreateHostKernelForTensor("Identity", {param.input, param.out});
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
    if (ActiveKernelDevice() != DeviceType::CUDA) {
        param_.out->ShareDataWith(*param_.input);
        param_.out->Resize(param_.input->dims().data());
        param_.out->set_data_type(param_.input->data_type());
        param_.out->set_layout(param_.input->layout());
        SyncIO();
        return 0;
    }

    const auto& out_shape = param_.input->dims().data();
    const size_t required_bytes =
        static_cast<size_t>(param_.input->numel()) *
        DataTypeBytes(ResolveExecutionDataType({param_.input, param_.out}, DataType::FP32));
    if (param_.out == nullptr || !param_.out->IsInitialized() || param_.out->memory_size() < required_bytes) {
        param_.out = std::make_shared<Tensor>(out_shape);
    } else {
        param_.out->Resize(out_shape);
    }
    param_.out->set_layout(param_.input->layout());
    SyncIO();
    return 0;
}

void IdentityOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) {
        kernel_->SetParam((void*)&param_);
    }
}

int32_t IdentityOp::Run() {
    if (InferOutputShapes() != 0 || kernel_ == nullptr) {
        return -1;
    }
    return kernel_->compute();
}

}  // namespace operators
}  // namespace feather
