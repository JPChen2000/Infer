#include "src/operator/relu_op.h"

#include <utility>

#include "core/operator_registry.h"
#include "util/types.h"

namespace feather {
namespace operators {

namespace {

std::unique_ptr<KernelBase> CreateReluKernel() {
    kernel::EnsureReluKernelsRegistered();
    return CreateHostKernelForTensor("ReLU", {});
}

std::shared_ptr<OpBase> BuildReluOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors) {
    if (node.inputs.size() != 1 || node.outputs.size() != 1) {
        return nullptr;
    }

    UnaryParam param;
    param.input = tensors[node.inputs[0]];
    param.out = tensors[node.outputs[0]];
    if (param.input == nullptr || param.out == nullptr) {
        return nullptr;
    }

    auto op = std::make_shared<ReluOp>(node.name.empty() ? "relu" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    auto kernel = CreateHostKernelForTensor("ReLU", {param.input, param.out});
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_relu_op_registered = []() {
    OperatorRegistry::instance().Register("ReLU", BuildReluOp);
    return true;
}();

}  // namespace

void EnsureReluOperatorRegistered() { (void)g_relu_op_registered; }

ReluOp::ReluOp() : OpBase("relu", "ReLU") {}

ReluOp::ReluOp(const UnaryParam& param) : ReluOp("relu", param) {}

ReluOp::ReluOp(std::string name, const UnaryParam& param) : OpBase(std::move(name), "ReLU"), param_(param) {
    SyncIO();
}

void ReluOp::SyncIO() {
    SetInputs({param_.input});
    SetOutputs({param_.out});
}

int32_t ReluOp::CheckShape() const {
    if (param_.input == nullptr || param_.out == nullptr) {
        return -1;
    }
    return 0;
}

int32_t ReluOp::InferOutputShapes() {
    if (CheckShape() != 0) {
        return -1;
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

int32_t ReluOp::Run() {
    if (InferOutputShapes() != 0) {
        return -1;
    }
    if (kernel_ == nullptr) {
        return -1;
    }
    return kernel_->compute();
}

}  // namespace operators
}  // namespace feather
