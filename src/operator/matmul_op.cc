#include "src/operator/matmul_op.h"

#include <utility>

#include "core/operator_registry.h"
#include "util/types.h"

namespace feather {
namespace operators {

namespace {

std::unique_ptr<KernelBase> CreateMatMulKernel() {
    kernel::EnsureMatMulKernelsRegistered();
    return CreateHostKernelForTensor("MatMul", {}, DataType::FP32);
}

std::shared_ptr<OpBase> BuildMatMulOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors) {
    if (node.inputs.size() != 2 || node.outputs.size() != 1) {
        return nullptr;
    }

    MatMulParam param{};
    param.a = tensors[node.inputs[0]];
    param.b = tensors[node.inputs[1]];
    param.out = tensors[node.outputs[0]];

    auto op = std::make_shared<MatMulOp>(node.name.empty() ? "matmul" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    auto kernel = CreateHostKernelForTensor("MatMul", {param.a, param.b, param.out}, DataType::FP32);
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_matmul_op_registered = []() {
    OperatorRegistry::instance().Register("MatMul", BuildMatMulOp);
    return true;
}();

}  // namespace

void EnsureMatMulOperatorRegistered() { (void)g_matmul_op_registered; }

MatMulOp::MatMulOp() : OpBase("matmul", "MatMul") {}

MatMulOp::MatMulOp(const MatMulParam& param) : MatMulOp("matmul", param) {}

MatMulOp::MatMulOp(std::string name, const MatMulParam& param) : OpBase(std::move(name), "MatMul"), param_(param) {
    SyncIO();
}

void MatMulOp::SyncIO() {
    SetInputs({param_.a, param_.b});
    SetOutputs({param_.out});
}

int32_t MatMulOp::CheckShape() const {
    if (param_.a == nullptr || param_.b == nullptr || param_.out == nullptr) {
        return -1;
    }
    if (param_.a->dims().size() != 2 || param_.b->dims().size() != 2) {
        return -1;
    }
    return param_.a->dims()[1] == param_.b->dims()[0] ? 0 : -1;
}

int32_t MatMulOp::InferOutputShapes() {
    if (CheckShape() != 0) {
        return -1;
    }

    const std::vector<int64_t> out_shape = {param_.a->dims()[0], param_.b->dims()[1]};
    const size_t required_bytes =
        static_cast<size_t>(out_shape[0] * out_shape[1]) *
        DataTypeBytes(ResolveExecutionDataType({param_.a, param_.b, param_.out}, DataType::FP32));
    if (param_.out == nullptr || !param_.out->IsInitialized() || param_.out->memory_size() < required_bytes) {
        param_.out = std::make_shared<Tensor>(out_shape);
    } else {
        param_.out->Resize(out_shape);
    }
    SyncIO();
    return 0;
}

void MatMulOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) {
        kernel_->SetParam((void*)&param_);
    }
}

int32_t MatMulOp::Run() {
    if (InferOutputShapes() != 0 || kernel_ == nullptr) {
        return -1;
    }
    return kernel_->compute();
}

}  // namespace operators
}  // namespace feather
