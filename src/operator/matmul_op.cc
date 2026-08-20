#include "src/operator/matmul_op.h"

#include <limits>
#include <utility>

#include "core/operator_registry.h"
#include "util/types.h"

namespace feather {
namespace operators {

namespace {

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
    kernel::EnsureMatMulKernelsRegistered();
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
    if (param_.a->dims().size() < 2 || param_.b->dims().size() < 2) {
        return -1;
    }
    return param_.a->dims()[param_.a->dims().size() - 1] == param_.b->dims()[param_.b->dims().size() - 2] ? 0 : -1;
}

int32_t MatMulOp::InferOutputShapes() {
    if (CheckShape() != 0) {
        return -1;
    }

    const auto& a_dims = param_.a->dims().data();
    const auto& b_dims = param_.b->dims().data();
    const size_t a_rank = a_dims.size();
    const size_t b_rank = b_dims.size();
    const size_t batch_rank = std::max(a_rank, b_rank) - 2;
    std::vector<int64_t> out_shape(batch_rank, 1);
    for (size_t i = 0; i < batch_rank; ++i) {
        const int64_t a_dim = i < batch_rank - (a_rank - 2) ? 1 : a_dims[i - (batch_rank - (a_rank - 2))];
        const int64_t b_dim = i < batch_rank - (b_rank - 2) ? 1 : b_dims[i - (batch_rank - (b_rank - 2))];
        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            return -1;
        }
        out_shape[i] = std::max(a_dim, b_dim);
    }
    out_shape.push_back(a_dims[a_rank - 2]);
    out_shape.push_back(b_dims[b_rank - 1]);
    int64_t output_numel = 1;
    for (const auto dim : out_shape) {
        if (dim <= 0 || output_numel > std::numeric_limits<int64_t>::max() / dim) {
            return -1;
        }
        output_numel *= dim;
    }
    const size_t required_bytes =
        static_cast<size_t>(output_numel) *
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
