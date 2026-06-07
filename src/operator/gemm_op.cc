#include "src/operator/gemm_op.h"

#include <utility>

#include "core/operator_registry.h"
#include "util/types.h"

namespace feather {
namespace operators {

namespace {

std::unique_ptr<KernelBase> CreateGemmKernel() {
    kernel::EnsureGemmKernelsRegistered();
    return CreateHostKernelForTensor("Gemm", {}, DataType::FP32);
}

std::shared_ptr<OpBase> BuildGemmOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors) {
    if (node.inputs.size() < 2 || node.outputs.size() != 1) {
        return nullptr;
    }

    GemmParam param;
    param.a = tensors[node.inputs[0]];
    param.b = tensors[node.inputs[1]];
    param.bias = node.inputs.size() > 2 ? tensors[node.inputs[2]] : nullptr;
    param.out = tensors[node.outputs[0]];
    if (param.a == nullptr || param.b == nullptr || param.out == nullptr) {
        return nullptr;
    }

    auto op = std::make_shared<GemmOp>(node.name.empty() ? "gemm" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    auto kernel = CreateHostKernelForTensor("Gemm", {param.a, param.b, param.bias, param.out}, DataType::FP32);
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_gemm_op_registered = []() {
    OperatorRegistry::instance().Register("Gemm", BuildGemmOp);
    return true;
}();

}  // namespace

void EnsureGemmOperatorRegistered() { (void)g_gemm_op_registered; }

GemmOp::GemmOp() : OpBase("gemm", "Gemm") {}

GemmOp::GemmOp(const GemmParam& param) : GemmOp("gemm", param) {}

GemmOp::GemmOp(std::string name, const GemmParam& param) : OpBase(std::move(name), "Gemm"), param_(param) {
    SyncIO();
}

void GemmOp::SyncIO() {
    SetInputs({param_.a, param_.b, param_.bias});
    SetOutputs({param_.out});
}

int32_t GemmOp::CheckShape() const {
    if (param_.a == nullptr || param_.b == nullptr || param_.out == nullptr) {
        return -1;
    }
    if (param_.a->dims().size() != 2 || param_.b->dims().size() != 2) {
        return -1;
    }
    if (param_.a->dims()[1] != param_.b->dims()[0]) {
        return -1;
    }
    if (param_.bias != nullptr && param_.bias->IsInitialized()) {
        if (param_.bias->dims().size() == 1) {
            if (param_.bias->dims()[0] != param_.b->dims()[1]) {
                return -1;
            }
        } else if (param_.bias->dims().size() == 2) {
            if (param_.bias->dims()[0] != param_.a->dims()[0] || param_.bias->dims()[1] != param_.b->dims()[1]) {
                return -1;
            }
        } else {
            return -1;
        }
    }
    return 0;
}

int32_t GemmOp::InferOutputShapes() {
    if (CheckShape() != 0) {
        return -1;
    }
    const std::vector<int64_t> out_shape = {param_.a->dims()[0], param_.b->dims()[1]};
    const size_t required_bytes =
        static_cast<size_t>(out_shape[0] * out_shape[1]) *
        DataTypeBytes(ResolveExecutionDataType({param_.a, param_.b, param_.bias, param_.out}, DataType::FP32));
    if (param_.out == nullptr || !param_.out->IsInitialized() || param_.out->memory_size() < required_bytes) {
        param_.out = std::make_shared<Tensor>(out_shape);
    } else {
        param_.out->Resize(out_shape);
    }
    SyncIO();
    return 0;
}

int32_t GemmOp::Run() {
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
