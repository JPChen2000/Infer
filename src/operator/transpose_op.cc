#include "src/operator/transpose_op.h"

#include <utility>

#include "core/operator_registry.h"
#include "util/types.h"

namespace feather {
namespace operators {

namespace {

std::vector<int64_t> GetShapeAttribute(const std::unordered_map<std::string, model::AttributeValue>& attributes,
                                       const std::string& key) {
    auto it = attributes.find(key);
    if (it == attributes.end()) {
        return {};
    }
    if (auto value = std::get_if<std::vector<int64_t>>(&it->second); value != nullptr) {
        return *value;
    }
    return {};
}

bool IsValidPermutation(const std::vector<int64_t>& perm, size_t rank) {
    if (perm.size() != rank) {
        return false;
    }
    std::vector<bool> seen(rank, false);
    for (const auto axis : perm) {
        if (axis < 0 || axis >= static_cast<int64_t>(rank) || seen[axis]) {
            return false;
        }
        seen[axis] = true;
    }
    return true;
}

std::unique_ptr<KernelBase> CreateTransposeKernel() {
    kernel::EnsureTransposeKernelsRegistered();
    return CreateHostKernelForTensor("Transpose", {});
}

std::shared_ptr<OpBase> BuildTransposeOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors) {
    if (node.inputs.size() != 1 || node.outputs.size() != 1) {
        return nullptr;
    }

    TransposeParam param{};
    param.input = tensors[node.inputs[0]];
    param.out = tensors[node.outputs[0]];
    param.perm = GetShapeAttribute(node.attributes, "perm");

    auto op = std::make_shared<TransposeOp>(node.name.empty() ? "transpose" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    auto kernel = CreateHostKernelForTensor("Transpose", {param.input, param.out});
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_transpose_op_registered = []() {
    OperatorRegistry::instance().Register("Transpose", BuildTransposeOp);
    return true;
}();

}  // namespace

void EnsureTransposeOperatorRegistered() { (void)g_transpose_op_registered; }

TransposeOp::TransposeOp() : OpBase("transpose", "Transpose") {}

TransposeOp::TransposeOp(const TransposeParam& param) : TransposeOp("transpose", param) {}

TransposeOp::TransposeOp(std::string name, const TransposeParam& param)
    : OpBase(std::move(name), "Transpose"), param_(param) {
    SyncIO();
}

void TransposeOp::SyncIO() {
    SetInputs({param_.input});
    SetOutputs({param_.out});
}

int32_t TransposeOp::CheckShape() const {
    if (param_.input == nullptr || param_.out == nullptr) {
        return -1;
    }
    if (param_.input->dims().empty()) {
        return -1;
    }
    return IsValidPermutation(param_.perm, param_.input->dims().size()) ? 0 : -1;
}

int32_t TransposeOp::InferOutputShapes() {
    if (CheckShape() != 0) {
        return -1;
    }

    std::vector<int64_t> out_shape(param_.perm.size(), 1);
    for (size_t i = 0; i < param_.perm.size(); ++i) {
        out_shape[i] = param_.input->dims()[param_.perm[i]];
    }
    const size_t required_bytes =
        static_cast<size_t>(param_.input->numel()) *
        DataTypeBytes(ResolveExecutionDataType({param_.input, param_.out}, DataType::FP32));
    if (param_.out == nullptr || !param_.out->IsInitialized() || param_.out->memory_size() < required_bytes) {
        param_.out = std::make_shared<Tensor>(out_shape);
    } else {
        param_.out->Resize(out_shape);
    }
    SyncIO();
    return 0;
}

void TransposeOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) {
        kernel_->SetParam((void*)&param_);
    }
}

int32_t TransposeOp::Run() {
    if (InferOutputShapes() != 0 || kernel_ == nullptr) {
        return -1;
    }
    return kernel_->compute();
}

}  // namespace operators
}  // namespace feather
