#include "src/operator/reshape_op.h"

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

std::unique_ptr<KernelBase> CreateReshapeKernel() {
    kernel::EnsureReshapeKernelsRegistered();
    return CreateHostKernelForTensor("Reshape", {});
}

std::shared_ptr<OpBase> BuildReshapeOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors) {
    if (node.inputs.size() != 1 || node.outputs.size() != 1) {
        return nullptr;
    }

    ReshapeParam param;
    param.input = tensors[node.inputs[0]];
    param.out = tensors[node.outputs[0]];
    param.target_shape = GetShapeAttribute(node.attributes, "shape");
    if (param.input == nullptr || param.out == nullptr || param.target_shape.empty()) {
        return nullptr;
    }

    auto op = std::make_shared<ReshapeOp>(node.name.empty() ? "reshape" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    auto kernel = CreateHostKernelForTensor("Reshape", {param.input, param.out});
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_reshape_op_registered = []() {
    OperatorRegistry::instance().Register("Reshape", BuildReshapeOp);
    return true;
}();

}  // namespace

void EnsureReshapeOperatorRegistered() { (void)g_reshape_op_registered; }

ReshapeOp::ReshapeOp() : OpBase("reshape", "Reshape") {}

ReshapeOp::ReshapeOp(const ReshapeParam& param) : ReshapeOp("reshape", param) {}

ReshapeOp::ReshapeOp(std::string name, const ReshapeParam& param) : OpBase(std::move(name), "Reshape"), param_(param) {
    SyncIO();
}

void ReshapeOp::SyncIO() {
    SetInputs({param_.input});
    SetOutputs({param_.out});
}

int32_t ReshapeOp::CheckShape() const {
    if (param_.input == nullptr || param_.out == nullptr || param_.target_shape.empty()) {
        return -1;
    }
    int64_t target_numel = 1;
    for (const auto dim : param_.target_shape) {
        if (dim <= 0) {
            return -1;
        }
        target_numel *= dim;
    }
    return target_numel == param_.input->numel() ? 0 : -1;
}

int32_t ReshapeOp::InferOutputShapes() {
    if (CheckShape() != 0) {
        return -1;
    }
    const size_t required_bytes =
        static_cast<size_t>(param_.input->numel()) *
        DataTypeBytes(ResolveExecutionDataType({param_.input, param_.out}, DataType::FP32));
    if (param_.out == nullptr || !param_.out->IsInitialized() || param_.out->memory_size() < required_bytes) {
        param_.out = std::make_shared<Tensor>(param_.target_shape);
    } else {
        param_.out->Resize(param_.target_shape);
    }
    SyncIO();
    return 0;
}

int32_t ReshapeOp::Run() {
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
