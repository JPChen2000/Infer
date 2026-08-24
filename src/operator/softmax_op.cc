#include "src/operator/softmax_op.h"

#include <cstdint>
#include <utility>

#include "core/operator_registry.h"
#include "util/types.h"

namespace feather {
namespace operators {

namespace {

int32_t GetIntAttribute(const std::unordered_map<std::string, model::AttributeValue>& attributes, const std::string& key,
                        int32_t default_value) {
    auto it = attributes.find(key);
    if (it == attributes.end()) {
        return default_value;
    }
    if (auto value = std::get_if<int64_t>(&it->second); value != nullptr) {
        return static_cast<int32_t>(*value);
    }
    return default_value;
}

std::unique_ptr<KernelBase> CreateSoftmaxKernel(const OperatorRegistry::BuildContext& context) {
    kernel::EnsureSoftmaxKernelsRegistered();
    return CreateKernelForTensor(context.device, "Softmax", {}, DataType::FP32);
}

std::shared_ptr<OpBase> BuildSoftmaxOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if (node.inputs.size() != 1 || node.outputs.size() != 1) {
        return nullptr;
    }

    SoftmaxParam param{};
    param.input = tensors[node.inputs[0]];
    param.out = tensors[node.outputs[0]];
    param.axis = GetIntAttribute(node.attributes, "axis", -1);

    auto op = std::make_shared<SoftmaxOp>(node.name.empty() ? "softmax" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    auto kernel = CreateKernelForTensor(context.device, "Softmax", {param.input, param.out}, DataType::FP32);
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_softmax_op_registered = []() {
    OperatorRegistry::instance().Register("Softmax", BuildSoftmaxOp);
    return true;
}();

}  // namespace

void EnsureSoftmaxOperatorRegistered() { (void)g_softmax_op_registered; }

SoftmaxOp::SoftmaxOp() : OpBase("softmax", "Softmax") {}

SoftmaxOp::SoftmaxOp(const SoftmaxParam& param) : SoftmaxOp("softmax", param) {}

SoftmaxOp::SoftmaxOp(std::string name, const SoftmaxParam& param) : OpBase(std::move(name), "Softmax"), param_(param) {
    SyncIO();
}

void SoftmaxOp::SyncIO() {
    SetInputs({param_.input});
    SetOutputs({param_.out});
}

int32_t SoftmaxOp::CheckShape() const {
    if (param_.input == nullptr || param_.out == nullptr) {
        return -1;
    }
    const auto rank = static_cast<int32_t>(param_.input->dims().size());
    if (rank <= 0) {
        return -1;
    }
    const int32_t axis = param_.axis < 0 ? param_.axis + rank : param_.axis;
    return axis >= 0 && axis < rank ? 0 : -1;
}

int32_t SoftmaxOp::InferOutputShapes() {
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
    SyncIO();
    return 0;
}

void SoftmaxOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) {
        kernel_->SetParam((void*)&param_);
    }
}

int32_t SoftmaxOp::Run() {
    if (InferOutputShapes() != 0 || kernel_ == nullptr) {
        return -1;
    }
    return kernel_->compute();
}

}  // namespace operators
}  // namespace feather
