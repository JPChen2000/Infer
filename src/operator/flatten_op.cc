#include "src/operator/flatten_op.h"

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

std::unique_ptr<KernelBase> CreateFlattenKernel() {
    kernel::EnsureFlattenKernelsRegistered();
    return CreateHostKernelForTensor("Flatten", {});
}

std::shared_ptr<OpBase> BuildFlattenOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors) {
    if (node.inputs.size() != 1 || node.outputs.size() != 1) {
        return nullptr;
    }

    FlattenParam param{};
    param.input = tensors[node.inputs[0]];
    param.out = tensors[node.outputs[0]];
    param.axis = GetIntAttribute(node.attributes, "axis", 1);

    auto op = std::make_shared<FlattenOp>(node.name.empty() ? "flatten" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    auto kernel = CreateHostKernelForTensor("Flatten", {param.input, param.out});
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_flatten_op_registered = []() {
    OperatorRegistry::instance().Register("Flatten", BuildFlattenOp);
    return true;
}();

}  // namespace

void EnsureFlattenOperatorRegistered() { (void)g_flatten_op_registered; }

FlattenOp::FlattenOp() : OpBase("flatten", "Flatten") {}

FlattenOp::FlattenOp(const FlattenParam& param) : FlattenOp("flatten", param) {}

FlattenOp::FlattenOp(std::string name, const FlattenParam& param) : OpBase(std::move(name), "Flatten"), param_(param) {
    SyncIO();
}

void FlattenOp::SyncIO() {
    SetInputs({param_.input});
    SetOutputs({param_.out});
}

int32_t FlattenOp::CheckShape() const {
    if (param_.input == nullptr || param_.out == nullptr) {
        return -1;
    }
    const int32_t rank = static_cast<int32_t>(param_.input->dims().size());
    return param_.axis >= 0 && param_.axis <= rank ? 0 : -1;
}

int32_t FlattenOp::InferOutputShapes() {
    if (CheckShape() != 0) {
        return -1;
    }

    const std::vector<int64_t> out_shape = {
        param_.input->dims().count(0, param_.axis),
        param_.input->dims().count(param_.axis, static_cast<int>(param_.input->dims().size())),
    };
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

void FlattenOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) {
        kernel_->SetParam((void*)&param_);
    }
}

int32_t FlattenOp::Run() {
    if (InferOutputShapes() != 0 || kernel_ == nullptr) {
        return -1;
    }
    return kernel_->compute();
}

}  // namespace operators
}  // namespace feather
