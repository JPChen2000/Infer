#include "src/operator/pow_op.h"

#include <utility>

#include "core/operator_registry.h"
#include "src/operator/control_tensor.h"
#include "util/types.h"

namespace feather {
namespace operators {

namespace {

float GetFloatAttribute(const std::unordered_map<std::string, model::AttributeValue>& attributes, const std::string& key,
                        float default_value) {
    auto it = attributes.find(key);
    if (it == attributes.end()) {
        return default_value;
    }
    if (auto value = std::get_if<float>(&it->second); value != nullptr) {
        return *value;
    }
    if (auto value = std::get_if<int64_t>(&it->second); value != nullptr) {
        return static_cast<float>(*value);
    }
    return default_value;
}

std::shared_ptr<OpBase> BuildPowOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if ((node.inputs.size() != 1 && node.inputs.size() != 2) || node.outputs.size() != 1) {
        return nullptr;
    }

    PowParam param{};
    param.input = tensors[node.inputs[0]];
    if (node.inputs.size() == 2) {
        param.exponent_tensor = tensors[node.inputs[1]];
    }
    param.out = tensors[node.outputs[0]];
    param.exponent = GetFloatAttribute(node.attributes, "exponent", 1.0f);
    if (param.input == nullptr || param.out == nullptr ||
        (param.exponent_tensor == nullptr && node.inputs.size() == 2)) {
        return nullptr;
    }

    auto op = std::make_shared<PowOp>(node.name.empty() ? "pow" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    auto kernel = CreateKernelForTensor(context.device, "Pow", {param.input, param.out});
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_pow_op_registered = []() {
    OperatorRegistry::instance().Register("Pow", BuildPowOp);
    return true;
}();

}  // namespace

void EnsurePowOperatorRegistered() { (void)g_pow_op_registered; }

PowOp::PowOp() : OpBase("pow", "Pow") {}

PowOp::PowOp(const PowParam& param) : PowOp("pow", param) {}

PowOp::PowOp(std::string name, const PowParam& param) : OpBase(std::move(name), "Pow"), param_(param) { SyncIO(); }

void PowOp::SyncIO() {
    std::vector<std::shared_ptr<Tensor>> inputs = {param_.input};
    if (param_.exponent_tensor != nullptr) {
        inputs.push_back(param_.exponent_tensor);
    }
    SetInputs(std::move(inputs));
    SetOutputs({param_.out});
}

int32_t PowOp::CheckShape() const {
    if (param_.input == nullptr || param_.out == nullptr) {
        return -1;
    }
    if (param_.exponent_tensor != nullptr) {
        float exponent = 0.0f;
        if (!ReadScalarFloatTensor(param_.exponent_tensor, &exponent)) {
            return -1;
        }
    }
    return 0;
}

int32_t PowOp::InferOutputShapes() {
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

void PowOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) {
        kernel_->SetParamOwner(std::make_shared<std::decay_t<decltype(param_)>>(param_));
    }
}

int32_t PowOp::Run() {
    if (InferOutputShapes() != 0 || kernel_ == nullptr) {
        return -1;
    }
    if (param_.exponent_tensor != nullptr && !ReadScalarFloatTensor(param_.exponent_tensor, &param_.exponent)) {
        return -1;
    }
    return kernel_->compute();
}

}  // namespace operators
}  // namespace feather
