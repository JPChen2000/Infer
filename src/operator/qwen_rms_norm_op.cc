#include "src/operator/qwen_rms_norm_op.h"

#include <algorithm>
#include <numeric>
#include <utility>

#include "core/operator_registry.h"

namespace feather {
namespace operators {
namespace {

int64_t Product(const std::vector<int64_t>& dims) {
    return std::accumulate(dims.begin(), dims.end(), int64_t{1}, std::multiplies<int64_t>());
}

float ReadWeightOffset(const model::NodeDesc& node) {
    const auto it = node.attributes.find("weight_offset");
    if (it == node.attributes.end()) {
        return 0.0f;
    }
    if (const auto* value = std::get_if<float>(&it->second)) {
        return *value;
    }
    if (const auto* value = std::get_if<int64_t>(&it->second)) {
        return static_cast<float>(*value);
    }
    return 0.0f;
}

std::shared_ptr<OpBase> BuildQwenRmsNormOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if (node.inputs.size() != 3 || node.outputs.size() != 1) {
        return nullptr;
    }
    QwenRmsNormParam param{};
    param.input = tensors[node.inputs[0]];
    param.weight = tensors[node.inputs[1]];
    param.epsilon = tensors[node.inputs[2]];
    param.out = tensors[node.outputs[0]];
    param.weight_offset = ReadWeightOffset(node);
    auto op = std::make_shared<QwenRmsNormOp>(node.name.empty() ? "qwen_rms_norm" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    kernel::EnsureQwenRmsNormKernelsRegistered();
    auto kernel = CreateKernelForTensor(context.device, "QwenRmsNorm", {param.input, param.out}, DataType::FP32);
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_qwen_rms_norm_registered = []() {
    OperatorRegistry::instance().Register("QwenRmsNorm", BuildQwenRmsNormOp);
    return true;
}();

}  // namespace

void EnsureQwenRmsNormOperatorRegistered() { (void)g_qwen_rms_norm_registered; }

QwenRmsNormOp::QwenRmsNormOp(std::string name, const QwenRmsNormParam& param)
    : OpBase(std::move(name), "QwenRmsNorm"), param_(param) {
    SyncIO();
}

void QwenRmsNormOp::SyncIO() { SetInputs({param_.input, param_.weight, param_.epsilon}); SetOutputs({param_.out}); }

int32_t QwenRmsNormOp::CheckShape() const {
    if (param_.input == nullptr || param_.weight == nullptr || param_.epsilon == nullptr || param_.out == nullptr ||
        param_.input->dims().empty() || param_.input->dims() != param_.out->dims() ||
        param_.input->numel() <= 0 || param_.weight->numel() != param_.input->dims()[param_.input->dims().size() - 1] ||
        param_.epsilon->numel() != 1) {
        return -1;
    }
    return 0;
}

int32_t QwenRmsNormOp::InferOutputShapes() {
    if (CheckShape() != 0) {
        return -1;
    }
    const DataType output_type = param_.out->data_type() == DataType::UNKNOWN ? param_.input->data_type()
                                                                                : param_.out->data_type();
    const size_t element_bytes = DataTypeBytes(output_type);
    if (element_bytes == 0) {
        return -1;
    }
    const size_t bytes = static_cast<size_t>(Product(param_.input->dims().data())) * element_bytes;
    if (!param_.out->IsInitialized() || param_.out->memory_size() < bytes) {
        param_.out = std::make_shared<Tensor>(bytes);
    }
    param_.out->Resize(param_.input->dims());
    param_.out->set_data_type(output_type);
    param_.out->set_layout(param_.input->layout());
    SyncIO();
    return 0;
}

void QwenRmsNormOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) {
        kernel_->SetParamOwner(std::make_shared<std::decay_t<decltype(param_)>>(param_));
    }
}

int32_t QwenRmsNormOp::Run() { return InferOutputShapes() == 0 && kernel_ != nullptr ? kernel_->compute() : -1; }

}  // namespace operators
}  // namespace feather
