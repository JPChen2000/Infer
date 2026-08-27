#include "src/operator/qwen_depthwise_conv_op.h"

#include <cstddef>
#include <cmath>
#include <vector>

#include "core/operator_registry.h"
#include "util/types.h"

namespace feather {
namespace operators {
namespace {

bool HasShape(const std::shared_ptr<Tensor>& tensor, const std::vector<int64_t>& shape) {
    return tensor != nullptr && tensor->dims().data() == shape;
}

bool IsFp8DataType(DataType data_type) {
    return data_type == DataType::FP8E4M3 || data_type == DataType::FP8E5M2;
}

bool IsValidScale(float scale) { return std::isfinite(scale) && scale > 0.0f; }

bool IsQwenDepthwiseConvStateShape(const QwenDepthwiseConvStateParam& param) {
    if (param.state == nullptr || param.mixed == nullptr || param.weight == nullptr || param.conv_out == nullptr ||
        param.discarded_prefix == nullptr || param.next_state == nullptr ||
        param.state->data_type() != DataType::BF16 || param.mixed->data_type() != DataType::BF16 ||
        param.conv_out->data_type() != DataType::BF16 ||
        param.discarded_prefix->data_type() != DataType::BF16 || param.next_state->data_type() != DataType::BF16 ||
        param.state->dims().size() != 3 || param.mixed->dims().size() != 3 || param.weight->dims().size() != 4) {
        return false;
    }

    const DataType weight_dtype = param.weight->data_type();
    const DataType fp8_dtype = param.fp8_dtype == DataType::UNKNOWN ? weight_dtype : param.fp8_dtype;
    if (weight_dtype == DataType::BF16) {
        if (param.fp8_dtype != DataType::UNKNOWN) {
            return false;
        }
    } else if (IsFp8DataType(weight_dtype)) {
        if (fp8_dtype != weight_dtype || !HasCompatiblePerTensorQuantization(param.weight->quantization()) ||
            !IsValidScale(param.weight->quantization_scale()) || !IsValidScale(param.fp8_input_scale) ||
            !IsValidScale(param.fp8_output_scale)) {
            return false;
        }
    } else {
        return false;
    }

    const int64_t channels = param.state->dims()[1];
    return channels > 0 && HasShape(param.state, {1, channels, 3}) && HasShape(param.mixed, {1, channels, 1}) &&
           HasShape(param.weight, {channels, 1, 1, 4}) && HasShape(param.conv_out, {1, channels, 1, 1}) &&
           HasShape(param.discarded_prefix, {1, channels, 1}) && HasShape(param.next_state, {1, channels, 3});
}

bool ResizeOutput(std::shared_ptr<Tensor>* output, const std::vector<int64_t>& shape) {
    if (output == nullptr) {
        return false;
    }
    size_t count = 1;
    for (const int64_t dim : shape) {
        if (dim <= 0 || count > static_cast<size_t>(-1) / static_cast<size_t>(dim)) {
            return false;
        }
        count *= static_cast<size_t>(dim);
    }
    const size_t bytes = count * DataTypeBytes(DataType::BF16);
    if (*output == nullptr || !(*output)->IsInitialized() || (*output)->memory_size() < bytes) {
        *output = std::make_shared<Tensor>(bytes);
    }
    (*output)->Resize(shape);
    (*output)->set_data_type(DataType::BF16);
    (*output)->set_layout(DataLayout::ND);
    return true;
}

std::shared_ptr<OpBase> BuildQwenDepthwiseConvStateOp(const model::NodeDesc& node,
                                                       OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if (node.inputs.size() != 3 || node.outputs.size() != 3) {
        return nullptr;
    }

    QwenDepthwiseConvStateParam param{};
    param.state = tensors[node.inputs[0]];
    param.mixed = tensors[node.inputs[1]];
    param.weight = tensors[node.inputs[2]];
    param.conv_out = tensors[node.outputs[0]];
    param.discarded_prefix = tensors[node.outputs[1]];
    param.next_state = tensors[node.outputs[2]];
    if (param.weight != nullptr && IsFp8DataType(param.weight->data_type())) {
        param.fp8_dtype = param.weight->data_type();
        const auto input_scale_it = node.attributes.find("fp8_input_scale");
        if (input_scale_it != node.attributes.end()) {
            const auto* scale = std::get_if<float>(&input_scale_it->second);
            if (scale == nullptr) {
                return nullptr;
            }
            param.fp8_input_scale = *scale;
        }
        const auto output_scale_it = node.attributes.find("fp8_output_scale");
        if (output_scale_it != node.attributes.end()) {
            const auto* scale = std::get_if<float>(&output_scale_it->second);
            if (scale == nullptr) {
                return nullptr;
            }
            param.fp8_output_scale = *scale;
        }
    }
    auto op = std::make_shared<QwenDepthwiseConvStateOp>(
        node.name.empty() ? "qwen_depthwise_conv_state" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    kernel::EnsureQwenDepthwiseConvStateKernelsRegistered();
    const DataType execution_dtype = IsFp8DataType(param.fp8_dtype) ? param.fp8_dtype : DataType::BF16;
    auto kernel = CreateKernelForTensor(context.device, "QwenDepthwiseConvState", {param.weight}, execution_dtype);
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_qwen_depthwise_conv_state_registered = []() {
    OperatorRegistry::instance().Register("QwenDepthwiseConvState", BuildQwenDepthwiseConvStateOp);
    return true;
}();

}  // namespace

void EnsureQwenDepthwiseConvStateOperatorRegistered() { (void)g_qwen_depthwise_conv_state_registered; }

QwenDepthwiseConvStateOp::QwenDepthwiseConvStateOp(std::string name, const QwenDepthwiseConvStateParam& param)
    : OpBase(std::move(name), "QwenDepthwiseConvState"), param_(param) {
    SyncIO();
}

void QwenDepthwiseConvStateOp::SyncIO() {
    SetInputs({param_.state, param_.mixed, param_.weight});
    SetOutputs({param_.conv_out, param_.discarded_prefix, param_.next_state});
}

int32_t QwenDepthwiseConvStateOp::CheckShape() const {
    return IsQwenDepthwiseConvStateShape(param_) ? 0 : -1;
}

int32_t QwenDepthwiseConvStateOp::InferOutputShapes() {
    if (param_.state == nullptr || param_.state->data_type() != DataType::BF16 || param_.state->dims().size() != 3 ||
        param_.state->dims()[0] != 1 || param_.state->dims()[1] <= 0 || param_.state->dims()[2] != 3 ||
        param_.mixed == nullptr || param_.weight == nullptr) {
        return -1;
    }
    const int64_t channels = param_.state->dims()[1];
    if (!ResizeOutput(&param_.conv_out, {1, channels, 1, 1}) ||
        !ResizeOutput(&param_.discarded_prefix, {1, channels, 1}) ||
        !ResizeOutput(&param_.next_state, {1, channels, 3})) {
        return -1;
    }
    return CheckShape() == 0 ? (SyncIO(), 0) : -1;
}

void QwenDepthwiseConvStateOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) {
        kernel_->SetParamOwner(std::make_shared<std::decay_t<decltype(param_)>>(param_));
    }
}

int32_t QwenDepthwiseConvStateOp::Run() {
    return InferOutputShapes() == 0 && kernel_ != nullptr ? kernel_->compute() : -1;
}

}  // namespace operators
}  // namespace feather
