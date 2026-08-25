#include "src/operator/resize_concat_op.h"

#include <algorithm>
#include <cmath>
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

std::vector<float> GetFloatVectorAttribute(const std::unordered_map<std::string, model::AttributeValue>& attributes,
                                           const std::string& key) {
    auto it = attributes.find(key);
    if (it == attributes.end()) {
        return {};
    }
    if (auto value = std::get_if<std::vector<float>>(&it->second); value != nullptr) {
        return *value;
    }
    return {};
}

std::shared_ptr<OpBase> BuildResizeConcatOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if (node.inputs.size() != 2 || node.outputs.size() != 1) {
        return nullptr;
    }

    ResizeConcatParam param{};
    param.resize_input = tensors[node.inputs[0]];
    param.concat_input = tensors[node.inputs[1]];
    param.out = tensors[node.outputs[0]];
    param.scales = GetFloatVectorAttribute(node.attributes, "scales");
    param.axis = GetIntAttribute(node.attributes, "axis", 1);
    param.resize_input_index = GetIntAttribute(node.attributes, "resize_input_index", 0);
    if (param.resize_input == nullptr || param.concat_input == nullptr || param.out == nullptr || param.scales.empty()) {
        return nullptr;
    }

    auto op = std::make_shared<ResizeConcatOp>(node.name.empty() ? "resize_concat" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    auto kernel =
        CreateKernelForTensor(context.device, "ResizeConcat", {param.resize_input, param.concat_input, param.out});
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_resize_concat_op_registered = []() {
    OperatorRegistry::instance().Register("ResizeConcat", BuildResizeConcatOp);
    return true;
}();

}  // namespace

void EnsureResizeConcatOperatorRegistered() { (void)g_resize_concat_op_registered; }

ResizeConcatOp::ResizeConcatOp() : OpBase("resize_concat", "ResizeConcat") {}

ResizeConcatOp::ResizeConcatOp(const ResizeConcatParam& param) : ResizeConcatOp("resize_concat", param) {}

ResizeConcatOp::ResizeConcatOp(std::string name, const ResizeConcatParam& param)
    : OpBase(std::move(name), "ResizeConcat"), param_(param) {
    SyncIO();
}

void ResizeConcatOp::SyncIO() {
    SetInputs({param_.resize_input, param_.concat_input});
    SetOutputs({param_.out});
}

int32_t ResizeConcatOp::CheckShape() const {
    if (param_.resize_input == nullptr || param_.concat_input == nullptr || param_.out == nullptr ||
        param_.scales.size() != 4 || param_.resize_input_index < 0 || param_.resize_input_index > 1) {
        return -1;
    }
    if (param_.resize_input->dims().size() != 4 || param_.concat_input->dims().size() != 4) {
        return -1;
    }
    const auto layout = NormalizeDataLayout(param_.resize_input->layout());
    if (layout != NormalizeDataLayout(param_.concat_input->layout())) {
        return -1;
    }
    const int32_t axis = param_.axis < 0 ? param_.axis + 4 : param_.axis;
    const int32_t channel_axis = layout == DataLayout::NHWC ? 3 : 1;
    if (axis != channel_axis) {
        return -1;
    }
    for (float scale : param_.scales) {
        if (scale <= 0.0f) {
            return -1;
        }
    }

    std::vector<int64_t> resized_shape;
    resized_shape.reserve(4);
    for (size_t i = 0; i < 4; ++i) {
        const auto dim = static_cast<int64_t>(
            std::llround(static_cast<double>(param_.resize_input->dims()[i]) * param_.scales[i]));
        if (dim <= 0) {
            return -1;
        }
        resized_shape.push_back(dim);
    }
    for (size_t i = 0; i < 4; ++i) {
        if (static_cast<int32_t>(i) == axis) {
            continue;
        }
        if (resized_shape[i] != param_.concat_input->dims()[i]) {
            return -1;
        }
    }
    return 0;
}

int32_t ResizeConcatOp::InferOutputShapes() {
    if (CheckShape() != 0) {
        return -1;
    }
    std::vector<int64_t> out_shape(4, 0);
    for (size_t i = 0; i < 4; ++i) {
        out_shape[i] = static_cast<int64_t>(
            std::llround(static_cast<double>(param_.resize_input->dims()[i]) * param_.scales[i]));
    }
    const int32_t axis = param_.axis < 0 ? param_.axis + 4 : param_.axis;
    out_shape[axis] += param_.concat_input->dims()[axis];

    int64_t out_numel = 1;
    for (const auto dim : out_shape) {
        out_numel *= dim;
    }
    const size_t required_bytes =
        static_cast<size_t>(std::max<int64_t>(1, out_numel)) *
        DataTypeBytes(ResolveExecutionDataType({param_.resize_input, param_.concat_input, param_.out}, DataType::FP32));
    if (param_.out == nullptr || !param_.out->IsInitialized() || param_.out->memory_size() < required_bytes) {
        param_.out = std::make_shared<Tensor>(out_shape);
    } else {
        param_.out->Resize(out_shape);
    }
    param_.out->set_layout(param_.resize_input->layout());
    SyncIO();
    return 0;
}

void ResizeConcatOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) {
        kernel_->SetParamOwner(std::make_shared<std::decay_t<decltype(param_)>>(param_));
    }
}

int32_t ResizeConcatOp::Run() {
    if (InferOutputShapes() != 0 || kernel_ == nullptr) {
        return -1;
    }
    return kernel_->compute();
}

}  // namespace operators
}  // namespace feather
