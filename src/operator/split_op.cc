#include "src/operator/split_op.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <utility>

#include "core/operator_registry.h"
#include "src/operator/control_tensor.h"
#include "util/types.h"

namespace feather {
namespace operators {

namespace {

int64_t ComputeNumel(const std::vector<int64_t>& dims) {
    return std::accumulate(dims.begin(), dims.end(), int64_t{1}, std::multiplies<int64_t>());
}

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

bool CheckSplitShape(const SplitParam& param) {
    if (param.input == nullptr || param.outputs.empty()) {
        return false;
    }
    std::vector<int64_t> split_sizes = param.split_sizes;
    if (param.split != nullptr && !ReadIntegerTensor(param.split, &split_sizes)) {
        return false;
    }
    if (param.input->dims().empty() || param.outputs.size() != split_sizes.size()) {
        return false;
    }
    int32_t axis = param.axis < 0 ? param.axis + static_cast<int32_t>(param.input->dims().size()) : param.axis;
    if (axis < 0 || axis >= static_cast<int32_t>(param.input->dims().size())) {
        return false;
    }
    for (const auto& output : param.outputs) {
        if (output == nullptr) {
            return false;
        }
    }
    for (const auto split_size : split_sizes) {
        if (split_size <= 0) {
            return false;
        }
    }
    const int64_t total = std::accumulate(split_sizes.begin(), split_sizes.end(), static_cast<int64_t>(0));
    return total == param.input->dims()[axis];
}

std::vector<int64_t> InferSplitOutputShape(const SplitParam& param, size_t output_index) {
    std::vector<int64_t> out_shape = param.input->dims().data();
    int32_t axis = param.axis < 0 ? param.axis + static_cast<int32_t>(out_shape.size()) : param.axis;
    std::vector<int64_t> split_sizes = param.split_sizes;
    if (param.split != nullptr && !ReadIntegerTensor(param.split, &split_sizes)) {
        return {};
    }
    out_shape[axis] = split_sizes[output_index];
    return out_shape;
}

std::shared_ptr<OpBase> BuildSplitOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if ((node.inputs.size() != 1 && node.inputs.size() != 2) || node.outputs.size() < 2) {
        return nullptr;
    }

    SplitParam param{};
    param.input = tensors[node.inputs[0]];
    if (node.inputs.size() == 2) {
        param.split = tensors[node.inputs[1]];
    }
    param.axis = GetIntAttribute(node.attributes, "axis", 1);
    param.split_sizes = GetShapeAttribute(node.attributes, "split_sizes");
    for (const auto& output_name : node.outputs) {
        param.outputs.push_back(tensors[output_name]);
    }

    auto op = std::make_shared<SplitOp>(node.name.empty() ? "split" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    auto kernel = CreateKernelForTensor(context.device, "Split", {param.input});
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_split_op_registered = []() {
    OperatorRegistry::instance().Register("Split", BuildSplitOp);
    return true;
}();

}  // namespace

void EnsureSplitOperatorRegistered() { (void)g_split_op_registered; }

SplitOp::SplitOp() : OpBase("split", "Split") {}

SplitOp::SplitOp(const SplitParam& param) : SplitOp("split", param) {}

SplitOp::SplitOp(std::string name, const SplitParam& param) : OpBase(std::move(name), "Split"), param_(param) {
    SyncIO();
}

void SplitOp::SyncIO() {
    std::vector<std::shared_ptr<Tensor>> inputs = {param_.input};
    if (param_.split != nullptr) {
        inputs.push_back(param_.split);
    }
    SetInputs(std::move(inputs));
    SetOutputs(param_.outputs);
}

int32_t SplitOp::CheckShape() const { return CheckSplitShape(param_) ? 0 : -1; }

int32_t SplitOp::InferOutputShapes() {
    if (CheckShape() != 0) {
        return -1;
    }

    if (param_.split != nullptr && !ReadIntegerTensor(param_.split, &param_.split_sizes)) {
        return -1;
    }
    for (size_t i = 0; i < param_.outputs.size(); ++i) {
        const auto out_shape = InferSplitOutputShape(param_, i);
        if (out_shape.empty()) {
            return -1;
        }
        const size_t required_bytes =
            static_cast<size_t>(ComputeNumel(out_shape)) *
            DataTypeBytes(ResolveExecutionDataType({param_.input, param_.outputs[i]}, DataType::FP32));
        if (param_.outputs[i] == nullptr || !param_.outputs[i]->IsInitialized() ||
            param_.outputs[i]->memory_size() < required_bytes) {
            param_.outputs[i] = std::make_shared<Tensor>(out_shape);
        } else {
            param_.outputs[i]->Resize(out_shape);
        }
        param_.outputs[i]->set_layout(param_.input->layout());
    }
    SyncIO();
    return 0;
}

void SplitOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) {
        kernel_->SetParamOwner(std::make_shared<std::decay_t<decltype(param_)>>(param_));
    }
}

int32_t SplitOp::Run() {
    if (InferOutputShapes() != 0 || kernel_ == nullptr) {
        return -1;
    }
    return kernel_->compute();
}

}  // namespace operators
}  // namespace feather
