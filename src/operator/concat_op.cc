#include "src/operator/concat_op.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <utility>

#include "core/operator_registry.h"
#include "src/operator/tensor_op_utils.h"
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

bool CheckConcatShape(const ConcatParam& param) {
    if (param.inputs.size() < 2 || param.out == nullptr) {
        return false;
    }

    const auto& base_dims = param.inputs[0] == nullptr ? std::vector<int64_t>{} : param.inputs[0]->dims().data();
    if (base_dims.empty()) {
        return false;
    }
    int32_t axis = param.axis < 0 ? param.axis + static_cast<int32_t>(base_dims.size()) : param.axis;
    if (axis < 0 || axis >= static_cast<int32_t>(base_dims.size())) {
        return false;
    }

    for (const auto& input : param.inputs) {
        if (input == nullptr || input->dims().size() != base_dims.size()) {
            return false;
        }
        for (size_t i = 0; i < base_dims.size(); ++i) {
            if (static_cast<int32_t>(i) == axis) {
                continue;
            }
            if (input->dims()[i] != base_dims[i]) {
                return false;
            }
        }
    }
    return true;
}

std::vector<int64_t> InferConcatOutputShape(const ConcatParam& param) {
    std::vector<int64_t> out_shape = param.inputs[0]->dims().data();
    int32_t axis = param.axis < 0 ? param.axis + static_cast<int32_t>(out_shape.size()) : param.axis;
    out_shape[axis] = 0;
    for (const auto& input : param.inputs) {
        out_shape[axis] += input->dims()[axis];
    }
    return out_shape;
}

std::shared_ptr<OpBase> BuildConcatOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if (node.inputs.size() < 2 || node.outputs.size() != 1) {
        return nullptr;
    }

    ConcatParam param{};
    param.axis = GetIntAttribute(node.attributes, "axis", 1);
    param.out = tensors[node.outputs[0]];
    for (const auto& input_name : node.inputs) {
        param.inputs.push_back(tensors[input_name]);
    }

    auto op = std::make_shared<ConcatOp>(node.name.empty() ? "concat" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    auto kernel = CreateKernelForTensor(context.device, "Concat", param.inputs);
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_concat_op_registered = []() {
    OperatorRegistry::instance().Register("Concat", BuildConcatOp);
    return true;
}();

}  // namespace

void EnsureConcatOperatorRegistered() { (void)g_concat_op_registered; }

ConcatOp::ConcatOp() : OpBase("concat", "Concat") {}

ConcatOp::ConcatOp(const ConcatParam& param) : ConcatOp("concat", param) {}

ConcatOp::ConcatOp(std::string name, const ConcatParam& param) : OpBase(std::move(name), "Concat"), param_(param) {
    SyncIO();
}

void ConcatOp::SyncIO() {
    SetInputs(param_.inputs);
    SetOutputs({param_.out});
}

int32_t ConcatOp::CheckShape() const { return CheckConcatShape(param_) ? 0 : -1; }

int32_t ConcatOp::InferOutputShapes() {
    if (CheckShape() != 0) {
        return -1;
    }

    const auto out_shape = InferConcatOutputShape(param_);
    if (param_.inputs.empty() || param_.inputs.front() == nullptr) return -1;
    if (tensor_op_detail::InferSameTypeOutput(param_.inputs.front(), &param_.out, out_shape) != 0) return -1;
    param_.out->set_layout(param_.inputs.empty() || param_.inputs[0] == nullptr ? DataLayout::ND : param_.inputs[0]->layout());
    SyncIO();
    return 0;
}

void ConcatOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) {
        kernel_->SetParamOwner(std::make_shared<std::decay_t<decltype(param_)>>(param_));
    }
}

int32_t ConcatOp::Run() {
    if (InferOutputShapes() != 0 || kernel_ == nullptr) {
        return -1;
    }
    return kernel_->compute();
}

}  // namespace operators
}  // namespace feather
