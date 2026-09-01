#include "src/operator/resize_op.h"

#include <cmath>
#include <functional>
#include <numeric>
#include <utility>

#include "core/operator_registry.h"
#include "src/operator/tensor_op_utils.h"
#include "src/operator/control_tensor.h"
#include "util/types.h"

namespace feather {
namespace operators {

namespace {

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

std::shared_ptr<OpBase> BuildResizeOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if (node.inputs.empty() || node.inputs.size() > 4 || node.outputs.size() != 1) {
        return nullptr;
    }

    ResizeParam param{};
    param.input = tensors[node.inputs[0]];
    if (node.inputs.size() == 2) {
        param.scales_tensor = tensors[node.inputs[1]];
    } else {
        if (node.inputs.size() > 1 && !node.inputs[1].empty()) {
            param.roi = tensors[node.inputs[1]];
        }
        if (node.inputs.size() > 2 && !node.inputs[2].empty()) {
            param.scales_tensor = tensors[node.inputs[2]];
        }
    }
    if (node.inputs.size() > 3 && !node.inputs[3].empty()) {
        param.sizes = tensors[node.inputs[3]];
    }
    param.out = tensors[node.outputs[0]];
    param.scales = GetFloatVectorAttribute(node.attributes, "scales");
    if (param.input == nullptr || param.out == nullptr ||
        (param.scales.empty() && param.scales_tensor == nullptr && param.sizes == nullptr)) {
        return nullptr;
    }

    auto op = std::make_shared<ResizeOp>(node.name.empty() ? "resize" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    auto kernel = CreateKernelForTensor(context.device, "Resize", {param.input, param.out});
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_resize_op_registered = []() {
    OperatorRegistry::instance().Register("Resize", BuildResizeOp);
    return true;
}();

}  // namespace

void EnsureResizeOperatorRegistered() { (void)g_resize_op_registered; }

ResizeOp::ResizeOp() : OpBase("resize", "Resize") {}

ResizeOp::ResizeOp(const ResizeParam& param) : ResizeOp("resize", param) {}

ResizeOp::ResizeOp(std::string name, const ResizeParam& param) : OpBase(std::move(name), "Resize"), param_(param) {
    SyncIO();
}

void ResizeOp::SyncIO() {
    std::vector<std::shared_ptr<Tensor>> inputs = {param_.input};
    if (param_.roi != nullptr || param_.scales_tensor != nullptr || param_.sizes != nullptr) {
        inputs.push_back(param_.roi == nullptr ? std::shared_ptr<Tensor>() : param_.roi);
    }
    if (param_.scales_tensor != nullptr || param_.sizes != nullptr) {
        inputs.push_back(param_.scales_tensor == nullptr ? std::shared_ptr<Tensor>() : param_.scales_tensor);
    }
    if (param_.sizes != nullptr) {
        inputs.push_back(param_.sizes);
    }
    SetInputs(std::move(inputs));
    SetOutputs({param_.out});
}

int32_t ResizeOp::CheckShape() const {
    if (param_.input == nullptr || param_.out == nullptr) {
        return -1;
    }
    if (param_.sizes != nullptr) {
        std::vector<int64_t> sizes;
        if (!ReadIntegerTensor(param_.sizes, &sizes) || sizes.size() != param_.input->dims().size()) {
            return -1;
        }
        for (const auto size : sizes) {
            if (size <= 0) {
                return -1;
            }
        }
        return 0;
    }
    std::vector<float> scales = param_.scales;
    if (param_.scales_tensor != nullptr && !ReadFloatTensor(param_.scales_tensor, &scales)) {
        return -1;
    }
    if (scales.empty() || param_.input->dims().size() != scales.size()) {
        return -1;
    }
    for (const float scale : scales) {
        if (scale <= 0.0f) {
            return -1;
        }
    }
    return 0;
}

int32_t ResizeOp::InferOutputShapes() {
    if (CheckShape() != 0) {
        return -1;
    }

    std::vector<int64_t> out_shape;
    if (param_.sizes != nullptr) {
        if (!ReadIntegerTensor(param_.sizes, &out_shape)) {
            return -1;
        }
        param_.scales.resize(out_shape.size());
        for (size_t i = 0; i < out_shape.size(); ++i) {
            param_.scales[i] = static_cast<float>(out_shape[i]) / static_cast<float>(param_.input->dims()[i]);
        }
    } else {
        std::vector<float> scales = param_.scales;
        if (param_.scales_tensor != nullptr && !ReadFloatTensor(param_.scales_tensor, &scales)) {
            return -1;
        }
        param_.scales = std::move(scales);
        out_shape.reserve(param_.scales.size());
        for (size_t i = 0; i < param_.scales.size(); ++i) {
            const auto dim = static_cast<int64_t>(
                std::llround(static_cast<double>(param_.input->dims()[i]) * param_.scales[i]));
            if (dim <= 0) {
                return -1;
            }
            out_shape.push_back(dim);
        }
    }

    const size_t required_bytes =
        static_cast<size_t>(std::max<int64_t>(1, std::accumulate(out_shape.begin(), out_shape.end(), int64_t{1},
                                                                 std::multiplies<int64_t>()))) *
        DataTypeBytes(ResolveExecutionDataType({param_.input, param_.out}, DataType::FP32));
    if (param_.out == nullptr || !param_.out->IsInitialized() || param_.out->memory_size() < required_bytes) {
        param_.out = std::make_shared<Tensor>(out_shape);
    } else {
        param_.out->Resize(out_shape);
    }
    param_.out->set_data_type(param_.input->data_type());
    param_.out->set_layout(param_.input->layout());
    SyncIO();
    return 0;
}

void ResizeOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) {
        kernel_->SetParamOwner(std::make_shared<std::decay_t<decltype(param_)>>(param_));
    }
}

int32_t ResizeOp::Run() {
    if (InferOutputShapes() != 0 || kernel_ == nullptr) {
        return -1;
    }
    return kernel_->compute();
}

}  // namespace operators
}  // namespace feather
