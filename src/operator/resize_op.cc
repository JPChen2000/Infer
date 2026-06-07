#include "src/operator/resize_op.h"

#include <cmath>
#include <utility>

#include "core/operator_registry.h"
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

std::shared_ptr<OpBase> BuildResizeOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors) {
    if (node.inputs.size() != 1 || node.outputs.size() != 1) {
        return nullptr;
    }

    ResizeParam param{};
    param.input = tensors[node.inputs[0]];
    param.out = tensors[node.outputs[0]];
    param.scales = GetFloatVectorAttribute(node.attributes, "scales");
    if (param.input == nullptr || param.out == nullptr || param.scales.empty()) {
        return nullptr;
    }

    auto op = std::make_shared<ResizeOp>(node.name.empty() ? "resize" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    auto kernel = CreateHostKernelForTensor("Resize", {param.input, param.out});
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
    SetInputs({param_.input});
    SetOutputs({param_.out});
}

int32_t ResizeOp::CheckShape() const {
    if (param_.input == nullptr || param_.out == nullptr || param_.scales.empty()) {
        return -1;
    }
    if (param_.input->dims().size() != param_.scales.size()) {
        return -1;
    }
    for (float scale : param_.scales) {
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
    out_shape.reserve(param_.scales.size());
    for (size_t i = 0; i < param_.scales.size(); ++i) {
        const auto dim = static_cast<int64_t>(std::llround(static_cast<double>(param_.input->dims()[i]) * param_.scales[i]));
        if (dim <= 0) {
            return -1;
        }
        out_shape.push_back(dim);
    }

    const size_t required_bytes =
        static_cast<size_t>(std::max<int64_t>(1, param_.input->numel())) *
        DataTypeBytes(ResolveExecutionDataType({param_.input, param_.out}, DataType::FP32));
    if (param_.out == nullptr || !param_.out->IsInitialized() || param_.out->memory_size() < required_bytes) {
        param_.out = std::make_shared<Tensor>(out_shape);
    } else {
        param_.out->Resize(out_shape);
    }
    SyncIO();
    return 0;
}

void ResizeOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) {
        kernel_->SetParam((void*)&param_);
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
