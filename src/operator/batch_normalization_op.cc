#include "src/operator/batch_normalization_op.h"

#include <cmath>
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

float GetFloatAttribute(const std::unordered_map<std::string, model::AttributeValue>& attributes,
                        const std::string& key, float default_value) {
    auto it = attributes.find(key);
    if (it == attributes.end()) {
        return default_value;
    }
    if (auto value = std::get_if<float>(&it->second); value != nullptr) {
        return *value;
    }
    return default_value;
}

std::shared_ptr<OpBase> BuildBatchNormalizationOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if (node.inputs.size() != 5 || node.outputs.size() != 1) {
        return nullptr;
    }

    BatchNormParam param{};
    param.input = tensors[node.inputs[0]];
    param.scale = tensors[node.inputs[1]];
    param.bias = tensors[node.inputs[2]];
    param.mean = tensors[node.inputs[3]];
    param.var = tensors[node.inputs[4]];
    param.out = tensors[node.outputs[0]];
    param.epsilon = GetFloatAttribute(node.attributes, "epsilon", 1e-5f);
    if (param.input == nullptr || param.scale == nullptr || param.bias == nullptr || param.mean == nullptr ||
        param.var == nullptr || param.out == nullptr) {
        return nullptr;
    }

    auto op = std::make_shared<BatchNormalizationOp>(node.name.empty() ? "batchnorm" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    auto kernel = CreateKernelForTensor(context.device, "BatchNormalization", {param.input, param.scale, param.bias, param.mean, param.var, param.out});
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_batch_normalization_op_registered = []() {
    OperatorRegistry::instance().Register("BatchNormalization", BuildBatchNormalizationOp);
    return true;
}();

}  // namespace

void EnsureBatchNormalizationOperatorRegistered() { (void)g_batch_normalization_op_registered; }

BatchNormalizationOp::BatchNormalizationOp() : OpBase("batchnorm", "BatchNormalization") {}

BatchNormalizationOp::BatchNormalizationOp(const BatchNormParam& param) : BatchNormalizationOp("batchnorm", param) {}

BatchNormalizationOp::BatchNormalizationOp(std::string name, const BatchNormParam& param)
    : OpBase(std::move(name), "BatchNormalization"), param_(param) {
    SyncIO();
}

void BatchNormalizationOp::SyncIO() {
    SetInputs({param_.input, param_.scale, param_.bias, param_.mean, param_.var});
    SetOutputs({param_.out});
}

int32_t BatchNormalizationOp::CheckShape() const {
    if (param_.input == nullptr || param_.scale == nullptr || param_.bias == nullptr || param_.mean == nullptr ||
        param_.var == nullptr || param_.out == nullptr) {
        return -1;
    }
    if (param_.input->dims().size() != 4) {
        return -1;
    }
    ImageShape4D shape{};
    if (!DecodeImageShape4D(param_.input->dims().data(), NormalizeDataLayout(param_.input->layout()), &shape)) {
        return -1;
    }
    return param_.scale->numel() == shape.c && param_.bias->numel() == shape.c && param_.mean->numel() == shape.c &&
           param_.var->numel() == shape.c
               ? 0
               : -1;
}

int32_t BatchNormalizationOp::InferOutputShapes() {
    if (CheckShape() != 0) {
        return -1;
    }
    const auto out_shape = param_.input->dims().data();
    const size_t required_bytes =
        static_cast<size_t>(param_.input->numel()) *
        DataTypeBytes(ResolveExecutionDataType({param_.input, param_.out}, DataType::FP32));
    if (param_.out == nullptr || !param_.out->IsInitialized() || param_.out->memory_size() < required_bytes) {
        param_.out = std::make_shared<Tensor>(out_shape);
    } else {
        param_.out->Resize(out_shape);
    }
    param_.out->set_layout(param_.input->layout());
    SyncIO();
    return 0;
}

void BatchNormalizationOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) {
        kernel_->SetParamOwner(std::make_shared<std::decay_t<decltype(param_)>>(param_));
    }
}

int32_t BatchNormalizationOp::Run() {
    if (InferOutputShapes() != 0 || kernel_ == nullptr) {
        return -1;
    }
    return kernel_->compute();
}

}  // namespace operators
}  // namespace feather
