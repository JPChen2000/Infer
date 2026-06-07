#include "src/operator/pool_op.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <utility>

#include "core/operator_registry.h"
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

bool CheckPoolShape(const PoolParam& param) {
    if (param.input == nullptr || param.out == nullptr) {
        return false;
    }
    if (param.input->dims().size() != 2 && param.input->dims().size() != 4) {
        return false;
    }
    if (param.kernel_h <= 0 || param.kernel_w <= 0 || param.stride_h <= 0 || param.stride_w <= 0 ||
        param.pad_h < 0 || param.pad_w < 0) {
        return false;
    }
    const auto height_axis = param.input->dims().size() == 2 ? 0 : 2;
    const auto width_axis = param.input->dims().size() == 2 ? 1 : 3;
    const int64_t out_h = (param.input->dims()[height_axis] - param.kernel_h + 2 * param.pad_h) / param.stride_h + 1;
    const int64_t out_w = (param.input->dims()[width_axis] - param.kernel_w + 2 * param.pad_w) / param.stride_w + 1;
    return out_h > 0 && out_w > 0;
}

std::vector<int64_t> InferPoolOutputShape(const PoolParam& param) {
    if (param.input->dims().size() == 2) {
        return {
            (param.input->dims()[0] - param.kernel_h + 2 * param.pad_h) / param.stride_h + 1,
            (param.input->dims()[1] - param.kernel_w + 2 * param.pad_w) / param.stride_w + 1,
        };
    }
    return {
        param.input->dims()[0],
        param.input->dims()[1],
        (param.input->dims()[2] - param.kernel_h + 2 * param.pad_h) / param.stride_h + 1,
        (param.input->dims()[3] - param.kernel_w + 2 * param.pad_w) / param.stride_w + 1,
    };
}

void AttachPoolKernel(std::unique_ptr<KernelBase>& slot, PoolParam* param, std::unique_ptr<KernelBase> kernel) {
    slot = std::move(kernel);
    if (slot != nullptr) {
        slot->SetParam((void*)param);
    }
}

PoolParam BuildPoolParam(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors) {
    PoolParam param{};
    param.input = tensors[node.inputs[0]];
    param.out = tensors[node.outputs[0]];
    param.kernel_h = GetIntAttribute(node.attributes, "kernel_h", 1);
    param.kernel_w = GetIntAttribute(node.attributes, "kernel_w", 1);
    param.stride_h = GetIntAttribute(node.attributes, "stride_h", 1);
    param.stride_w = GetIntAttribute(node.attributes, "stride_w", 1);
    param.pad_h = GetIntAttribute(node.attributes, "pad_h", 0);
    param.pad_w = GetIntAttribute(node.attributes, "pad_w", 0);
    return param;
}

std::unique_ptr<KernelBase> CreateAvgPoolKernel() {
    kernel::EnsurePoolKernelsRegistered();
    return CreateHostKernelForTensor("AvgPool", {});
}

std::unique_ptr<KernelBase> CreateMaxPoolKernel() {
    kernel::EnsurePoolKernelsRegistered();
    return CreateHostKernelForTensor("MaxPool", {});
}

std::shared_ptr<OpBase> BuildAvgPoolOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors) {
    if (node.inputs.size() != 1 || node.outputs.size() != 1) {
        return nullptr;
    }
    PoolParam param = BuildPoolParam(node, tensors);
    if (param.input == nullptr || param.out == nullptr) {
        return nullptr;
    }
    auto op = std::make_shared<AvgPoolOp>(node.name.empty() ? "avgpool" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    auto kernel = CreateHostKernelForTensor("AvgPool", {param.input, param.out});
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

std::shared_ptr<OpBase> BuildMaxPoolOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors) {
    if (node.inputs.size() != 1 || node.outputs.size() != 1) {
        return nullptr;
    }
    PoolParam param = BuildPoolParam(node, tensors);
    if (param.input == nullptr || param.out == nullptr) {
        return nullptr;
    }
    auto op = std::make_shared<MaxPoolOp>(node.name.empty() ? "maxpool" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    auto kernel = CreateHostKernelForTensor("MaxPool", {param.input, param.out});
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_avg_pool_registered = []() {
    OperatorRegistry::instance().Register("AvgPool", BuildAvgPoolOp);
    return true;
}();

bool g_max_pool_registered = []() {
    OperatorRegistry::instance().Register("MaxPool", BuildMaxPoolOp);
    return true;
}();

}  // namespace

void EnsurePoolOperatorsRegistered() {
    (void)g_avg_pool_registered;
    (void)g_max_pool_registered;
}

AvgPoolOp::AvgPoolOp() : OpBase("avgpool", "AvgPool") {}

AvgPoolOp::AvgPoolOp(const PoolParam& param) : AvgPoolOp("avgpool", param) {}

AvgPoolOp::AvgPoolOp(std::string name, const PoolParam& param) : OpBase(std::move(name), "AvgPool"), param_(param) {
    SyncIO();
}

void AvgPoolOp::SyncIO() {
    SetInputs({param_.input});
    SetOutputs({param_.out});
}

int32_t AvgPoolOp::CheckShape() const { return CheckPoolShape(param_) ? 0 : -1; }

int32_t AvgPoolOp::InferOutputShapes() {
    if (CheckShape() != 0) {
        return -1;
    }
    const auto out_shape = InferPoolOutputShape(param_);
    const size_t required_bytes =
        static_cast<size_t>(ComputeNumel(out_shape)) *
        DataTypeBytes(ResolveExecutionDataType({param_.input, param_.out}, DataType::FP32));
    if (param_.out == nullptr || !param_.out->IsInitialized() || param_.out->memory_size() < required_bytes) {
        param_.out = std::make_shared<Tensor>(out_shape);
    } else {
        param_.out->Resize(out_shape);
    }
    SyncIO();
    return 0;
}

void AvgPoolOp::AttachKernel(std::unique_ptr<KernelBase> kernel) { AttachPoolKernel(kernel_, &param_, std::move(kernel)); }

int32_t AvgPoolOp::Run() {
    if (InferOutputShapes() != 0 || kernel_ == nullptr) {
        return -1;
    }
    return kernel_->compute();
}

MaxPoolOp::MaxPoolOp() : OpBase("maxpool", "MaxPool") {}

MaxPoolOp::MaxPoolOp(const PoolParam& param) : MaxPoolOp("maxpool", param) {}

MaxPoolOp::MaxPoolOp(std::string name, const PoolParam& param) : OpBase(std::move(name), "MaxPool"), param_(param) {
    SyncIO();
}

void MaxPoolOp::SyncIO() {
    SetInputs({param_.input});
    SetOutputs({param_.out});
}

int32_t MaxPoolOp::CheckShape() const { return CheckPoolShape(param_) ? 0 : -1; }

int32_t MaxPoolOp::InferOutputShapes() {
    if (CheckShape() != 0) {
        return -1;
    }
    const auto out_shape = InferPoolOutputShape(param_);
    const size_t required_bytes =
        static_cast<size_t>(ComputeNumel(out_shape)) *
        DataTypeBytes(ResolveExecutionDataType({param_.input, param_.out}, DataType::FP32));
    if (param_.out == nullptr || !param_.out->IsInitialized() || param_.out->memory_size() < required_bytes) {
        param_.out = std::make_shared<Tensor>(out_shape);
    } else {
        param_.out->Resize(out_shape);
    }
    SyncIO();
    return 0;
}

void MaxPoolOp::AttachKernel(std::unique_ptr<KernelBase> kernel) { AttachPoolKernel(kernel_, &param_, std::move(kernel)); }

int32_t MaxPoolOp::Run() {
    if (InferOutputShapes() != 0 || kernel_ == nullptr) {
        return -1;
    }
    return kernel_->compute();
}

}  // namespace operators
}  // namespace feather
