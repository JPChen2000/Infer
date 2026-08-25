#include "src/operator/qwen_gated_delta_op.h"

#include <algorithm>
#include <numeric>
#include <utility>

#include "core/operator_registry.h"
#include "util/types.h"

namespace feather {
namespace operators {
namespace {

int64_t Product(const std::vector<int64_t>& dims) {
    return std::accumulate(dims.begin(), dims.end(), int64_t{1}, std::multiplies<int64_t>());
}

bool IsStateShape(const Tensor* tensor) {
    return tensor != nullptr && tensor->dims().size() == 4 && tensor->dims()[0] == 1 && tensor->dims()[2] > 0 &&
           tensor->dims()[3] > 0;
}

bool IsHeadVector(const Tensor* tensor, int64_t heads, int64_t width) {
    return tensor != nullptr && tensor->numel() == heads * width;
}

std::shared_ptr<OpBase> BuildState(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if (node.inputs.size() != 5 || node.outputs.size() != 1) return nullptr;
    QwenGatedDeltaStateParam param{};
    param.state = tensors[node.inputs[0]];
    param.k = tensors[node.inputs[1]];
    param.v = tensors[node.inputs[2]];
    param.beta = tensors[node.inputs[3]];
    param.decay = tensors[node.inputs[4]];
    param.out = tensors[node.outputs[0]];
    auto op = std::make_shared<QwenGatedDeltaStateOp>(node.name.empty() ? "qwen_gated_delta_state" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) return nullptr;
    auto kernel = CreateKernelForTensor(context.device, "QwenGatedDeltaState", {param.state, param.k, param.v, param.beta,
                                                                      param.decay, param.out}, DataType::FP32);
    if (kernel == nullptr) return nullptr;
    op->AttachKernel(std::move(kernel));
    return op;
}

std::shared_ptr<OpBase> BuildOutput(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if (node.inputs.size() != 2 || node.outputs.size() != 1) return nullptr;
    QwenGatedDeltaOutputParam param{};
    param.state = tensors[node.inputs[0]];
    param.q = tensors[node.inputs[1]];
    param.out = tensors[node.outputs[0]];
    auto op = std::make_shared<QwenGatedDeltaOutputOp>(node.name.empty() ? "qwen_gated_delta_output" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) return nullptr;
    auto kernel = CreateKernelForTensor(context.device, "QwenGatedDeltaOutput", {param.state, param.q, param.out}, DataType::FP32);
    if (kernel == nullptr) return nullptr;
    op->AttachKernel(std::move(kernel));
    return op;
}

std::shared_ptr<OpBase> BuildCombined(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if (node.inputs.size() != 6 || node.outputs.size() != 2) return nullptr;
    QwenGatedDeltaParam param{};
    param.state = tensors[node.inputs[0]];
    param.k = tensors[node.inputs[1]];
    param.v = tensors[node.inputs[2]];
    param.beta = tensors[node.inputs[3]];
    param.decay = tensors[node.inputs[4]];
    param.q = tensors[node.inputs[5]];
    param.next_state = tensors[node.outputs[0]];
    param.out = tensors[node.outputs[1]];
    auto op = std::make_shared<QwenGatedDeltaOp>(node.name.empty() ? "qwen_gated_delta" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) return nullptr;
    auto kernel = CreateKernelForTensor(context.device, "QwenGatedDelta",
                                            {param.state, param.k, param.v, param.beta, param.decay, param.q,
                                             param.next_state, param.out},
                                            DataType::FP32);
    if (kernel == nullptr) return nullptr;
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_registered = []() {
    OperatorRegistry::instance().Register("QwenGatedDeltaState", BuildState);
    OperatorRegistry::instance().Register("QwenGatedDeltaOutput", BuildOutput);
    OperatorRegistry::instance().Register("QwenGatedDelta", BuildCombined);
    return true;
}();

}  // namespace

void EnsureQwenGatedDeltaOperatorsRegistered() { (void)g_registered; }

QwenGatedDeltaStateOp::QwenGatedDeltaStateOp(std::string name, const QwenGatedDeltaStateParam& param)
    : OpBase(std::move(name), "QwenGatedDeltaState"), param_(param) {
    SyncIO();
}

void QwenGatedDeltaStateOp::SyncIO() {
    SetInputs({param_.state, param_.k, param_.v, param_.beta, param_.decay});
    SetOutputs({param_.out});
}

int32_t QwenGatedDeltaStateOp::CheckShape() const {
    if (!IsStateShape(param_.state.get()) || param_.out == nullptr || param_.k == nullptr || param_.v == nullptr ||
        param_.beta == nullptr || param_.decay == nullptr) return -1;
    const int64_t heads = param_.state->dims()[1];
    const int64_t key = param_.state->dims()[2];
    const int64_t value = param_.state->dims()[3];
    return IsHeadVector(param_.k.get(), heads, key) && IsHeadVector(param_.v.get(), heads, value) &&
                   IsHeadVector(param_.beta.get(), heads, 1) && IsHeadVector(param_.decay.get(), heads, 1)
               ? 0
               : -1;
}

int32_t QwenGatedDeltaStateOp::InferOutputShapes() {
    if (CheckShape() != 0) return -1;
    const auto shape = param_.state->dims().data();
    const size_t bytes = static_cast<size_t>(Product(shape)) * sizeof(float);
    if (!param_.out->IsInitialized() || param_.out->memory_size() < bytes) param_.out = std::make_shared<Tensor>(bytes);
    param_.out->Resize(shape);
    param_.out->set_data_type(DataType::FP32);
    SyncIO();
    return 0;
}

void QwenGatedDeltaStateOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) kernel_->SetParamOwner(std::make_shared<std::decay_t<decltype(param_)>>(param_));
}

int32_t QwenGatedDeltaStateOp::Run() { return InferOutputShapes() == 0 && kernel_ != nullptr ? kernel_->compute() : -1; }

QwenGatedDeltaOutputOp::QwenGatedDeltaOutputOp(std::string name, const QwenGatedDeltaOutputParam& param)
    : OpBase(std::move(name), "QwenGatedDeltaOutput"), param_(param) {
    SyncIO();
}

void QwenGatedDeltaOutputOp::SyncIO() { SetInputs({param_.state, param_.q}); SetOutputs({param_.out}); }

int32_t QwenGatedDeltaOutputOp::CheckShape() const {
    if (!IsStateShape(param_.state.get()) || param_.q == nullptr || param_.out == nullptr) return -1;
    const int64_t heads = param_.state->dims()[1];
    const int64_t key = param_.state->dims()[2];
    return IsHeadVector(param_.q.get(), heads, key) ? 0 : -1;
}

int32_t QwenGatedDeltaOutputOp::InferOutputShapes() {
    if (CheckShape() != 0) return -1;
    const std::vector<int64_t> shape = {param_.state->dims()[0], param_.state->dims()[1], param_.state->dims()[3]};
    const size_t bytes = static_cast<size_t>(Product(shape)) * sizeof(float);
    if (!param_.out->IsInitialized() || param_.out->memory_size() < bytes) param_.out = std::make_shared<Tensor>(bytes);
    param_.out->Resize(shape);
    param_.out->set_data_type(DataType::FP32);
    SyncIO();
    return 0;
}

void QwenGatedDeltaOutputOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) kernel_->SetParamOwner(std::make_shared<std::decay_t<decltype(param_)>>(param_));
}

int32_t QwenGatedDeltaOutputOp::Run() { return InferOutputShapes() == 0 && kernel_ != nullptr ? kernel_->compute() : -1; }

QwenGatedDeltaOp::QwenGatedDeltaOp(std::string name, const QwenGatedDeltaParam& param)
    : OpBase(std::move(name), "QwenGatedDelta"), param_(param) {
    SyncIO();
}

void QwenGatedDeltaOp::SyncIO() {
    SetInputs({param_.state, param_.k, param_.v, param_.beta, param_.decay, param_.q});
    SetOutputs({param_.next_state, param_.out});
}

int32_t QwenGatedDeltaOp::CheckShape() const {
    if (!IsStateShape(param_.state.get()) || param_.k == nullptr || param_.v == nullptr || param_.beta == nullptr ||
        param_.decay == nullptr || param_.q == nullptr || param_.next_state == nullptr || param_.out == nullptr) {
        return -1;
    }
    const int64_t heads = param_.state->dims()[1];
    const int64_t key = param_.state->dims()[2];
    const int64_t value = param_.state->dims()[3];
    return IsHeadVector(param_.k.get(), heads, key) && IsHeadVector(param_.v.get(), heads, value) &&
                   IsHeadVector(param_.beta.get(), heads, 1) && IsHeadVector(param_.decay.get(), heads, 1) &&
                   IsHeadVector(param_.q.get(), heads, key)
               ? 0
               : -1;
}

int32_t QwenGatedDeltaOp::InferOutputShapes() {
    if (CheckShape() != 0) return -1;
    const auto state_shape = param_.state->dims().data();
    const size_t state_bytes = static_cast<size_t>(Product(state_shape)) * sizeof(float);
    if (!param_.next_state->IsInitialized() || param_.next_state->memory_size() < state_bytes) {
        param_.next_state = std::make_shared<Tensor>(state_bytes);
    }
    param_.next_state->Resize(state_shape);
    param_.next_state->set_data_type(DataType::FP32);

    const std::vector<int64_t> output_shape = {state_shape[0], state_shape[1], state_shape[3]};
    const size_t output_bytes = static_cast<size_t>(Product(output_shape)) * sizeof(float);
    if (!param_.out->IsInitialized() || param_.out->memory_size() < output_bytes) {
        param_.out = std::make_shared<Tensor>(output_bytes);
    }
    param_.out->Resize(output_shape);
    param_.out->set_data_type(DataType::FP32);
    SyncIO();
    return 0;
}

void QwenGatedDeltaOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) kernel_->SetParamOwner(std::make_shared<std::decay_t<decltype(param_)>>(param_));
}

int32_t QwenGatedDeltaOp::Run() { return InferOutputShapes() == 0 && kernel_ != nullptr ? kernel_->compute() : -1; }

}  // namespace operators
}  // namespace feather
