// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#include "src/operator/non_max_suppression_op.h"

#include <cmath>
#include <cstring>
#include <utility>

#include "core/operator_registry.h"
#include "core/shape_utils.h"

namespace feather {
namespace operators {
namespace {

bool ScalarShape(const std::shared_ptr<Tensor>& tensor, DataType type) {
    if (!tensor) return true;
    const auto& dims = tensor->dims().data();
    return tensor->data_type() == type &&
           (dims.empty() || (dims.size() == 1 && dims[0] == 1));
}

bool CapacityRows(const NmsParam& p, int64_t* rows) {
    if (!p.boxes || !p.scores || p.boxes->dims().size() != 3 || p.scores->dims().size() != 3) return false;
    size_t count = 0;
    if (!CheckedElementCount({p.boxes->dims()[0], p.scores->dims()[1], p.boxes->dims()[1]}, &count)) return false;
    size_t bytes = 0;
    if (!CheckedByteCount({static_cast<int64_t>(count), 3}, sizeof(int64_t), &bytes)) return false;
    *rows = static_cast<int64_t>(count);
    return true;
}

bool HasBytes(const std::shared_ptr<Tensor>& tensor, size_t bytes) {
    return tensor && tensor->IsInitialized() && tensor->storage() &&
           tensor->offset() <= tensor->storage()->size() &&
           bytes <= tensor->storage()->size() - tensor->offset();
}

class NonMaxSuppressionOp final : public OpBase {
 public:
    NonMaxSuppressionOp(std::string name, NmsParam param)
        : OpBase(std::move(name), "NonMaxSuppression"), param_(std::move(param)) {
        std::vector<std::shared_ptr<Tensor>> inputs{param_.boxes, param_.scores};
        for (const auto& control : {param_.max_output, param_.iou_threshold, param_.score_threshold}) {
            if (control) inputs.push_back(control);
        }
        SetInputs(std::move(inputs));
        SetOutputs({param_.out});
    }
    int32_t CheckShape() const override { return ValidateNms(param_) ? 0 : -1; }
    int32_t InferOutputShapes() override { return PrepareNmsOutput(param_) ? 0 : -1; }
    void AttachKernel(std::unique_ptr<KernelBase> kernel) override {
        kernel_ = std::move(kernel);
        RefreshKernelParams();
    }
    std::unique_ptr<KernelBase> DetachKernel() override { return std::move(kernel_); }
    bool HasKernel() const override { return kernel_ != nullptr; }
    const KernelBase* kernel() const override { return kernel_.get(); }
    void RefreshKernelParams() override {
        if (kernel_) kernel_->SetParamOwner(std::make_shared<NmsParam>(param_));
    }
    int32_t Run() override {
        if (!kernel_ || InferOutputShapes() != 0) return -1;
        return kernel_->compute();
    }
 private:
    NmsParam param_;
    std::unique_ptr<KernelBase> kernel_;
};

std::shared_ptr<OpBase> BuildNonMaxSuppression(const model::NodeDesc& node,
        OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if (node.inputs.size() < 2 || node.inputs.size() > 5 || node.outputs.size() != 1 ||
        node.inputs[0].empty() || node.inputs[1].empty() || node.outputs[0].empty()) return nullptr;
    auto lookup = [&](const std::string& name) -> std::shared_ptr<Tensor> {
        const auto it = tensors.find(name);
        return it == tensors.end() ? nullptr : it->second;
    };
    NmsParam p;
    p.boxes = lookup(node.inputs[0]);
    p.scores = lookup(node.inputs[1]);
    p.out = lookup(node.outputs[0]);
    std::shared_ptr<Tensor>* optional[] = {&p.max_output, &p.iou_threshold, &p.score_threshold};
    for (size_t i = 2; i < node.inputs.size(); ++i) {
        if (!node.inputs[i].empty()) {
            *optional[i - 2] = lookup(node.inputs[i]);
            if (!*optional[i - 2]) return nullptr;
        }
    }
    const auto center = node.attributes.find("center_point_box");
    if (center != node.attributes.end()) {
        const auto* value = std::get_if<int64_t>(&center->second);
        if (!value) return nullptr;
        p.center_point_box = *value;
    }
    if (!ValidateNms(p) || !CapacityRows(p, &p.capacity_rows)) return nullptr;
    auto op = std::make_shared<NonMaxSuppressionOp>(node.name.empty() ? "NonMaxSuppression" : node.name, p);
    if (op->InferOutputShapes() != 0) return nullptr;
    kernel::EnsureCommonNonMaxSuppressionKernelsRegistered();
    auto kernel = CreateKernelForTensor(context.device, "NonMaxSuppression",
        {p.boxes, p.scores, p.max_output, p.iou_threshold, p.score_threshold, p.out}, DataType::FP32);
    if (!kernel) return nullptr;
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_registered = [] {
    OperatorRegistry::instance().Register("NonMaxSuppression", BuildNonMaxSuppression);
    return true;
}();
}  // namespace

bool ValidateNms(const NmsParam& p) {
    if (!p.boxes || !p.scores || !p.out || p.boxes->data_type() != DataType::FP32 ||
        p.scores->data_type() != DataType::FP32 ||
        (p.out->data_type() != DataType::UNKNOWN && p.out->data_type() != DataType::INT64) ||
        (p.center_point_box != 0 && p.center_point_box != 1) || p.capacity_rows < 0) return false;
    for (const auto& input : {p.boxes, p.scores, p.max_output, p.iou_threshold, p.score_threshold})
        if (input && input == p.out) return false;
    const auto& boxes = p.boxes->dims().data();
    const auto& scores = p.scores->dims().data();
    if (boxes.size() != 3 || scores.size() != 3 || boxes[2] != 4 ||
        boxes[0] != scores[0] || boxes[1] != scores[2]) return false;
    size_t bytes = 0;
    int64_t rows = 0;
    if (!(CheckedByteCount(boxes, sizeof(float), &bytes) &&
           CheckedByteCount(scores, sizeof(float), &bytes) && CapacityRows(p, &rows) &&
           ScalarShape(p.max_output, DataType::INT64) &&
           ScalarShape(p.iou_threshold, DataType::FP32) && ScalarShape(p.score_threshold, DataType::FP32))) return false;
    size_t output_bytes = 0;
    if (!CheckedByteCount({p.capacity_rows, 3}, sizeof(int64_t), &output_bytes)) return false;
    // Disjoint views in a session arena are valid; overlapping views would
    // overwrite inputs and controls needed by subsequent runs.
    for (const auto& input : {p.boxes, p.scores, p.max_output, p.iou_threshold, p.score_threshold}) {
        if (!input || !p.out->storage() || input->storage() != p.out->storage()) continue;
        size_t input_bytes = 0;
        if (!CheckedByteCount(input->dims().data(), DataTypeBytes(input->data_type()), &input_bytes)) return false;
        if (output_bytes && input_bytes &&
            (p.out->offset() <= input->offset()
                ? input->offset() - p.out->offset() < output_bytes
                : p.out->offset() - input->offset() < input_bytes)) return false;
    }
    return true;
}

bool ReadNmsControls(const NmsParam& p, int64_t* maximum, float* iou, float* score) {
    if (!maximum || !iou || !score || !ValidateNms(p)) return false;
    *maximum = 0;
    *iou = 0.0f;
    *score = 0.0f;
    if (p.max_output) {
        if (!HasBytes(p.max_output, sizeof(int64_t))) return false;
        *maximum = p.max_output->data<int64_t>()[0];
    }
    if (p.iou_threshold) {
        if (!HasBytes(p.iou_threshold, sizeof(float))) return false;
        *iou = p.iou_threshold->data<float>()[0];
    }
    if (p.score_threshold) {
        if (!HasBytes(p.score_threshold, sizeof(float))) return false;
        *score = p.score_threshold->data<float>()[0];
    }
    return *maximum >= 0 && std::isfinite(*iou) && *iou >= 0.0f && *iou <= 1.0f && std::isfinite(*score);
}

bool PrepareNmsOutput(const NmsParam& p) {
    int64_t rows = 0;
    size_t bytes = 0;
    if (!ValidateNms(p) || !CapacityRows(p, &rows) || rows > p.capacity_rows ||
        !CheckedByteCount({p.capacity_rows, 3}, sizeof(int64_t), &bytes)) return false;
    // Keep the compiled allocation even after a previous run shrank the logical shape.
    if (!HasBytes(p.out, bytes)) {
        if (DeferredAllocationScope::Enabled()) p.out->ResetStorage(std::make_shared<Storage>(bytes), bytes);
        else p.out->ResetBuffer(std::make_shared<Buffer>(bytes), bytes);
    }
    p.out->Resize(std::vector<int64_t>{p.capacity_rows, 3});
    p.out->set_data_type(DataType::INT64);
    p.out->set_layout(DataLayout::ND);
    return true;
}

bool FinishNmsOutput(const NmsParam& p, const std::vector<int64_t>& indices) {
    if (indices.size() % 3 != 0 || p.capacity_rows < 0 ||
        indices.size() / 3 > static_cast<uint64_t>(p.capacity_rows) || !PrepareNmsOutput(p)) return false;
    if (!indices.empty()) {
        std::memcpy(p.out->mutable_data<int64_t>(), indices.data(), indices.size() * sizeof(int64_t));
    }
    p.out->Resize(std::vector<int64_t>{static_cast<int64_t>(indices.size() / 3), 3});
    return true;
}

void EnsureNonMaxSuppressionOperatorRegistered() { (void)g_registered; }
}  // namespace operators
}  // namespace feather
