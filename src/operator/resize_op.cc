// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "src/operator/resize_op.h"

#include <cmath>
#include <functional>
#include <limits>
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

bool GetResizeString(const model::NodeDesc& node, const std::string& name, std::string* value) {
    const auto it = node.attributes.find(name);
    if (it == node.attributes.end()) return true;
    const auto* attr = std::get_if<std::string>(&it->second);
    if (attr == nullptr) return false;
    *value = *attr;
    return true;
}

bool ReadResizeAttributes(const model::NodeDesc& node, ResizeParam* param) {
    std::string mode = "nearest", coordinate = "asymmetric", nearest = "floor";
    if (!GetResizeString(node, "mode", &mode) ||
        !GetResizeString(node, "coordinate_transformation_mode", &coordinate) ||
        !GetResizeString(node, "nearest_mode", &nearest)) return false;
    if (mode == "linear") param->mode = ResizeParam::Mode::kLinear;
    else if (mode != "nearest") return false;
    if (coordinate == "half_pixel") param->coordinate_mode = ResizeParam::CoordinateMode::kHalfPixel;
    else if (coordinate == "pytorch_half_pixel") param->coordinate_mode = ResizeParam::CoordinateMode::kPytorchHalfPixel;
    else if (coordinate == "align_corners") param->coordinate_mode = ResizeParam::CoordinateMode::kAlignCorners;
    else if (coordinate != "asymmetric") return false;
    if (nearest == "ceil") param->nearest_mode = ResizeParam::NearestMode::kCeil;
    else if (nearest == "round_prefer_floor") param->nearest_mode = ResizeParam::NearestMode::kRoundPreferFloor;
    else if (nearest == "round_prefer_ceil") param->nearest_mode = ResizeParam::NearestMode::kRoundPreferCeil;
    else if (nearest != "floor") return false;
    for (const auto* key : {"antialias", "exclude_outside"}) {
        auto it = node.attributes.find(key);
        if (it != node.attributes.end()) {
            const auto* value = std::get_if<int64_t>(&it->second);
            if (value == nullptr || *value != 0) return false;
        }
    }
    // Axes subsets, crop ROI, and aspect-preserving size policies are not implemented.
    if (node.attributes.count("axes") || node.attributes.count("roi")) return false;
    auto controls = node.attributes.find("onnx_nchw_controls");
    if (controls != node.attributes.end()) {
        const auto* value = std::get_if<int64_t>(&controls->second);
        if (value == nullptr || (*value != 0 && *value != 1)) return false;
        param->onnx_nchw_controls = *value != 0;
    }
    auto extrapolation = node.attributes.find("extrapolation_value");
    if (extrapolation != node.attributes.end()) {
        const auto* value = std::get_if<float>(&extrapolation->second);
        if (value == nullptr || *value != 0.0f) return false;
    }
    std::string policy = "stretch";
    return GetResizeString(node, "keep_aspect_ratio_policy", &policy) && policy == "stretch";
}

bool HasResizeValues(const std::shared_ptr<Tensor>& tensor) {
    return tensor != nullptr && tensor->numel() != 0;
}

template <typename T>
void RemapResizeControls(const ResizeParam& param, std::vector<T>* controls) {
    if (param.onnx_nchw_controls && IsChannelLastLayout(param.input->layout()) && controls->size() == 4) {
        *controls = {(*controls)[0], (*controls)[2], (*controls)[3], (*controls)[1]};
    }
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
    if (!ReadResizeAttributes(node, &param) ||
        (HasResizeValues(param.sizes) && (!param.scales.empty() || HasResizeValues(param.scales_tensor)))) return nullptr;
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
    if (param_.input == nullptr || param_.out == nullptr || param_.input->dims().size() == 0 ||
        HasResizeValues(param_.roi)) return -1;
    const auto& dims = param_.input->dims().data();
    for (const auto dim : dims) if (dim <= 0) return -1;
    std::vector<int64_t> shape;
    std::vector<float> scales = param_.scales;
    if (HasResizeValues(param_.sizes)) {
        if (HasResizeValues(param_.scales_tensor) || param_.sizes->dims().size() != 1 ||
            param_.sizes->data_type() != DataType::INT64 || !ReadIntegerTensor(param_.sizes, &shape) ||
            shape.size() != dims.size()) return -1;
        RemapResizeControls(param_, &shape);
        for (const auto size : shape) if (size <= 0) return -1;
    } else {
        if (HasResizeValues(param_.scales_tensor) &&
            (param_.scales_tensor->dims().size() != 1 || !ReadFloatTensor(param_.scales_tensor, &scales))) return -1;
        if (HasResizeValues(param_.scales_tensor)) RemapResizeControls(param_, &scales);
        if (scales.size() != dims.size()) return -1;
        for (size_t axis = 0; axis < scales.size(); ++axis) {
            const double size = std::floor(static_cast<double>(dims[axis]) * scales[axis]);
            if (!std::isfinite(scales[axis]) || scales[axis] <= 0.0f || size < 1.0 ||
                size >= static_cast<double>(std::numeric_limits<int64_t>::max())) return -1;
            shape.push_back(static_cast<int64_t>(size));
        }
    }
    if (param_.mode == ResizeParam::Mode::kLinear) {
        if (dims.size() != 4) return -1;
        const int h = HeightAxisForLayout(param_.input->layout());
        const int w = WidthAxisForLayout(param_.input->layout());
        for (int axis = 0; axis < 4; ++axis) {
            if (axis != h && axis != w && (shape[axis] != dims[axis] ||
                (!HasResizeValues(param_.sizes) && scales[axis] != 1.0f))) return -1;
        }
    }
    int64_t total = 1;
    for (const auto dim : shape) {
        if (total > std::numeric_limits<int64_t>::max() / dim) return -1;
        total *= dim;
    }
    const size_t bytes = DataTypeBytes(param_.input->data_type());
    return bytes != 0 && static_cast<uint64_t>(total) <= std::numeric_limits<size_t>::max() / bytes ? 0 : -1;
}

int32_t ResizeOp::InferOutputShapes() {
    if (CheckShape() != 0) {
        return -1;
    }

    std::vector<int64_t> out_shape;
    if (HasResizeValues(param_.sizes)) {
        if (!ReadIntegerTensor(param_.sizes, &out_shape)) {
            return -1;
        }
        RemapResizeControls(param_, &out_shape);
        param_.scales.resize(out_shape.size());
        param_.effective_scales.resize(out_shape.size());
        for (size_t i = 0; i < out_shape.size(); ++i) {
            param_.effective_scales[i] = static_cast<double>(out_shape[i]) / param_.input->dims()[i];
            param_.scales[i] = static_cast<float>(param_.effective_scales[i]);
        }
    } else {
        std::vector<float> scales = param_.scales;
        if (HasResizeValues(param_.scales_tensor) && !ReadFloatTensor(param_.scales_tensor, &scales)) {
            return -1;
        }
        if (HasResizeValues(param_.scales_tensor)) RemapResizeControls(param_, &scales);
        param_.scales = std::move(scales);
        param_.effective_scales.assign(param_.scales.begin(), param_.scales.end());
        out_shape.reserve(param_.scales.size());
        for (size_t i = 0; i < param_.scales.size(); ++i) {
            const auto dim = static_cast<int64_t>(
                std::floor(static_cast<double>(param_.input->dims()[i]) * param_.scales[i]));
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
    kernel_->SetParamOwner(std::make_shared<ResizeParam>(param_));
    return kernel_->compute();
}

}  // namespace operators
}  // namespace feather
