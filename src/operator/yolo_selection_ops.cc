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

#include "src/operator/yolo_selection_ops.h"

#include <algorithm>
#include <numeric>
#include <set>
#include <utility>

#include "core/operator_registry.h"
#include "core/shape_utils.h"
#include "src/kernel/yolo_selection.h"
#include "src/operator/control_tensor.h"
#include "src/operator/tensor_op_utils.h"

namespace feather {
namespace operators {
namespace {

int64_t NormalizeAxis(int64_t axis, size_t rank) {
    return axis < 0 ? axis + static_cast<int64_t>(rank) : axis;
}

std::shared_ptr<OpBase> BuildYoloSelection(const model::NodeDesc& node,
                                          OperatorRegistry::TensorMap& tensors,
                                          const OperatorRegistry::BuildContext& context) {
    YoloSelectionParam param;
    if (node.op_type == "TopK") {
        if (node.inputs.size() != 2 || node.outputs.size() != 2) return nullptr;
        param.kind = YoloSelectionKind::kTopK;
        param.axis = tensor_op_detail::GetIntAttribute(node.attributes, "axis", -1);
        const int64_t largest = tensor_op_detail::GetIntAttribute(node.attributes, "largest", 1);
        const int64_t sorted = tensor_op_detail::GetIntAttribute(node.attributes, "sorted", 1);
        if ((largest != 0 && largest != 1) || (sorted != 0 && sorted != 1)) return nullptr;
        param.largest = largest != 0;
        param.sorted = sorted != 0;
        param.indices_out = tensors[node.outputs[1]];
    } else if (node.op_type == "GatherElements") {
        if (node.inputs.size() != 2 || node.outputs.size() != 1) return nullptr;
        param.kind = YoloSelectionKind::kGatherElements;
        param.axis = tensor_op_detail::GetIntAttribute(node.attributes, "axis", 0);
    } else if (node.op_type == "ReduceMax") {
        if (node.inputs.empty() || node.inputs.size() > 2 || node.outputs.size() != 1) return nullptr;
        param.kind = YoloSelectionKind::kReduceMax;
        param.axes = tensor_op_detail::GetIntVectorAttribute(node.attributes, "axes");
        param.keepdims = tensor_op_detail::GetIntAttribute(node.attributes, "keepdims", 1) != 0;
        param.noop_with_empty_axes = tensor_op_detail::GetIntAttribute(node.attributes, "noop_with_empty_axes", 0) != 0;
    } else if (node.op_type == "Mod") {
        if (node.inputs.size() != 2 || node.outputs.size() != 1) return nullptr;
        param.kind = YoloSelectionKind::kMod;
        const int64_t fmod = tensor_op_detail::GetIntAttribute(node.attributes, "fmod", 0);
        if (fmod != 0 && fmod != 1) return nullptr;
        param.fmod = fmod != 0;
    } else {
        return nullptr;
    }
    param.data = tensors[node.inputs[0]];
    if (node.inputs.size() > 1 && !node.inputs[1].empty()) param.control = tensors[node.inputs[1]];
    param.out = tensors[node.outputs[0]];
    auto op = std::make_shared<YoloSelectionOp>(node.name.empty() ? node.op_type : node.name, node.op_type, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) return nullptr;
    kernel::EnsureYoloSelectionKernelsRegistered();
    auto kernel = CreateKernelForTensor(context.device, node.op_type,
                                       {param.data, param.control, param.out, param.indices_out});
    if (!kernel) return nullptr;
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_registered = [] {
    for (const auto* type : {"TopK", "GatherElements", "ReduceMax", "Mod"}) {
        OperatorRegistry::instance().Register(type, BuildYoloSelection);
    }
    return true;
}();
}  // namespace

bool IsYoloSelectionInteger(DataType dtype) {
    return dtype == DataType::INT64 || dtype == DataType::INT32 || dtype == DataType::INT8 || dtype == DataType::UINT8;
}

bool IsYoloSelectionNumeric(DataType dtype) {
    return IsYoloSelectionInteger(dtype) || tensor_op_detail::IsFloatingPointDataType(dtype);
}

bool YoloSelectionBroadcast(const std::vector<int64_t>& lhs, const std::vector<int64_t>& rhs,
                            std::vector<int64_t>* output) {
    const size_t rank = std::max(lhs.size(), rhs.size());
    std::vector<int64_t> result(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
        const int64_t a = i < rank - lhs.size() ? 1 : lhs[i - (rank - lhs.size())];
        const int64_t b = i < rank - rhs.size() ? 1 : rhs[i - (rank - rhs.size())];
        if (a < 0 || b < 0 || (a != b && a != 1 && b != 1)) return false;
        result[i] = a == 1 ? b : a;
    }
    *output = std::move(result);
    return true;
}

bool YoloSelectionAxes(const YoloSelectionParam& param, std::vector<int64_t>* axes) {
    *axes = param.axes;
    if (param.control) {
        if (param.control->data_type() != DataType::INT64) return false;
        if (param.control->numel() == 0) axes->clear();
        else if (!ReadIntegerTensor(param.control, axes)) return false;
    }
    const int64_t rank = static_cast<int64_t>(param.data->dims().size());
    if (axes->empty() && !param.noop_with_empty_axes) {
        axes->resize(rank);
        std::iota(axes->begin(), axes->end(), 0);
    }
    std::set<int64_t> seen;
    for (auto& axis : *axes) {
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank || !seen.insert(axis).second) return false;
    }
    std::sort(axes->begin(), axes->end());
    return true;
}

YoloSelectionOp::YoloSelectionOp(std::string name, std::string type, YoloSelectionParam param)
    : OpBase(std::move(name), std::move(type)), param_(std::move(param)) { SyncIO(); }

void YoloSelectionOp::SyncIO() {
    std::vector<std::shared_ptr<Tensor>> inputs{param_.data};
    if (param_.control) inputs.push_back(param_.control);
    SetInputs(inputs);
    if (param_.indices_out) SetOutputs({param_.out, param_.indices_out});
    else SetOutputs({param_.out});
}

int32_t YoloSelectionOp::CheckShape() const {
    if (!param_.data || !param_.out) return -1;
    const auto dtype = param_.data->data_type();
    if (!IsYoloSelectionNumeric(dtype) && !(param_.kind == YoloSelectionKind::kGatherElements && dtype == DataType::BOOL)) return -1;
    const auto& dims = param_.data->dims().data();
    size_t bytes = 0;
    if (!CheckedTensorBytes(dims, dtype, &bytes)) return -1;
    if (param_.kind == YoloSelectionKind::kTopK) {
        const int64_t axis = NormalizeAxis(param_.axis, dims.size());
        if (!param_.control || !param_.indices_out || param_.control->data_type() != DataType::INT64 ||
            param_.control->numel() != 1 || param_.control->dims().size() != 1 || axis < 0 || axis >= static_cast<int64_t>(dims.size())) return -1;
    } else if (param_.kind == YoloSelectionKind::kGatherElements) {
        const int64_t axis = NormalizeAxis(param_.axis, dims.size());
        if (!param_.control || param_.control->data_type() != DataType::INT64 ||
            param_.control->dims().size() != dims.size() || dims.empty() || axis < 0 || axis >= static_cast<int64_t>(dims.size())) return -1;
        const auto& indices_dims = param_.control->dims().data();
        for (size_t i = 0; i < dims.size(); ++i) {
            if (indices_dims[i] < 0 || (static_cast<int64_t>(i) != axis && indices_dims[i] > dims[i])) return -1;
        }
    } else if (param_.kind == YoloSelectionKind::kReduceMax) {
        if (param_.control && (param_.control->data_type() != DataType::INT64 || param_.control->dims().size() != 1)) return -1;
    } else if (param_.kind == YoloSelectionKind::kMod) {
        std::vector<int64_t> out;
        if (!param_.control || param_.control->data_type() != dtype ||
            (!IsYoloSelectionInteger(dtype) && !param_.fmod) ||
            !YoloSelectionBroadcast(dims, param_.control->dims().data(), &out)) return -1;
    }
    return 0;
}

int32_t YoloSelectionOp::InferOutputShapes() {
    if (CheckShape() != 0) return -1;
    auto shape = param_.data->dims().data();
    if (param_.kind == YoloSelectionKind::kTopK) {
        const int64_t axis = NormalizeAxis(param_.axis, shape.size());
        if (param_.control->IsInitialized()) {
            const int64_t k = param_.control->data<int64_t>()[0];
            if (k < 0 || k > shape[axis]) return -1;
            shape[axis] = k;
        } else {
            // Static ONNX output metadata also supports K produced by a graph
            // node. The kernel validates the live K against this fixed shape.
            const auto& declared = param_.out->dims().data();
            if (declared.size() != shape.size() || declared[axis] < 0 || declared[axis] > shape[axis]) return -1;
            shape[axis] = declared[axis];
        }
    } else if (param_.kind == YoloSelectionKind::kGatherElements) {
        shape = param_.control->dims().data();
    } else if (param_.kind == YoloSelectionKind::kReduceMax) {
        if (param_.control && param_.control->numel() != 0 && !param_.control->IsInitialized()) {
            shape = param_.out->dims().data();
        } else {
            std::vector<int64_t> axes;
            if (!YoloSelectionAxes(param_, &axes)) return -1;
            std::vector<int64_t> reduced;
            for (size_t i = 0; i < shape.size(); ++i) {
                if (std::binary_search(axes.begin(), axes.end(), static_cast<int64_t>(i))) {
                    if (param_.keepdims) reduced.push_back(1);
                } else reduced.push_back(shape[i]);
            }
            shape = std::move(reduced);
        }
    } else if (!YoloSelectionBroadcast(shape, param_.control->dims().data(), &shape)) return -1;
    size_t bytes = 0;
    if (!CheckedTensorBytes(shape, param_.data->data_type(), &bytes)) return -1;
    param_.out = tensor_op_detail::AllocateOutput(param_.out, shape, param_.data->data_type());
    if (!param_.out) return -1;
    param_.out->set_quantization(param_.data->quantization());
    param_.out->set_layout(param_.data->layout());
    if (param_.indices_out) {
        param_.indices_out = tensor_op_detail::AllocateOutput(param_.indices_out, shape, DataType::INT64);
        if (!param_.indices_out) return -1;
    }
    SyncIO();
    return 0;
}

void YoloSelectionOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_) kernel_->SetParamOwner(std::make_shared<YoloSelectionParam>(param_));
}

int32_t YoloSelectionOp::Run() {
    if (InferOutputShapes() != 0 || !kernel_) return -1;
    return kernel_->compute();
}

void EnsureYoloSelectionOperatorsRegistered() { (void)g_registered; kernel::EnsureYoloSelectionKernelsRegistered(); }
}  // namespace operators
}  // namespace feather
