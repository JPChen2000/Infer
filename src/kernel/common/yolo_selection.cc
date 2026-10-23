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

#include "src/kernel/yolo_selection.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include "src/kernel/common/tensor_op_utils.h"
#include "src/operator/control_tensor.h"
#include "util/fp16.h"

namespace feather {
namespace kernel {
namespace {
using operators::YoloSelectionKind;
using operators::YoloSelectionParam;
namespace detail = common_tensor_detail;

void CopyElement(const Tensor& input, int64_t source, char* output, int64_t destination) {
    const size_t bytes = DataTypeBytes(input.data_type());
    std::memcpy(output + static_cast<size_t>(destination) * bytes,
                static_cast<const char*>(input.raw_data()) + static_cast<size_t>(source) * bytes, bytes);
}

bool GreaterValue(const Tensor& input, int64_t lhs, int64_t rhs) {
    if (operators::IsYoloSelectionInteger(input.data_type())) {
        return detail::ReadInteger(&input, lhs) > detail::ReadInteger(&input, rhs);
    }
    const float a = detail::ReadFloat(&input, lhs), b = detail::ReadFloat(&input, rhs);
    // Give NaNs a deterministic place while preserving their original payload.
    if (std::isnan(a)) return !std::isnan(b);
    if (std::isnan(b)) return false;
    return a > b;
}

int32_t TopK(YoloSelectionParam* param) {
    if (!param->control || !param->indices_out || !param->control->IsInitialized() ||
        param->control->data_type() != DataType::INT64 || param->control->numel() != 1) return -1;
    const auto& dims = param->data->dims().data();
    const int64_t rank = static_cast<int64_t>(dims.size());
    const int64_t axis = param->axis < 0 ? param->axis + rank : param->axis;
    if (axis < 0 || axis >= rank) return -1;
    const int64_t length = dims[axis], k = param->control->data<int64_t>()[0];
    auto expected = dims;
    if (k < 0 || k > length) return -1;
    expected[axis] = k;
    if (param->out->dims().data() != expected || param->indices_out->dims().data() != expected ||
        param->indices_out->data_type() != DataType::INT64) return -1;
    if (param->out->numel() == 0) return 0;
    auto* output = static_cast<char*>(param->out->mutable_data(param->out->memory_size()));
    auto* indices = param->indices_out->mutable_data<int64_t>();
    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < axis; ++i) outer *= dims[i];
    for (int64_t i = axis + 1; i < rank; ++i) inner *= dims[i];
    std::vector<int64_t> order(static_cast<size_t>(length));
    for (int64_t batch = 0; batch < outer; ++batch) {
        for (int64_t lane = 0; lane < inner; ++lane) {
            std::iota(order.begin(), order.end(), 0);
            const auto offset = [&](int64_t index) { return (batch * length + index) * inner + lane; };
            const auto before = [&](int64_t a, int64_t b) {
                const bool a_greater = GreaterValue(*param->data, offset(a), offset(b));
                const bool b_greater = GreaterValue(*param->data, offset(b), offset(a));
                if (!a_greater && !b_greater) return a < b;
                return param->largest ? a_greater : b_greater;
            };
            std::partial_sort(order.begin(), order.begin() + k, order.end(), before);
            // sorted=0 permits any order; the sorted result also satisfies it.
            for (int64_t selected = 0; selected < k; ++selected) {
                const int64_t destination = (batch * k + selected) * inner + lane;
                CopyElement(*param->data, offset(order[selected]), output, destination);
                indices[destination] = order[selected];
            }
        }
    }
    return 0;
}

int32_t GatherElements(YoloSelectionParam* param) {
    if (!param->control || param->control->data_type() != DataType::INT64 ||
        param->out->dims().data() != param->control->dims().data()) return -1;
    const auto& dims = param->data->dims().data();
    const auto& indices_dims = param->control->dims().data();
    const int64_t rank = static_cast<int64_t>(dims.size());
    const int64_t axis = param->axis < 0 ? param->axis + rank : param->axis;
    if (axis < 0 || axis >= rank || indices_dims.size() != dims.size()) return -1;
    for (int64_t i = 0; i < rank; ++i) {
        if (i != axis && indices_dims[i] > dims[i]) return -1;
    }
    if (param->out->numel() == 0) return 0;
    const auto source_strides = detail::ComputeStrides(dims);
    const auto index_strides = detail::ComputeStrides(indices_dims);
    const auto* indices = param->control->data<int64_t>();
    auto* output = static_cast<char*>(param->out->mutable_data(param->out->memory_size()));
    std::vector<int64_t> coords;
    for (int64_t i = 0; i < param->out->numel(); ++i) {
        detail::LinearToCoords(i, indices_dims, index_strides, &coords);
        int64_t index = indices[i];
        if (index < 0) index += dims[axis];
        if (index < 0 || index >= dims[axis]) return -1;
        coords[axis] = index;
        int64_t source = 0;
        for (int64_t j = 0; j < rank; ++j) source += coords[j] * source_strides[j];
        CopyElement(*param->data, source, output, i);
    }
    return 0;
}

void WriteLowest(Tensor* output, int64_t index) {
    switch (output->data_type()) {
        case DataType::INT64: output->mutable_data<int64_t>()[index] = std::numeric_limits<int64_t>::lowest(); break;
        case DataType::INT32: output->mutable_data<int32_t>()[index] = std::numeric_limits<int32_t>::lowest(); break;
        case DataType::INT8: output->mutable_data<int8_t>()[index] = std::numeric_limits<int8_t>::lowest(); break;
        case DataType::UINT8: output->mutable_data<uint8_t>()[index] = 0; break;
        case DataType::FP32: TensorIO<DataType::FP32>::Write(output, index, -std::numeric_limits<float>::infinity()); break;
        case DataType::FP16: TensorIO<DataType::FP16>::Write(output, index, -std::numeric_limits<float>::infinity()); break;
        case DataType::BF16: TensorIO<DataType::BF16>::Write(output, index, -std::numeric_limits<float>::infinity()); break;
        case DataType::FP8E4M3: TensorIO<DataType::FP8E4M3>::Write(output, index, -std::numeric_limits<float>::infinity()); break;
        case DataType::FP8E5M2: TensorIO<DataType::FP8E5M2>::Write(output, index, -std::numeric_limits<float>::infinity()); break;
        default: break;
    }
}

int32_t ReduceMax(YoloSelectionParam* param) {
    std::vector<int64_t> axes;
    if (!operators::YoloSelectionAxes(*param, &axes)) return -1;
    const auto& dims = param->data->dims().data();
    std::vector<int64_t> expected;
    for (size_t i = 0; i < dims.size(); ++i) {
        if (std::binary_search(axes.begin(), axes.end(), static_cast<int64_t>(i))) {
            if (param->keepdims) expected.push_back(1);
        } else expected.push_back(dims[i]);
    }
    if (param->out->dims().data() != expected) return -1;
    if (param->out->numel() == 0) return 0;
    const auto in_strides = detail::ComputeStrides(dims);
    const auto out_strides = detail::ComputeStrides(expected);
    std::vector<int64_t> selected(static_cast<size_t>(param->out->numel()), -1);
    std::vector<int64_t> coords;
    for (int64_t i = 0; i < param->data->numel(); ++i) {
        detail::LinearToCoords(i, dims, in_strides, &coords);
        int64_t target = 0;
        size_t out_axis = 0;
        for (size_t axis = 0; axis < dims.size(); ++axis) {
            if (std::binary_search(axes.begin(), axes.end(), static_cast<int64_t>(axis))) {
                if (param->keepdims) ++out_axis;
            } else target += coords[axis] * out_strides[out_axis++];
        }
        auto& choice = selected[static_cast<size_t>(target)];
        if (choice < 0 || GreaterValue(*param->data, i, choice)) choice = i;
    }
    auto* output = static_cast<char*>(param->out->mutable_data(param->out->memory_size()));
    for (int64_t i = 0; i < param->out->numel(); ++i) {
        if (selected[static_cast<size_t>(i)] < 0) WriteLowest(param->out.get(), i);
        else CopyElement(*param->data, selected[static_cast<size_t>(i)], output, i);
    }
    return 0;
}

void WriteInteger(Tensor* output, int64_t index, int64_t value) {
    switch (output->data_type()) {
        case DataType::INT64: output->mutable_data<int64_t>()[index] = value; break;
        case DataType::INT32: output->mutable_data<int32_t>()[index] = static_cast<int32_t>(value); break;
        case DataType::INT8: output->mutable_data<int8_t>()[index] = static_cast<int8_t>(value); break;
        case DataType::UINT8: output->mutable_data<uint8_t>()[index] = static_cast<uint8_t>(value); break;
        default: break;
    }
}

void WriteFloat(Tensor* output, int64_t index, float value) {
    switch (output->data_type()) {
        case DataType::FP32: TensorIO<DataType::FP32>::Write(output, index, value); break;
        case DataType::FP16: TensorIO<DataType::FP16>::Write(output, index, value); break;
        case DataType::BF16: TensorIO<DataType::BF16>::Write(output, index, value); break;
        case DataType::FP8E4M3: TensorIO<DataType::FP8E4M3>::Write(output, index, value); break;
        case DataType::FP8E5M2: TensorIO<DataType::FP8E5M2>::Write(output, index, value); break;
        default: break;
    }
}

int32_t IntegerBinary(Tensor* lhs, Tensor* rhs, Tensor* output, bool divide, bool fmod) {
    if (!lhs || !rhs || !output || lhs->data_type() != rhs->data_type() || lhs->data_type() != output->data_type()) return -1;
    std::vector<int64_t> shape;
    if (!operators::YoloSelectionBroadcast(lhs->dims().data(), rhs->dims().data(), &shape) || output->dims().data() != shape) return -1;
    const auto out_strides = detail::ComputeStrides(shape);
    const auto lhs_strides = detail::ComputeStrides(lhs->dims().data());
    const auto rhs_strides = detail::ComputeStrides(rhs->dims().data());
    std::vector<int64_t> coords;
    for (int64_t i = 0; i < output->numel(); ++i) {
        detail::LinearToCoords(i, shape, out_strides, &coords);
        const int64_t a_index = detail::BroadcastOffset(coords, lhs->dims().data(), lhs_strides);
        const int64_t b_index = detail::BroadcastOffset(coords, rhs->dims().data(), rhs_strides);
        if (operators::IsYoloSelectionInteger(lhs->data_type())) {
            const int64_t a = detail::ReadInteger(lhs, a_index), b = detail::ReadInteger(rhs, b_index);
            if (b == 0) return -1;
            const bool overflow = a == std::numeric_limits<int64_t>::min() && b == -1;
            if (divide && overflow) return -1;
            int64_t value = divide ? a / b : (overflow ? 0 : a % b);
            if (!divide && !fmod && value != 0 && ((value < 0) != (b < 0))) value += b;
            WriteInteger(output, i, value);
        } else {
            if (divide || !fmod) return -1;
            WriteFloat(output, i, std::fmod(detail::ReadFloat(lhs, a_index), detail::ReadFloat(rhs, b_index)));
        }
    }
    return 0;
}
}  // namespace

int32_t YoloSelectionKernel::compute() {
    auto* param = ParamAs<YoloSelectionParam>();
    if (!param || !param->data || !param->out || param->data->data_type() != param->out->data_type()) return -1;
    switch (param->kind) {
        case YoloSelectionKind::kTopK: return TopK(param);
        case YoloSelectionKind::kGatherElements: return GatherElements(param);
        case YoloSelectionKind::kReduceMax: return ReduceMax(param);
        case YoloSelectionKind::kMod: return IntegerBinary(param->data.get(), param->control.get(), param->out.get(), false, param->fmod);
    }
    return -1;
}

int32_t IntegerDivKernel::compute() {
    auto* param = ParamAs<operators::BinaryParam>();
    if (!param || !param->lhs || param->lhs->data_type() != DataType::INT64) return -1;
    return IntegerBinary(param->lhs.get(), param->rhs.get(), param->out.get(), true, false);
}

void EnsureYoloSelectionKernelsRegistered() {
    auto& dispatcher = KernelDispatcher::instance();
    static const bool registered = [&dispatcher] {
        for (const auto dtype : {DataType::FP32, DataType::FP16, DataType::BF16, DataType::FP8E4M3,
                                 DataType::FP8E5M2, DataType::INT64, DataType::INT32, DataType::INT8,
                                 DataType::UINT8, DataType::BOOL}) {
            for (const auto* op : {"TopK", "GatherElements", "ReduceMax", "Mod"}) {
                if (dtype == DataType::BOOL && std::string(op) != "GatherElements") continue;
                dispatcher.registerKernel(DeviceType::COMMON, dtype, op, [] { return std::make_unique<YoloSelectionKernel>(); });
            }
        }
        dispatcher.registerKernel(DeviceType::COMMON, DataType::INT64, "Div", [] { return std::make_unique<IntegerDivKernel>(); });
        return true;
    }();
    (void)registered;
}
}  // namespace kernel
}  // namespace feather
