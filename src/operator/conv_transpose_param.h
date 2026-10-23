// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#ifndef FEATHER_OPERATOR_CONV_TRANSPOSE_PARAM_H
#define FEATHER_OPERATOR_CONV_TRANSPOSE_PARAM_H

#include <algorithm>
#include <array>
#include <limits>

#include "core/tensor.h"

namespace feather {
namespace operators {

struct ConvTransposeParam {
    std::shared_ptr<Tensor> input, weight, bias, out;
    std::array<int64_t, 2> strides{{1, 1}};
    std::array<int64_t, 2> dilations{{1, 1}};
    std::array<int64_t, 2> output_padding{{0, 0}};
    std::array<int64_t, 4> pads{{0, 0, 0, 0}};
    int64_t group{1};

    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(input); visit(weight); visit(bias); visit(out);
    }
};

// ONNX uses NCHW data and [Cin, Cout/group, kH, kW] weights.
inline bool ConvTransposeOutputShape(const ConvTransposeParam& param, std::vector<int64_t>* shape) {
    if (!shape || !param.input || !param.weight || !param.out || param.input == param.out ||
        param.weight == param.out || (param.bias && param.bias == param.out) ||
        param.input->dims().size() != 4 || param.weight->dims().size() != 4 || param.group <= 0 ||
        param.input->data_type() != DataType::FP32 || param.weight->data_type() != DataType::FP32 ||
        (param.out->data_type() != DataType::UNKNOWN && param.out->data_type() != DataType::FP32) ||
        param.input->layout() == DataLayout::NHWC || param.out->layout() == DataLayout::NHWC) return false;
    const auto& input = param.input->dims();
    const auto& weight = param.weight->dims();
    for (int axis = 0; axis < 4; ++axis)
        if (input[axis] <= 0 || weight[axis] <= 0 || param.pads[axis] < 0) return false;
    if (input[1] != weight[0] || input[1] % param.group != 0 ||
        weight[1] > std::numeric_limits<int64_t>::max() / param.group) return false;
    const int64_t channels = weight[1] * param.group;
    if (param.bias && (param.bias->data_type() != DataType::FP32 || param.bias->dims().size() != 1 ||
                       param.bias->dims()[0] != channels)) return false;
    std::vector<int64_t> result{input[0], channels, 0, 0};
    for (int axis = 0; axis < 2; ++axis) {
        if (param.strides[axis] <= 0 || param.dilations[axis] <= 0 || param.output_padding[axis] < 0 ||
            param.output_padding[axis] >= std::max(param.strides[axis], param.dilations[axis])) return false;
        const __int128 size = static_cast<__int128>(param.strides[axis]) * (input[axis + 2] - 1) +
                              static_cast<__int128>(param.dilations[axis]) * (weight[axis + 2] - 1) + 1 +
                              param.output_padding[axis] - param.pads[axis] - param.pads[axis + 2];
        if (size <= 0 || size > std::numeric_limits<int64_t>::max()) return false;
        result[axis + 2] = static_cast<int64_t>(size);
    }
    size_t bytes = 0;
    if (!CheckedTensorBytes(result, DataType::FP32, &bytes) ||
        !CheckedTensorBytes(input.data(), DataType::FP32, &bytes) ||
        !CheckedTensorBytes(weight.data(), DataType::FP32, &bytes)) return false;
    *shape = std::move(result);
    return true;
}

}  // namespace operators
}  // namespace feather
#endif
