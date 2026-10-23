// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#ifndef FEATHER_OPERATOR_NORMALIZATION_PARAM_H
#define FEATHER_OPERATOR_NORMALIZATION_PARAM_H

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "core/tensor.h"

namespace feather {
namespace operators {

struct NormalizationParam {
    std::shared_ptr<Tensor> input, weight, bias, out, mean, inv_std_dev;
    int64_t axis{-1};
    float epsilon{1e-5f};
    int64_t stash_type{1};  // ONNX FLOAT: reductions and saved statistics use FP32.
    float weight_offset{0.0f};  // RMS extension for weights stored as scale - 1.
    bool layer_norm{false};

    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(input); visit(weight); visit(bias); visit(out); visit(mean); visit(inv_std_dev);
    }
};

namespace normalization_detail {

inline bool IsFloatType(DataType type) {
    return type == DataType::FP32 || type == DataType::FP16 || type == DataType::BF16;
}

struct Shape {
    int64_t axis{0}, rows{1}, width{1};
    std::vector<int64_t> input_strides, weight_strides, bias_strides, stats_dims;
};

// An aligned broadcast stride of zero repeats that affine value along the axis.
inline bool AffineStrides(const Tensor& affine, const std::vector<int64_t>& dims,
                          std::vector<int64_t>* strides) {
    if (!IsFloatType(affine.data_type()) || affine.dims().size() > dims.size()) return false;
    strides->assign(dims.size(), 0);
    int64_t stride = 1;
    const size_t gap = dims.size() - affine.dims().size();
    for (size_t i = affine.dims().size(); i-- > 0;) {
        const int64_t dim = affine.dims()[i];
        if (dim <= 0 || (dim != 1 && dim != dims[gap + i]) ||
            stride > std::numeric_limits<int64_t>::max() / dim) return false;
        (*strides)[gap + i] = dim == 1 ? 0 : stride;
        stride *= dim;
    }
    size_t bytes = 0;
    return CheckedTensorBytes(affine.dims().data(), affine.data_type(), &bytes);
}

inline bool OverlappingRanges(const Tensor& a, size_t a_bytes, const Tensor& b, size_t b_bytes) {
    if (&a == &b) return true;
    if (a_bytes == 0 || b_bytes == 0) return false;
    if (a.storage() && a.storage() == b.storage()) {
        const size_t first = a.offset(), second = b.offset();
        return first <= second ? second - first < a_bytes : first - second < b_bytes;
    }
    // Different Buffer/Storage wrappers may still refer to overlapping memory.
    // Do not materialize deferred allocations merely to compare addresses.
    if (!a.storage() || !b.storage() || !a.storage()->materialized() || !b.storage()->materialized() ||
        !a.IsInitialized() || !b.IsInitialized()) return false;
    const auto first = reinterpret_cast<std::uintptr_t>(a.raw_data());
    const auto second = reinterpret_cast<std::uintptr_t>(b.raw_data());
    return first <= second ? second - first < a_bytes : first - second < b_bytes;
}

inline bool Describe(const NormalizationParam& param, Shape* shape) {
    if (!shape || !param.input || !param.weight || !param.out || param.input->dims().empty() ||
        !IsFloatType(param.input->data_type()) ||
        (param.out->data_type() != DataType::UNKNOWN && param.out->data_type() != param.input->data_type()) ||
        !std::isfinite(param.epsilon) || param.epsilon < 0.0f || param.stash_type != 1 ||
        !std::isfinite(param.weight_offset) || (param.layer_norm && param.weight_offset != 0.0f) ||
        (!param.layer_norm && (param.mean || param.inv_std_dev))) return false;
    const auto& dims = param.input->dims().data();
    const int64_t rank = static_cast<int64_t>(dims.size());
    if (param.axis < -rank || param.axis >= rank) return false;
    Shape result;
    result.axis = param.axis < 0 ? param.axis + rank : param.axis;
    result.input_strides.resize(dims.size());
    result.stats_dims = dims;
    int64_t count = 1;
    for (size_t i = dims.size(); i-- > 0;) {
        if (dims[i] <= 0 || count > std::numeric_limits<int64_t>::max() / dims[i]) return false;
        result.input_strides[i] = count;
        count *= dims[i];
        if (i >= static_cast<size_t>(result.axis)) result.stats_dims[i] = 1;
    }
    result.width = dims[static_cast<size_t>(result.axis)] * result.input_strides[static_cast<size_t>(result.axis)];
    result.rows = count / result.width;
    if (!AffineStrides(*param.weight, dims, &result.weight_strides) ||
        (param.bias && !AffineStrides(*param.bias, dims, &result.bias_strides))) return false;
    if (!param.bias) result.bias_strides.assign(dims.size(), 0);
    size_t output_bytes = 0, stats_bytes = 0;
    if (!CheckedTensorBytes(dims, param.input->data_type(), &output_bytes) ||
        !CheckedTensorBytes(result.stats_dims, DataType::FP32, &stats_bytes)) return false;
    const std::shared_ptr<Tensor> inputs[] = {param.input, param.weight, param.bias};
    const std::shared_ptr<Tensor> outputs[] = {param.out, param.mean, param.inv_std_dev};
    const size_t output_sizes[] = {output_bytes, stats_bytes, stats_bytes};
    for (size_t i = 0; i < 3; ++i) {
        if (!outputs[i]) continue;
        for (const auto& input : inputs) {
            if (!input) continue;
            size_t input_bytes = 0;
            if (!CheckedTensorBytes(input->dims().data(), input->data_type(), &input_bytes) ||
                OverlappingRanges(*input, input_bytes, *outputs[i], output_sizes[i])) return false;
        }
        for (size_t j = 0; j < i; ++j)
            if (outputs[j] && OverlappingRanges(*outputs[j], output_sizes[j], *outputs[i], output_sizes[i]))
                return false;
        if (i != 0 && outputs[i]->data_type() != DataType::UNKNOWN &&
            outputs[i]->data_type() != DataType::FP32) return false;
    }
    *shape = std::move(result);
    return true;
}

inline bool HasStorage(const std::shared_ptr<Tensor>& tensor) {
    size_t bytes = 0;
    return tensor && tensor->IsInitialized() &&
           CheckedTensorBytes(tensor->dims().data(), tensor->data_type(), &bytes) && tensor->memory_size() >= bytes;
}

inline bool ValidateStorage(const NormalizationParam& param, Shape* shape) {
    return Describe(param, shape) && HasStorage(param.input) && HasStorage(param.weight) && HasStorage(param.out) &&
           (!param.bias || HasStorage(param.bias)) && param.out->dims() == param.input->dims() &&
           param.out->data_type() == param.input->data_type() &&
           (!param.mean || (HasStorage(param.mean) && param.mean->data_type() == DataType::FP32 &&
                            param.mean->dims().data() == shape->stats_dims)) &&
           (!param.inv_std_dev || (HasStorage(param.inv_std_dev) && param.inv_std_dev->data_type() == DataType::FP32 &&
                                   param.inv_std_dev->dims().data() == shape->stats_dims));
}

inline int64_t AffineOffset(int64_t index, const Shape& shape, const std::vector<int64_t>& strides) {
    int64_t offset = 0;
    for (size_t axis = 0; axis < strides.size(); ++axis) {
        const int64_t coordinate = index / shape.input_strides[axis];
        index %= shape.input_strides[axis];
        offset += coordinate * strides[axis];
    }
    return offset;
}

}  // namespace normalization_detail
}  // namespace operators
}  // namespace feather
#endif
