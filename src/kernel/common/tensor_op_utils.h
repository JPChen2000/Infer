#ifndef FEATHER_KERNEL_COMMON_TENSOR_OP_UTILS_H
#define FEATHER_KERNEL_COMMON_TENSOR_OP_UTILS_H

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "src/kernel/common/kernel_io.h"
#include "src/operator/params.h"
#include "util/bf16.h"

namespace feather {
namespace kernel {
namespace common_tensor_detail {

inline std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (int64_t i = static_cast<int64_t>(dims.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * dims[static_cast<size_t>(i + 1)];
    }
    return strides;
}

inline std::vector<int64_t> NormalizeAxes(const std::vector<int64_t>& axes, int64_t rank) {
    std::vector<int64_t> normalized;
    normalized.reserve(axes.size());
    for (auto axis : axes) {
        if (axis < 0) {
            axis += rank;
        }
        normalized.push_back(axis);
    }
    std::sort(normalized.begin(), normalized.end());
    return normalized;
}

inline void LinearToCoords(int64_t linear, const std::vector<int64_t>& dims, const std::vector<int64_t>& strides,
                           std::vector<int64_t>* coords) {
    coords->assign(dims.size(), 0);
    for (size_t axis = 0; axis < dims.size(); ++axis) {
        (*coords)[axis] = linear / strides[axis];
        linear %= strides[axis];
    }
}

inline int64_t ReadIndex(const Tensor* tensor, int64_t offset) {
    if (tensor->data_type() == DataType::INT64) {
        return tensor->data<int64_t>()[offset];
    }
    return tensor->data<int32_t>()[offset];
}

inline bool ReadBool(const Tensor* tensor, int64_t offset) {
    switch (tensor->data_type()) {
        case DataType::BOOL:
        case DataType::UINT8:
            return tensor->data<uint8_t>()[offset] != 0;
        case DataType::INT8:
            return tensor->data<int8_t>()[offset] != 0;
        case DataType::INT32:
            return tensor->data<int32_t>()[offset] != 0;
        case DataType::INT64:
            return tensor->data<int64_t>()[offset] != 0;
        case DataType::FP16:
            return HalfToFloat(tensor->data<uint16_t>()[offset]) != 0.0f;
        case DataType::BF16:
            return BFloat16ToFloat(tensor->data<BFloat16>()[offset].bits) != 0.0f;
        case DataType::FP32:
            return tensor->data<float>()[offset] != 0.0f;
        default:
            return false;
    }
}

inline float ReadFloat(const Tensor* tensor, int64_t offset) {
    switch (tensor->data_type()) {
        case DataType::BOOL:
        case DataType::UINT8:
            return static_cast<float>(tensor->data<uint8_t>()[offset]);
        case DataType::INT8:
            return static_cast<float>(tensor->data<int8_t>()[offset]);
        case DataType::FP16:
            return HalfToFloat(tensor->data<uint16_t>()[offset]);
        case DataType::BF16:
            return BFloat16ToFloat(tensor->data<BFloat16>()[offset].bits);
        case DataType::FP32:
            return tensor->data<float>()[offset];
        case DataType::INT32:
            return static_cast<float>(tensor->data<int32_t>()[offset]);
        case DataType::INT64:
            return static_cast<float>(tensor->data<int64_t>()[offset]);
        default:
            return static_cast<float>(tensor->data<uint8_t>()[offset]);
    }
}

inline int64_t ReadInteger(const Tensor* tensor, int64_t offset) {
    switch (tensor->data_type()) {
        case DataType::INT64:
            return tensor->data<int64_t>()[offset];
        case DataType::INT32:
            return tensor->data<int32_t>()[offset];
        case DataType::INT8:
            return tensor->data<int8_t>()[offset];
        case DataType::BOOL:
        case DataType::UINT8:
            return tensor->data<uint8_t>()[offset];
        default:
            return static_cast<int64_t>(ReadFloat(tensor, offset));
    }
}

inline bool SameValue(const Tensor* lhs, int64_t lhs_offset, const Tensor* rhs, int64_t rhs_offset) {
    if (lhs == nullptr || rhs == nullptr || lhs->data_type() != rhs->data_type()) {
        return false;
    }
    if (lhs->data_type() == DataType::FP16 || lhs->data_type() == DataType::BF16 ||
        lhs->data_type() == DataType::FP32) {
        return ReadFloat(lhs, lhs_offset) == ReadFloat(rhs, rhs_offset);
    }
    return ReadInteger(lhs, lhs_offset) == ReadInteger(rhs, rhs_offset);
}

inline int64_t BroadcastOffset(const std::vector<int64_t>& output_coordinates,
                               const std::vector<int64_t>& input_dims,
                               const std::vector<int64_t>& input_strides) {
    const size_t rank_gap = output_coordinates.size() - input_dims.size();
    int64_t offset = 0;
    for (size_t axis = 0; axis < input_dims.size(); ++axis) {
        const int64_t coordinate = input_dims[axis] == 1 ? 0 : output_coordinates[axis + rank_gap];
        offset += coordinate * input_strides[axis];
    }
    return offset;
}

inline int32_t CopyTensor(const feather::operators::AxesParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr ||
        !param->input->IsInitialized() || !param->out->IsInitialized() ||
        param->input->numel() != param->out->numel()) {
        return -1;
    }
    const size_t element_bytes = DataTypeBytes(param->input->data_type());
    const size_t bytes = static_cast<size_t>(param->input->numel()) * element_bytes;
    if (element_bytes == 0 || param->out->memory_size() < bytes) {
        return -1;
    }
    std::memcpy(param->out->raw_data(), param->input->raw_data(), bytes);
    param->out->set_data_type(param->input->data_type());
    return 0;
}

}  // namespace common_tensor_detail
}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_COMMON_TENSOR_OP_UTILS_H
