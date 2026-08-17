#ifndef FEATHER_KERNEL_COMMON_ELEMENTWISE_BROADCAST_H
#define FEATHER_KERNEL_COMMON_ELEMENTWISE_BROADCAST_H

#include <algorithm>
#include <cstdint>
#include <vector>

#include "src/kernel/common/kernel_io.h"

namespace feather {
namespace kernel {
namespace common_detail {

inline std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (int64_t i = static_cast<int64_t>(dims.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * dims[static_cast<size_t>(i + 1)];
    }
    return strides;
}

inline bool InferBroadcastShape(const std::vector<int64_t>& lhs_dims, const std::vector<int64_t>& rhs_dims,
                                std::vector<int64_t>* out_dims) {
    if (out_dims == nullptr) {
        return false;
    }
    const size_t out_rank = std::max(lhs_dims.size(), rhs_dims.size());
    out_dims->assign(out_rank, 1);
    for (size_t axis = 0; axis < out_rank; ++axis) {
        const int64_t lhs_dim = axis < out_rank - lhs_dims.size() ? 1 : lhs_dims[axis - (out_rank - lhs_dims.size())];
        const int64_t rhs_dim = axis < out_rank - rhs_dims.size() ? 1 : rhs_dims[axis - (out_rank - rhs_dims.size())];
        if (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1) {
            return false;
        }
        (*out_dims)[axis] = std::max(lhs_dim, rhs_dim);
    }
    return true;
}

inline int64_t ComputeBroadcastOffset(const std::vector<int64_t>& out_coords,
                                      const std::vector<int64_t>& input_dims,
                                      const std::vector<int64_t>& input_strides) {
    const size_t rank_gap = out_coords.size() - input_dims.size();
    int64_t offset = 0;
    for (size_t axis = 0; axis < input_dims.size(); ++axis) {
        const int64_t coord = input_dims[axis] == 1 ? 0 : out_coords[axis + rank_gap];
        offset += coord * input_strides[axis];
    }
    return offset;
}

template <DataType dtype, typename Fn>
int32_t RunBinary(Tensor* output, const Tensor* lhs, const Tensor* rhs, Fn fn) {
    if (lhs == nullptr || rhs == nullptr || output == nullptr || lhs->data_type() != dtype ||
        rhs->data_type() != dtype || output->data_type() != dtype) {
        return -1;
    }
    std::vector<int64_t> output_dims;
    if (!InferBroadcastShape(lhs->dims().data(), rhs->dims().data(), &output_dims) ||
        output->dims().data() != output_dims) {
        return -1;
    }

    const auto output_strides = ComputeStrides(output_dims);
    const auto lhs_strides = ComputeStrides(lhs->dims().data());
    const auto rhs_strides = ComputeStrides(rhs->dims().data());
    std::vector<int64_t> output_coords(output_dims.size(), 0);
    for (int64_t linear = 0; linear < output->numel(); ++linear) {
        int64_t remaining = linear;
        for (size_t axis = 0; axis < output_dims.size(); ++axis) {
            output_coords[axis] = remaining / output_strides[axis];
            remaining %= output_strides[axis];
        }
        const int64_t lhs_offset = ComputeBroadcastOffset(output_coords, lhs->dims().data(), lhs_strides);
        const int64_t rhs_offset = ComputeBroadcastOffset(output_coords, rhs->dims().data(), rhs_strides);
        const float lhs_value = TensorIO<dtype>::Read(lhs, lhs_offset);
        const float rhs_value = TensorIO<dtype>::Read(rhs, rhs_offset);
        TensorIO<dtype>::Write(output, linear, fn(lhs_value, rhs_value));
    }
    return 0;
}

template <DataType dtype, typename Fn>
int32_t RunUnary(Tensor* output, const Tensor* input, Fn fn) {
    if (input == nullptr || output == nullptr || input->data_type() != dtype || output->data_type() != dtype ||
        input->dims().data() != output->dims().data()) {
        return -1;
    }
    for (int64_t i = 0; i < input->numel(); ++i) {
        TensorIO<dtype>::Write(output, i, fn(TensorIO<dtype>::Read(input, i)));
    }
    return 0;
}

}  // namespace common_detail
}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_COMMON_ELEMENTWISE_BROADCAST_H
