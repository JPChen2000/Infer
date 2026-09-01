#ifndef FEATHER_OPERATOR_ELEMENTWISE_UTILS_H
#define FEATHER_OPERATOR_ELEMENTWISE_UTILS_H

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>

#include "core/operator_registry.h"
#include "src/operator/params.h"
#include "util/types.h"

namespace feather {
namespace operators {
namespace elementwise_detail {

inline bool InferBroadcastShape(const std::vector<int64_t>& lhs_dims, const std::vector<int64_t>& rhs_dims,
                                std::vector<int64_t>* out_dims) {
    if (out_dims == nullptr) {
        return false;
    }
    const size_t out_rank = std::max(lhs_dims.size(), rhs_dims.size());
    out_dims->assign(out_rank, 1);
    for (size_t i = 0; i < out_rank; ++i) {
        const int64_t lhs_dim = i < out_rank - lhs_dims.size() ? 1 : lhs_dims[i - (out_rank - lhs_dims.size())];
        const int64_t rhs_dim = i < out_rank - rhs_dims.size() ? 1 : rhs_dims[i - (out_rank - rhs_dims.size())];
        if (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1) {
            return false;
        }
        (*out_dims)[i] = std::max(lhs_dim, rhs_dim);
    }
    return true;
}

inline int32_t CheckBinaryParam(const BinaryParam& param) {
    if (param.lhs == nullptr || param.rhs == nullptr || param.out == nullptr) {
        return -1;
    }
    std::vector<int64_t> out_shape;
    return InferBroadcastShape(param.lhs->dims().data(), param.rhs->dims().data(), &out_shape) ? 0 : -1;
}

inline int32_t InferBinaryOutput(BinaryParam* param) {
    if (param == nullptr || CheckBinaryParam(*param) != 0) {
        return -1;
    }
    std::vector<int64_t> out_shape;
    if (!InferBroadcastShape(param->lhs->dims().data(), param->rhs->dims().data(), &out_shape)) {
        return -1;
    }
    const auto dtype = ResolveExecutionDataType({param->lhs, param->rhs, param->out}, DataType::FP32);
    const auto output_quantization = param->out->quantization().enabled
                                         ? param->out->quantization()
                                         : (param->lhs->quantization().enabled ? param->lhs->quantization()
                                                                                : param->rhs->quantization());
    const auto output_layout = param->out->layout();
    const size_t required_bytes =
        static_cast<size_t>(std::max<int64_t>(1, std::accumulate(out_shape.begin(), out_shape.end(), int64_t{1},
                                                                 std::multiplies<int64_t>()))) *
        DataTypeBytes(dtype);
    if (!param->out->IsInitialized() || param->out->memory_size() < required_bytes) {
        param->out = std::make_shared<Tensor>(out_shape);
        param->out->set_quantization(output_quantization);
    } else {
        param->out->Resize(out_shape);
    }
    param->out->set_data_type(dtype);
    param->out->set_quantization(output_quantization);
    param->out->set_layout(output_layout != DataLayout::ND
                               ? output_layout
                               : (param->lhs->layout() != DataLayout::ND ? param->lhs->layout() : param->rhs->layout()));
    return 0;
}

inline int32_t CheckUnaryParam(const UnaryParam& param) {
    return param.input != nullptr && param.out != nullptr ? 0 : -1;
}

inline int32_t InferUnaryOutput(UnaryParam* param) {
    if (param == nullptr || CheckUnaryParam(*param) != 0) {
        return -1;
    }
    const auto dtype = ResolveExecutionDataType({param->input, param->out}, DataType::FP32);
    const auto output_quantization = param->out->quantization().enabled
                                         ? param->out->quantization()
                                         : param->input->quantization();
    const auto output_layout = param->out->layout();
    const size_t required_bytes = static_cast<size_t>(std::max<int64_t>(1, param->input->numel())) * DataTypeBytes(dtype);
    if (!param->out->IsInitialized() || param->out->memory_size() < required_bytes) {
        param->out = std::make_shared<Tensor>(param->input->dims().data());
        param->out->set_quantization(output_quantization);
    } else {
        param->out->Resize(param->input->dims().data());
    }
    param->out->set_data_type(dtype);
    param->out->set_quantization(output_quantization);
    param->out->set_layout(output_layout != DataLayout::ND ? output_layout : param->input->layout());
    return 0;
}

}  // namespace elementwise_detail
}  // namespace operators
}  // namespace feather

#endif  // FEATHER_OPERATOR_ELEMENTWISE_UTILS_H
