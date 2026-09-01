#ifndef FEATHER_OPERATOR_TENSOR_OP_UTILS_H
#define FEATHER_OPERATOR_TENSOR_OP_UTILS_H

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "core/tensor.h"
#include "model/model_format.h"
#include "src/operator/control_tensor.h"
#include "src/operator/params.h"

namespace feather {
namespace operators {
namespace tensor_op_detail {

inline std::vector<int64_t> GetIntVectorAttribute(
    const std::unordered_map<std::string, model::AttributeValue>& attributes, const std::string& key) {
    const auto it = attributes.find(key);
    if (it == attributes.end()) {
        return {};
    }
    if (const auto* value = std::get_if<std::vector<int64_t>>(&it->second); value != nullptr) {
        return *value;
    }
    return {};
}

inline int64_t GetIntAttribute(const std::unordered_map<std::string, model::AttributeValue>& attributes,
                               const std::string& key, int64_t default_value) {
    const auto it = attributes.find(key);
    if (it == attributes.end()) {
        return default_value;
    }
    if (const auto* value = std::get_if<int64_t>(&it->second); value != nullptr) {
        return *value;
    }
    return default_value;
}

inline bool IsFloatingPointDataType(DataType data_type) {
    return data_type == DataType::FP16 || data_type == DataType::BF16 || data_type == DataType::FP32 ||
           data_type == DataType::FP8E4M3 || data_type == DataType::FP8E5M2;
}

inline std::vector<int64_t> NormalizeAxes(const std::vector<int64_t>& axes, int64_t rank, bool for_unsqueeze) {
    std::vector<int64_t> normalized;
    const int64_t target_rank = for_unsqueeze ? rank + static_cast<int64_t>(axes.size()) : rank;
    normalized.reserve(axes.size());
    for (auto axis : axes) {
        if (axis < 0) {
            axis += target_rank;
        }
        normalized.push_back(axis);
    }
    std::sort(normalized.begin(), normalized.end());
    return normalized;
}

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
        if (lhs_dim < 0 || rhs_dim < 0) {
            return false;
        }
        if (lhs_dim == 0 || rhs_dim == 0) {
            // Zero is the graph's unknown-dimension placeholder. Preserve it
            // until runtime rather than treating it as a zero-sized tensor.
            (*out_dims)[i] = 0;
            continue;
        }
        if (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1) {
            return false;
        }
        (*out_dims)[i] = std::max(lhs_dim, rhs_dim);
    }
    return true;
}

inline size_t NumelForShape(const std::vector<int64_t>& shape) {
    size_t numel = 1;
    for (const auto dim : shape) {
        if (dim < 0) {
            return 0;
        }
        numel *= static_cast<size_t>(dim);
    }
    return numel;
}

inline std::shared_ptr<Tensor> AllocateOutput(const std::shared_ptr<Tensor>& output,
                                               const std::vector<int64_t>& shape, DataType data_type) {
    const size_t numel = NumelForShape(shape);
    const size_t element_bytes = DataTypeBytes(data_type);
    if (element_bytes == 0) {
        return nullptr;
    }
    const size_t bytes = numel * element_bytes;
    auto result = output;
    const QuantizationParams quantization = result == nullptr ? QuantizationParams{} : result->quantization();
    const DataLayout layout = result == nullptr ? DataLayout::ND : result->layout();
    if (numel == 0) {
        if (result == nullptr) {
            result = std::make_shared<Tensor>();
        }
        result->Resize(shape);
        result->set_data_type(data_type);
        result->set_quantization(quantization);
        result->set_layout(layout);
        return result;
    }
    if (result == nullptr) {
        result = std::make_shared<Tensor>();
    }
    if (!result->IsInitialized() || result->memory_size() < bytes) {
        result->ResetBuffer(std::make_shared<Buffer>(bytes), bytes);
    }
    result->Resize(shape);
    result->set_data_type(data_type);
    result->set_quantization(quantization);
    result->set_layout(layout);
    return result;
}

inline bool ReadShapeValues(const std::shared_ptr<Tensor>& shape_tensor, std::vector<int64_t>* shape) {
    return ReadIntegerTensor(shape_tensor, shape);
}

inline int32_t InferSameTypeOutput(const std::shared_ptr<Tensor>& input, std::shared_ptr<Tensor>* output,
                                   const std::vector<int64_t>& shape) {
    if (input == nullptr || output == nullptr) return -1;
    const DataType output_data_type = input->data_type() == DataType::UNKNOWN ? DataType::FP32 : input->data_type();
    const QuantizationParams output_quantization =
        *output != nullptr && (*output)->quantization().enabled ? (*output)->quantization() : input->quantization();
    const DataLayout output_layout =
        *output != nullptr && (*output)->layout() != DataLayout::ND ? (*output)->layout() : input->layout();
    auto result = AllocateOutput(*output, shape, output_data_type);
    if (result == nullptr) return -1;
    result->set_quantization(output_quantization);
    result->set_layout(output_layout);
    *output = std::move(result);
    return 0;
}

inline int32_t InferAxesOutputShape(AxesParam* param, bool unsqueeze) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }
    std::vector<int64_t> axes = param->axes;
    if (param->axes_tensor != nullptr && !ReadIntegerTensor(param->axes_tensor, &axes)) {
        return -1;
    }
    if (axes.empty()) {
        return -1;
    }

    std::vector<int64_t> out_shape = param->input->dims().data();
    if (unsqueeze) {
        const auto normalized_axes = NormalizeAxes(axes, static_cast<int64_t>(out_shape.size()), true);
        if (std::set<int64_t>(normalized_axes.begin(), normalized_axes.end()).size() != normalized_axes.size()) {
            return -1;
        }
        for (const auto axis : normalized_axes) {
            if (axis < 0 || axis > static_cast<int64_t>(out_shape.size())) {
                return -1;
            }
            out_shape.insert(out_shape.begin() + axis, 1);
        }
    } else {
        const auto normalized_axes = NormalizeAxes(axes, static_cast<int64_t>(out_shape.size()), false);
        const std::set<int64_t> axis_set(normalized_axes.begin(), normalized_axes.end());
        if (axis_set.size() != normalized_axes.size()) {
            return -1;
        }
        for (const auto axis : normalized_axes) {
            if (axis < 0 || axis >= static_cast<int64_t>(out_shape.size())) {
                return -1;
            }
        }
        std::vector<int64_t> squeezed;
        for (int64_t axis = 0; axis < static_cast<int64_t>(out_shape.size()); ++axis) {
            if (axis_set.count(axis) != 0) {
                if (out_shape[axis] != 1) {
                    return -1;
                }
                continue;
            }
            squeezed.push_back(out_shape[axis]);
        }
        out_shape = squeezed.empty() ? std::vector<int64_t>{1} : std::move(squeezed);
    }

    const QuantizationParams output_quantization =
        param->out->quantization().enabled ? param->out->quantization() : param->input->quantization();
    const DataLayout output_layout =
        param->out->layout() != DataLayout::ND ? param->out->layout() : param->input->layout();
    auto resolved_output = AllocateOutput(param->out, out_shape, param->input->data_type());
    if (resolved_output == nullptr) return -1;
    param->out = std::move(resolved_output);
    param->out->Resize(out_shape);
    param->out->set_data_type(param->input->data_type());
    param->out->set_quantization(output_quantization);
    param->out->set_layout(output_layout);
    return 0;
}

inline int32_t InferIdentityOutputShape(const std::shared_ptr<Tensor>& input, const std::shared_ptr<Tensor>& output,
                                        std::shared_ptr<Tensor>* resolved_output) {
    if (input == nullptr || output == nullptr || resolved_output == nullptr) {
        return -1;
    }
    *resolved_output = output;
    return InferSameTypeOutput(input, resolved_output, input->dims().data());
}

}  // namespace tensor_op_detail
}  // namespace operators
}  // namespace feather

#endif  // FEATHER_OPERATOR_TENSOR_OP_UTILS_H
