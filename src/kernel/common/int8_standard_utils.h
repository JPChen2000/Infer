#ifndef FEATHER_KERNEL_COMMON_INT8_STANDARD_UTILS_H
#define FEATHER_KERNEL_COMMON_INT8_STANDARD_UTILS_H

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <vector>

#if defined(FEATHER_WITH_OPENMP)
#include <omp.h>
#endif

#include "core/tensor.h"
#include "quant/quantization.h"
#include "src/kernel/common/tensor_op_utils.h"

namespace feather {
namespace kernel {
namespace int8_standard_detail {

struct View {
    float scale{1.0f};
    int32_t zero_point{0};
};

inline bool Ready(const Tensor* tensor) {
    return tensor != nullptr && tensor->data_type() == DataType::INT8 && tensor->IsInitialized() &&
           tensor->numel() > 0 && tensor->memory_size() >= static_cast<size_t>(tensor->numel());
}

inline bool BuildView(const Tensor* tensor, View* view) {
    if (view == nullptr || !Ready(tensor)) return false;
    const auto& params = tensor->quantization();
    if (!params.enabled) {
        *view = View{};
        return true;
    }
    if (!ValidateQuantizationParams(params, tensor->dims().data()) ||
        params.granularity != QuantizationGranularity::kPerTensor) {
        return false;
    }
    view->scale = params.scale_at(0);
    view->zero_point = params.zero_point_at(0);
    return std::isfinite(view->scale) && view->scale > 0.0f && view->zero_point >= -128 && view->zero_point <= 127;
}

inline bool BuildOutputView(const Tensor* tensor, View* view) {
    return BuildView(tensor, view);
}

inline float ReadReal(const Tensor* tensor, int64_t index, const View& view) {
    return (static_cast<int32_t>(tensor->data<int8_t>()[index]) - view.zero_point) * view.scale;
}

inline bool WriteReal(Tensor* tensor, int64_t index, float value, const View& view) {
    if (tensor == nullptr || !std::isfinite(value) || !std::isfinite(view.scale) || view.scale <= 0.0f) {
        return false;
    }
    tensor->mutable_data<int8_t>()[index] = QuantizeInt8Value(value, view.scale, view.zero_point);
    return true;
}

inline std::vector<int64_t> Strides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> result(dims.size(), 1);
    int64_t suffix = 1;
    for (int64_t axis = static_cast<int64_t>(dims.size()) - 1; axis >= 0; --axis) {
        if (dims[static_cast<size_t>(axis)] <= 0 ||
            suffix > std::numeric_limits<int64_t>::max() / dims[static_cast<size_t>(axis)]) return {};
        result[static_cast<size_t>(axis)] = suffix;
        suffix *= dims[static_cast<size_t>(axis)];
    }
    return result;
}

inline bool BroadcastShape(const std::vector<int64_t>& lhs, const std::vector<int64_t>& rhs,
                           std::vector<int64_t>* output) {
    if (output == nullptr) return false;
    const size_t rank = std::max(lhs.size(), rhs.size());
    output->assign(rank, 1);
    for (size_t axis = 0; axis < rank; ++axis) {
        const int64_t lhs_dim = axis < rank - lhs.size() ? 1 : lhs[axis - (rank - lhs.size())];
        const int64_t rhs_dim = axis < rank - rhs.size() ? 1 : rhs[axis - (rank - rhs.size())];
        if (lhs_dim <= 0 || rhs_dim <= 0 || (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1)) return false;
        (*output)[axis] = std::max(lhs_dim, rhs_dim);
    }
    return true;
}

inline int64_t BroadcastOffset(const std::vector<int64_t>& coordinates,
                               const std::vector<int64_t>& dims,
                               const std::vector<int64_t>& strides) {
    const size_t gap = coordinates.size() - dims.size();
    int64_t offset = 0;
    for (size_t axis = 0; axis < dims.size(); ++axis) {
        offset += (dims[axis] == 1 ? 0 : coordinates[axis + gap]) * strides[axis];
    }
    return offset;
}

inline void LinearToCoords(int64_t linear, const std::vector<int64_t>& dims,
                           const std::vector<int64_t>& strides, std::vector<int64_t>* coordinates) {
    coordinates->assign(dims.size(), 0);
    for (size_t axis = 0; axis < dims.size(); ++axis) {
        (*coordinates)[axis] = linear / strides[axis];
        linear %= strides[axis];
    }
}

inline int32_t Copy(const Tensor* input, Tensor* output) {
    View input_view;
    View output_view;
    if (!BuildView(input, &input_view) || !BuildOutputView(output, &output_view) ||
        input->numel() != output->numel() || output->memory_size() < static_cast<size_t>(output->numel())) return -1;
    for (int64_t index = 0; index < input->numel(); ++index) {
        if (!WriteReal(output, index, ReadReal(input, index, input_view), output_view)) return -1;
    }
    return 0;
}

template <typename Function>
inline int32_t Binary(operators::BinaryParam* param, Function function) {
    if (param == nullptr || !Ready(param->lhs.get()) || !Ready(param->rhs.get()) ||
        !Ready(param->out.get())) return -1;
    View lhs_view;
    View rhs_view;
    View output_view;
    if (!BuildView(param->lhs.get(), &lhs_view) || !BuildView(param->rhs.get(), &rhs_view) ||
        !BuildOutputView(param->out.get(), &output_view)) return -1;
    std::vector<int64_t> output_dims;
    if (!BroadcastShape(param->lhs->dims().data(), param->rhs->dims().data(), &output_dims) ||
        param->out->dims().data() != output_dims) return -1;
    const auto output_strides = Strides(output_dims);
    const auto lhs_strides = Strides(param->lhs->dims().data());
    const auto rhs_strides = Strides(param->rhs->dims().data());
    if (output_strides.empty() || lhs_strides.empty() || rhs_strides.empty()) return -1;

    const bool lhs_matches_output = param->lhs->dims().data() == output_dims;
    const bool rhs_matches_output = param->rhs->dims().data() == output_dims;
    const bool lhs_scalar = param->lhs->numel() == 1;
    const bool rhs_scalar = param->rhs->numel() == 1;
    if ((lhs_matches_output && rhs_matches_output) ||
        (lhs_scalar && rhs_matches_output) || (rhs_scalar && lhs_matches_output)) {
        const int8_t* lhs_data = param->lhs->data<int8_t>();
        const int8_t* rhs_data = param->rhs->data<int8_t>();
        int8_t* output_data = param->out->mutable_data<int8_t>();
        if (lhs_data == nullptr || rhs_data == nullptr || output_data == nullptr) return -1;
        const float lhs_scale = lhs_view.scale;
        const float rhs_scale = rhs_view.scale;
        const float output_scale = output_view.scale;
        const int32_t lhs_zero_point = lhs_view.zero_point;
        const int32_t rhs_zero_point = rhs_view.zero_point;
        const int32_t output_zero_point = output_view.zero_point;
        const int64_t count = param->out->numel();
        std::atomic<bool> failed{false};
        auto run_index = [&](int64_t index) {
            const int64_t lhs_index = lhs_matches_output ? index : 0;
            const int64_t rhs_index = rhs_matches_output ? index : 0;
            const float lhs = (static_cast<int32_t>(lhs_data[lhs_index]) - lhs_zero_point) * lhs_scale;
            const float rhs = (static_cast<int32_t>(rhs_data[rhs_index]) - rhs_zero_point) * rhs_scale;
            const float value = function(lhs, rhs);
            if (!std::isfinite(value)) {
                failed.store(true, std::memory_order_relaxed);
                return;
            }
            output_data[index] = QuantizeInt8Value(value, output_scale, output_zero_point);
        };
#if defined(FEATHER_WITH_OPENMP)
        if (count >= 4096 && !omp_in_parallel()) {
#pragma omp parallel for schedule(static)
            for (int64_t index = 0; index < count; ++index) run_index(index);
        } else
#endif
        {
            for (int64_t index = 0; index < count; ++index) run_index(index);
        }
        return failed.load(std::memory_order_relaxed) ? -1 : 0;
    }

    std::vector<int64_t> coordinates;
    for (int64_t index = 0; index < param->out->numel(); ++index) {
        LinearToCoords(index, output_dims, output_strides, &coordinates);
        const float lhs = ReadReal(param->lhs.get(), BroadcastOffset(coordinates, param->lhs->dims().data(), lhs_strides), lhs_view);
        const float rhs = ReadReal(param->rhs.get(), BroadcastOffset(coordinates, param->rhs->dims().data(), rhs_strides), rhs_view);
        if (!WriteReal(param->out.get(), index, function(lhs, rhs), output_view)) return -1;
    }
    return 0;
}

template <typename Function>
inline int32_t Unary(operators::UnaryParam* param, Function function) {
    if (param == nullptr || !Ready(param->input.get()) || !Ready(param->out.get()) ||
        param->input->dims().data() != param->out->dims().data()) return -1;
    View input_view;
    View output_view;
    if (!BuildView(param->input.get(), &input_view) || !BuildOutputView(param->out.get(), &output_view)) return -1;
    const int8_t* input_data = param->input->data<int8_t>();
    int8_t* output_data = param->out->mutable_data<int8_t>();
    if (input_data == nullptr || output_data == nullptr) return -1;
    const float input_scale = input_view.scale;
    const float output_scale = output_view.scale;
    const int32_t input_zero_point = input_view.zero_point;
    const int32_t output_zero_point = output_view.zero_point;
    const int64_t count = param->input->numel();
    std::atomic<bool> failed{false};
    auto run_index = [&](int64_t index) {
        const float input = (static_cast<int32_t>(input_data[index]) - input_zero_point) * input_scale;
        const float value = function(input);
        if (!std::isfinite(value)) {
            failed.store(true, std::memory_order_relaxed);
            return;
        }
        output_data[index] = QuantizeInt8Value(value, output_scale, output_zero_point);
    };
#if defined(FEATHER_WITH_OPENMP)
    if (count >= 4096 && !omp_in_parallel()) {
#pragma omp parallel for schedule(static)
        for (int64_t index = 0; index < count; ++index) run_index(index);
    } else
#endif
    {
        for (int64_t index = 0; index < count; ++index) run_index(index);
    }
    return failed.load(std::memory_order_relaxed) ? -1 : 0;
}

inline std::vector<int64_t> NormalizeReductionAxes(const std::vector<int64_t>& axes, int64_t rank) {
    std::vector<int64_t> result = axes;
    if (result.empty()) {
        result.resize(static_cast<size_t>(rank));
        std::iota(result.begin(), result.end(), 0);
        return result;
    }
    for (auto& axis : result) if (axis < 0) axis += rank;
    std::sort(result.begin(), result.end());
    return result;
}

inline int32_t Reduce(operators::ReduceSumParam* param, bool mean) {
    if (param == nullptr || !Ready(param->input.get()) || !Ready(param->out.get())) return -1;
    const auto input_dims = param->input->dims().data();
    const auto axes = NormalizeReductionAxes(param->axes, static_cast<int64_t>(input_dims.size()));
    const std::set<int64_t> axis_set(axes.begin(), axes.end());
    if (axis_set.size() != axes.size() || std::any_of(axes.begin(), axes.end(), [&input_dims](int64_t axis) {
            return axis < 0 || axis >= static_cast<int64_t>(input_dims.size());
        })) return -1;
    const auto output_dims = param->out->dims().data();
    const auto input_strides = Strides(input_dims);
    const auto output_strides = Strides(output_dims);
    if (input_strides.empty() || output_strides.empty()) return -1;
    View input_view;
    View output_view;
    if (!BuildView(param->input.get(), &input_view) || !BuildOutputView(param->out.get(), &output_view)) return -1;
    std::vector<double> sums(static_cast<size_t>(param->out->numel()), 0.0);
    std::vector<int64_t> input_coordinates;
    std::vector<int64_t> output_coordinates;
    double count = 1.0;
    for (const int64_t axis : axes) count *= static_cast<double>(input_dims[static_cast<size_t>(axis)]);
    for (int64_t index = 0; index < param->input->numel(); ++index) {
        LinearToCoords(index, input_dims, input_strides, &input_coordinates);
        output_coordinates.clear();
        for (int64_t axis = 0; axis < static_cast<int64_t>(input_dims.size()); ++axis) {
            if (axis_set.count(axis) != 0) {
                if (param->keepdims) output_coordinates.push_back(0);
            } else {
                output_coordinates.push_back(input_coordinates[static_cast<size_t>(axis)]);
            }
        }
        if (output_coordinates.empty()) output_coordinates.push_back(0);
        int64_t output_index = 0;
        for (size_t axis = 0; axis < output_coordinates.size(); ++axis) output_index += output_coordinates[axis] * output_strides[axis];
        sums[static_cast<size_t>(output_index)] += ReadReal(param->input.get(), index, input_view);
    }
    for (int64_t index = 0; index < param->out->numel(); ++index) {
        if (!WriteReal(param->out.get(), index, static_cast<float>(mean ? sums[static_cast<size_t>(index)] / count : sums[static_cast<size_t>(index)]), output_view)) return -1;
    }
    return 0;
}

inline int32_t ReduceMean(operators::ReduceMeanParam* param) {
    if (param == nullptr) return -1;
    operators::ReduceSumParam reduce;
    reduce.input = param->input;
    reduce.out = param->out;
    reduce.axes = param->axes;
    reduce.keepdims = param->keepdims;
    return Reduce(&reduce, true);
}

inline int32_t Softmax(operators::SoftmaxParam* param) {
    if (param == nullptr || !Ready(param->input.get()) || !Ready(param->out.get()) ||
        param->input->dims().data() != param->out->dims().data()) return -1;
    View input_view;
    View output_view;
    if (!BuildView(param->input.get(), &input_view) || !BuildOutputView(param->out.get(), &output_view)) return -1;
    const auto dims = param->input->dims().data();
    const int64_t rank = static_cast<int64_t>(dims.size());
    const int64_t axis = param->axis < 0 ? param->axis + rank : param->axis;
    if (axis < 0 || axis >= rank) return -1;
    int64_t outer = 1;
    int64_t inner = 1;
    for (int64_t i = 0; i < axis; ++i) outer *= dims[static_cast<size_t>(i)];
    for (int64_t i = axis + 1; i < rank; ++i) inner *= dims[static_cast<size_t>(i)];
    const int64_t axis_dim = dims[static_cast<size_t>(axis)];
    for (int64_t outer_index = 0; outer_index < outer; ++outer_index) {
        for (int64_t inner_index = 0; inner_index < inner; ++inner_index) {
            const int64_t base = outer_index * axis_dim * inner + inner_index;
            float maximum = -std::numeric_limits<float>::infinity();
            for (int64_t axis_index = 0; axis_index < axis_dim; ++axis_index) maximum = std::max(maximum, ReadReal(param->input.get(), base + axis_index * inner, input_view));
            float sum = 0.0f;
            for (int64_t axis_index = 0; axis_index < axis_dim; ++axis_index) sum += std::exp(ReadReal(param->input.get(), base + axis_index * inner, input_view) - maximum);
            for (int64_t axis_index = 0; axis_index < axis_dim; ++axis_index) {
                if (!WriteReal(param->out.get(), base + axis_index * inner, std::exp(ReadReal(param->input.get(), base + axis_index * inner, input_view) - maximum) / sum, output_view)) return -1;
            }
        }
    }
    return 0;
}

inline int32_t CopyFloatTransform(const Tensor* input, Tensor* output, const std::vector<int64_t>& output_dims) {
    if (!Ready(input) || !Ready(output) || input->numel() != output->numel() || output->dims().data() != output_dims) return -1;
    return Copy(input, output);
}

}  // namespace int8_standard_detail
}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_COMMON_INT8_STANDARD_UTILS_H
