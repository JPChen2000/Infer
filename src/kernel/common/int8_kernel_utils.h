#ifndef FEATHER_KERNEL_COMMON_INT8_KERNEL_UTILS_H
#define FEATHER_KERNEL_COMMON_INT8_KERNEL_UTILS_H

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "core/tensor.h"
#include "quant/quantization.h"

namespace feather {
namespace kernel {
namespace int8_detail {

struct QuantizationView {
    float scale{1.0f};
    int32_t zero_point{0};
    QuantizationGranularity granularity{QuantizationGranularity::kPerTensor};
    int64_t axis{-1};
    std::vector<float> scales;
    std::vector<int32_t> zero_points;

    float scale_for(size_t index) const {
        if (scales.empty() || scales.size() == 1) {
            return scale;
        }
        return scales[index];
    }

    int32_t zero_point_for(size_t index) const {
        if (zero_points.empty() || zero_points.size() == 1) {
            return zero_point;
        }
        return zero_points[index];
    }
};

inline bool BuildQuantizationView(const Tensor& tensor, bool allow_per_channel, int64_t expected_axis,
                                  int64_t expected_channels, QuantizationView* view) {
    if (view == nullptr || tensor.data_type() != DataType::INT8 || !tensor.IsInitialized()) {
        return false;
    }
    const QuantizationParams& params = tensor.quantization();
    if (!params.enabled || !ValidateQuantizationParams(params, tensor.dims().data())) {
        return false;
    }
    if (params.granularity == QuantizationGranularity::kPerTensor) {
        if (allow_per_channel && expected_channels <= 0) {
            return false;
        }
    } else if (params.granularity == QuantizationGranularity::kPerChannel) {
        if (!allow_per_channel || expected_axis < 0 || params.axis != expected_axis ||
            expected_channels <= 0 || params.scales.size() != static_cast<size_t>(expected_channels)) {
            return false;
        }
    } else {
        return false;
    }

    view->scale = params.scale_at(0);
    view->zero_point = params.zero_point_at(0);
    view->granularity = params.granularity;
    view->axis = params.axis;
    view->scales = params.scales;
    view->zero_points = params.zero_points;
    return true;
}

inline bool BuildInputQuantizationView(const std::shared_ptr<Tensor>& tensor, QuantizationView* view) {
    return tensor != nullptr && BuildQuantizationView(*tensor, false, -1, 1, view);
}

inline bool BuildWeightQuantizationView(const std::shared_ptr<Tensor>& tensor, int64_t channel_axis,
                                        int64_t channels, QuantizationView* view) {
    return tensor != nullptr && BuildQuantizationView(*tensor, true, channel_axis, channels, view);
}

inline bool BuildOutputQuantizationView(const std::shared_ptr<Tensor>& tensor, QuantizationView* view) {
    return tensor != nullptr && BuildQuantizationView(*tensor, false, -1, 1, view);
}

inline bool FitsInt32(int64_t value) {
    return value >= static_cast<int64_t>(std::numeric_limits<int32_t>::min()) &&
           value <= static_cast<int64_t>(std::numeric_limits<int32_t>::max());
}

inline bool AddInt32Bias(int64_t* accumulator, int32_t bias) {
    if (accumulator == nullptr) {
        return false;
    }
    const int64_t result = *accumulator + static_cast<int64_t>(bias);
    if (!FitsInt32(result)) {
        return false;
    }
    *accumulator = result;
    return true;
}

inline bool IsVectorBias(const Tensor& bias, int64_t channels) {
    if (!bias.IsInitialized() || bias.data_type() != DataType::INT32 || bias.dims().empty() ||
        bias.dims()[bias.dims().size() - 1] != channels || bias.numel() != channels) {
        return false;
    }
    for (size_t axis = 0; axis + 1 < bias.dims().size(); ++axis) {
        if (bias.dims()[axis] != 1) {
            return false;
        }
    }
    return true;
}

inline bool IsMatrixBias(const Tensor& bias, int64_t rows, int64_t channels) {
    return bias.IsInitialized() && bias.data_type() == DataType::INT32 && bias.dims().size() == 2 &&
           bias.dims()[0] == rows && bias.dims()[1] == channels;
}

inline bool ValidateLinearBias(const std::shared_ptr<Tensor>& bias, int64_t rows, int64_t channels) {
    if (bias == nullptr) {
        return true;
    }
    return IsVectorBias(*bias, channels) || IsMatrixBias(*bias, rows, channels);
}

inline int32_t ReadLinearBias(const std::shared_ptr<Tensor>& bias, int64_t row, int64_t channel,
                              int64_t channels) {
    if (bias == nullptr) {
        return 0;
    }
    const int32_t* values = bias->data<int32_t>();
    return bias->numel() == channels ? values[channel] : values[row * channels + channel];
}

inline bool ValidateConvBias(const std::shared_ptr<Tensor>& bias, int64_t channels) {
    return bias == nullptr || (bias->IsInitialized() && bias->data_type() == DataType::INT32 &&
                               bias->dims().size() == 1 && bias->dims()[0] == channels);
}

inline int32_t ReadConvBias(const std::shared_ptr<Tensor>& bias, int64_t channel) {
    return bias == nullptr ? 0 : bias->data<int32_t>()[channel];
}

inline bool QuantizeReal(double real_value, const QuantizationView& output, int8_t* result) {
    if (result == nullptr || !std::isfinite(real_value) || !std::isfinite(output.scale) || output.scale <= 0.0f) {
        return false;
    }
    const double transformed = real_value / static_cast<double>(output.scale) + output.zero_point;
    if (!std::isfinite(transformed)) {
        return false;
    }
    const double rounded = std::round(transformed);
    const double clamped = std::max(-128.0, std::min(127.0, rounded));
    *result = static_cast<int8_t>(clamped);
    return true;
}

inline bool QuantizeAccumulator(int64_t accumulator, double accumulator_scale,
                                const QuantizationView& output, int8_t* result) {
    if (!std::isfinite(accumulator_scale)) {
        return false;
    }
    return QuantizeReal(static_cast<double>(accumulator) * accumulator_scale, output, result);
}

}  // namespace int8_detail
}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_COMMON_INT8_KERNEL_UTILS_H
