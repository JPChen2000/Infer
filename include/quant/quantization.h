#ifndef FEATHER_QUANT_QUANTIZATION_H
#define FEATHER_QUANT_QUANTIZATION_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "util/types.h"

namespace feather {

inline bool IsSupportedInt8Granularity(QuantizationGranularity granularity) {
    return granularity == QuantizationGranularity::kPerTensor ||
           granularity == QuantizationGranularity::kPerChannel;
}

inline int64_t QuantizationNumel(const std::vector<int64_t>& dims) {
    int64_t numel = 1;
    for (const int64_t dim : dims) {
        if (dim <= 0 || numel > std::numeric_limits<int64_t>::max() / dim) {
            return 0;
        }
        numel *= dim;
    }
    return numel;
}

inline bool ValidateQuantizationParams(const QuantizationParams& params, const std::vector<int64_t>& dims) {
    if (!params.enabled) {
        return true;
    }
    if (!IsSupportedInt8Granularity(params.granularity) || QuantizationNumel(dims) <= 0) {
        return false;
    }

    const size_t scale_count = params.scales.empty() ? 1 : params.scales.size();
    if (params.granularity == QuantizationGranularity::kPerTensor) {
        if (scale_count != 1 || (!params.zero_points.empty() && params.zero_points.size() != 1)) {
            return false;
        }
    } else {
        if (params.axis < 0 || params.axis >= static_cast<int64_t>(dims.size()) ||
            scale_count != static_cast<size_t>(dims[static_cast<size_t>(params.axis)])) {
            return false;
        }
        if (!params.zero_points.empty() && params.zero_points.size() != 1 &&
            params.zero_points.size() != scale_count) {
            return false;
        }
    }
    for (size_t i = 0; i < scale_count; ++i) {
        const float scale = params.scale_at(i);
        if (!std::isfinite(scale) || scale <= 0.0f) {
            return false;
        }
        const int32_t zero_point = params.zero_point_at(i);
        if (zero_point < -128 || zero_point > 127) {
            return false;
        }
    }
    return true;
}

inline size_t QuantizationParameterIndex(const std::vector<int64_t>& dims, size_t linear_index,
                                         const QuantizationParams& params) {
    if (params.granularity == QuantizationGranularity::kPerTensor || dims.empty()) {
        return 0;
    }
    size_t stride = 1;
    for (size_t axis = static_cast<size_t>(params.axis) + 1; axis < dims.size(); ++axis) {
        stride *= static_cast<size_t>(dims[axis]);
    }
    return (linear_index / stride) % static_cast<size_t>(dims[static_cast<size_t>(params.axis)]);
}

inline int8_t QuantizeInt8Value(float value, float scale, int32_t zero_point) {
    const double transformed = static_cast<double>(value) / static_cast<double>(scale) + zero_point;
    const double rounded = std::round(transformed);
    const double clamped = std::max(-128.0, std::min(127.0, rounded));
    return static_cast<int8_t>(clamped);
}

inline float DequantizeInt8Value(int8_t value, float scale, int32_t zero_point) {
    return (static_cast<int32_t>(value) - zero_point) * scale;
}

inline int32_t QuantizeInt8(const float* source, int8_t* destination, const std::vector<int64_t>& dims,
                            const QuantizationParams& params) {
    if (source == nullptr || destination == nullptr || !ValidateQuantizationParams(params, dims)) {
        return -1;
    }
    const int64_t count = QuantizationNumel(dims);
    for (int64_t i = 0; i < count; ++i) {
        const size_t parameter_index = QuantizationParameterIndex(dims, static_cast<size_t>(i), params);
        destination[i] = QuantizeInt8Value(source[i], params.scale_at(parameter_index),
                                           params.zero_point_at(parameter_index));
    }
    return 0;
}

inline int32_t QuantizeInt8(const float* source, int8_t* destination, size_t count,
                            const QuantizationParams& params) {
    if (count > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        return -1;
    }
    return QuantizeInt8(source, destination, std::vector<int64_t>{static_cast<int64_t>(count)}, params);
}

inline int32_t DequantizeInt8(const int8_t* source, float* destination, const std::vector<int64_t>& dims,
                              const QuantizationParams& params) {
    if (source == nullptr || destination == nullptr || !ValidateQuantizationParams(params, dims)) {
        return -1;
    }
    const int64_t count = QuantizationNumel(dims);
    for (int64_t i = 0; i < count; ++i) {
        const size_t parameter_index = QuantizationParameterIndex(dims, static_cast<size_t>(i), params);
        destination[i] = DequantizeInt8Value(source[i], params.scale_at(parameter_index),
                                             params.zero_point_at(parameter_index));
    }
    return 0;
}

inline int32_t DequantizeInt8(const int8_t* source, float* destination, size_t count,
                              const QuantizationParams& params) {
    if (count > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        return -1;
    }
    return DequantizeInt8(source, destination, std::vector<int64_t>{static_cast<int64_t>(count)}, params);
}

inline int32_t RequantizeInt32(const int32_t* source, int8_t* destination, const std::vector<int64_t>& dims,
                               const QuantizationParams& input, const QuantizationParams& output) {
    if (source == nullptr || destination == nullptr || !ValidateQuantizationParams(input, dims) ||
        !ValidateQuantizationParams(output, dims)) {
        return -1;
    }
    const int64_t count = QuantizationNumel(dims);
    for (int64_t i = 0; i < count; ++i) {
        const size_t input_index = QuantizationParameterIndex(dims, static_cast<size_t>(i), input);
        const size_t output_index = QuantizationParameterIndex(dims, static_cast<size_t>(i), output);
        const float real_value = static_cast<float>(source[i]) * input.scale_at(input_index);
        destination[i] = QuantizeInt8Value(real_value, output.scale_at(output_index),
                                           output.zero_point_at(output_index));
    }
    return 0;
}

inline int32_t RequantizeInt32(const int32_t* source, int8_t* destination, size_t count,
                               const QuantizationParams& input, const QuantizationParams& output) {
    if (count > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        return -1;
    }
    return RequantizeInt32(source, destination, std::vector<int64_t>{static_cast<int64_t>(count)}, input, output);
}

}  // namespace feather

#endif  // FEATHER_QUANT_QUANTIZATION_H
