#ifndef FEATHER_KERNEL_COMMON_QUANTIZE_LINEAR_UTILS_H
#define FEATHER_KERNEL_COMMON_QUANTIZE_LINEAR_UTILS_H

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#if defined(FEATHER_WITH_OPENMP)
#include <omp.h>
#endif

#include "quant/quantization.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {
namespace quantize_linear_detail {

template <typename Function>
inline void ParallelForElements(size_t count, Function function) {
#if defined(FEATHER_WITH_OPENMP)
    if (count >= 4096 && !omp_in_parallel()) {
#pragma omp parallel for schedule(static)
        for (int64_t index = 0; index < static_cast<int64_t>(count); ++index) {
            function(static_cast<size_t>(index));
        }
        return;
    }
#endif
    for (size_t index = 0; index < count; ++index) function(index);
}

inline float DecodeFloat16(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    const uint32_t exponent = (bits >> 10) & 0x1fu;
    const uint32_t fraction = bits & 0x03ffu;
    uint32_t result = sign;
    if (exponent == 0) {
        if (fraction != 0) {
            float value = std::ldexp(static_cast<float>(fraction), -24);
            return sign == 0 ? value : -value;
        }
    } else if (exponent == 0x1fu) {
        result |= 0x7f800000u | (fraction << 13);
    } else {
        result |= ((exponent + (127 - 15)) << 23) | (fraction << 13);
    }
    float value = 0.0f;
    std::memcpy(&value, &result, sizeof(value));
    return value;
}

inline uint16_t EncodeFloat16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t exponent = (bits >> 23) & 0xffu;
    const uint32_t fraction = bits & 0x7fffffu;
    if (exponent == 0xffu) return static_cast<uint16_t>(sign | 0x7c00u | (fraction != 0 ? 0x0200u : 0));
    const int32_t half_exponent = static_cast<int32_t>(exponent) - 127 + 15;
    if (half_exponent >= 0x1f) return static_cast<uint16_t>(sign | 0x7c00u);
    if (half_exponent <= 0) {
        if (half_exponent < -10) return static_cast<uint16_t>(sign);
        const uint32_t mantissa = fraction | 0x800000u;
        const int shift = 14 - half_exponent;
        uint32_t rounded = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u))) ++rounded;
        return static_cast<uint16_t>(sign | rounded);
    }
    uint32_t rounded_fraction = fraction >> 13;
    const uint32_t remainder = fraction & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded_fraction & 1u))) {
        ++rounded_fraction;
        if (rounded_fraction == 0x400u) {
            rounded_fraction = 0;
            return static_cast<uint16_t>(sign | (static_cast<uint32_t>(half_exponent + 1) << 10));
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(half_exponent) << 10) | rounded_fraction);
}

inline float DecodeBFloat16(uint16_t bits) {
    const uint32_t full = static_cast<uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &full, sizeof(value));
    return value;
}

inline uint16_t EncodeBFloat16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t low = bits & 0xffffu;
    uint32_t high = bits >> 16;
    if (low > 0x8000u || (low == 0x8000u && (high & 1u))) ++high;
    return static_cast<uint16_t>(high);
}

inline bool ReadFloatTensor(const Tensor& tensor, std::vector<float>* values) {
    if (values == nullptr || !tensor.IsInitialized() || tensor.numel() <= 0) return false;
    const size_t count = static_cast<size_t>(tensor.numel());
    values->resize(count);
    const auto* raw = static_cast<const uint8_t*>(tensor.raw_data());
    switch (tensor.data_type()) {
        case DataType::FP32:
            std::memcpy(values->data(), raw, count * sizeof(float));
            return true;
        case DataType::FP16:
            for (size_t i = 0; i < count; ++i) {
                uint16_t bits = 0;
                std::memcpy(&bits, raw + i * sizeof(bits), sizeof(bits));
                (*values)[i] = DecodeFloat16(bits);
            }
            return true;
        case DataType::BF16:
            for (size_t i = 0; i < count; ++i) {
                uint16_t bits = 0;
                std::memcpy(&bits, raw + i * sizeof(bits), sizeof(bits));
                (*values)[i] = DecodeBFloat16(bits);
            }
            return true;
        default:
            return false;
    }
}

inline bool ReadZeroPointTensor(const Tensor& tensor, std::vector<int32_t>* values) {
    if (values == nullptr || !tensor.IsInitialized() || tensor.numel() <= 0) return false;
    const size_t count = static_cast<size_t>(tensor.numel());
    values->resize(count);
    const auto* raw = static_cast<const uint8_t*>(tensor.raw_data());
    for (size_t i = 0; i < count; ++i) {
        int64_t value = 0;
        switch (tensor.data_type()) {
            case DataType::INT8: {
                int8_t x = 0;
                std::memcpy(&x, raw + i, sizeof(x));
                value = x;
                break;
            }
            case DataType::UINT8: {
                uint8_t x = 0;
                std::memcpy(&x, raw + i, sizeof(x));
                value = x;
                break;
            }
            case DataType::INT32: {
                int32_t x = 0;
                std::memcpy(&x, raw + i * sizeof(x), sizeof(x));
                value = x;
                break;
            }
            case DataType::INT64: {
                int64_t x = 0;
                std::memcpy(&x, raw + i * sizeof(x), sizeof(x));
                value = x;
                break;
            }
            default:
                return false;
        }
        if (value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max()) return false;
        (*values)[i] = static_cast<int32_t>(value);
    }
    return true;
}

inline bool BuildQuantizationParams(const Tensor& input, const Tensor& scale_tensor, const Tensor& zero_tensor,
                                    int64_t requested_axis, QuantizationParams* params) {
    if (params == nullptr) return false;
    std::vector<float> scales;
    std::vector<int32_t> zero_points;
    if (!ReadFloatTensor(scale_tensor, &scales) || !ReadZeroPointTensor(zero_tensor, &zero_points) ||
        scales.empty() || zero_points.empty() || (zero_points.size() != 1 && zero_points.size() != scales.size())) {
        return false;
    }
    params->enabled = true;
    params->scales = std::move(scales);
    params->zero_points = std::move(zero_points);
    params->scale = params->scales.front();
    params->zero_point = params->zero_points.front();
    if (params->scales.size() == 1) {
        params->granularity = QuantizationGranularity::kPerTensor;
        params->axis = -1;
    } else {
        int64_t axis = requested_axis;
        if (axis < 0) axis += static_cast<int64_t>(input.dims().size());
        params->granularity = QuantizationGranularity::kPerChannel;
        params->axis = axis;
    }
    return ValidateQuantizationParams(*params, input.dims().data());
}

inline float ReadValue(const Tensor& tensor, size_t index) {
    const auto* raw = static_cast<const uint8_t*>(tensor.raw_data());
    switch (tensor.data_type()) {
        case DataType::FP32: {
            float value = 0.0f;
            std::memcpy(&value, raw + index * sizeof(value), sizeof(value));
            return value;
        }
        case DataType::FP16: {
            uint16_t bits = 0;
            std::memcpy(&bits, raw + index * sizeof(bits), sizeof(bits));
            return DecodeFloat16(bits);
        }
        case DataType::BF16: {
            uint16_t bits = 0;
            std::memcpy(&bits, raw + index * sizeof(bits), sizeof(bits));
            return DecodeBFloat16(bits);
        }
        default:
            return std::numeric_limits<float>::quiet_NaN();
    }
}

inline bool WriteFloatValues(Tensor& output, DataType dtype, const std::vector<float>& values) {
    const size_t element_bytes = DataTypeBytes(dtype);
    if (!output.IsInitialized() || element_bytes == 0 || output.memory_size() < values.size() * element_bytes) return false;
    if (dtype == DataType::FP32) {
        float* destination = output.mutable_data<float>();
        std::memcpy(destination, values.data(), values.size() * sizeof(float));
        output.set_data_type(DataType::FP32);
        return true;
    }
    auto* raw = static_cast<uint8_t*>(output.mutable_data(values.size() * element_bytes));
    for (size_t i = 0; i < values.size(); ++i) {
        const uint16_t bits = dtype == DataType::FP16 ? EncodeFloat16(values[i]) : EncodeBFloat16(values[i]);
        std::memcpy(raw + i * sizeof(bits), &bits, sizeof(bits));
    }
    output.set_data_type(dtype);
    return true;
}

inline int32_t ComputeQuantizeLinear(operators::QuantizeLinearParam* param) {
    if (param == nullptr || param->input == nullptr || param->scale == nullptr || param->zero_point == nullptr ||
        param->out == nullptr || !param->input->IsInitialized() || param->input->numel() <= 0) return -1;
    QuantizationParams quantization;
    if (!BuildQuantizationParams(*param->input, *param->scale, *param->zero_point, param->axis, &quantization)) return -1;
    const size_t count = static_cast<size_t>(param->input->numel());
    if (!param->out->IsInitialized() || param->out->memory_size() < count) return -1;

    if (param->input->data_type() == DataType::FP32 &&
        quantization.granularity == QuantizationGranularity::kPerTensor && quantization.scales.size() == 1 &&
        quantization.zero_points.size() == 1) {
        const float* source = param->input->data<float>();
        int8_t* destination = param->out->mutable_data<int8_t>();
        if (source == nullptr || destination == nullptr) return -1;
        const float scale = quantization.scale;
        const int32_t zero_point = quantization.zero_point;
        ParallelForElements(count, [&](size_t index) {
            destination[index] = QuantizeInt8Value(source[index], scale, zero_point);
        });
        param->out->Resize(param->input->dims().data());
        param->out->set_quantization(quantization);
        param->out->set_data_type(DataType::INT8);
        return 0;
    }

    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) values[i] = ReadValue(*param->input, i);
    int8_t* destination = param->out->mutable_data<int8_t>();
    const int32_t result = QuantizeInt8(values.data(), destination, param->input->dims().data(), quantization);
    if (result == 0) {
        param->out->Resize(param->input->dims().data());
        param->out->set_quantization(quantization);
        param->out->set_data_type(DataType::INT8);
    }
    return result;
}

inline int32_t ComputeDequantizeLinear(operators::DequantizeLinearParam* param) {
    if (param == nullptr || param->input == nullptr || param->scale == nullptr || param->zero_point == nullptr ||
        param->out == nullptr || !param->input->IsInitialized() || param->input->numel() <= 0 ||
        param->input->data_type() != DataType::INT8 ||
        (param->to != DataType::FP32 && param->to != DataType::FP16 && param->to != DataType::BF16)) return -1;
    QuantizationParams quantization;
    if (!BuildQuantizationParams(*param->input, *param->scale, *param->zero_point, param->axis, &quantization)) return -1;
    const size_t count = static_cast<size_t>(param->input->numel());

    if (quantization.granularity == QuantizationGranularity::kPerTensor && quantization.scales.size() == 1 &&
        quantization.zero_points.size() == 1) {
        const int8_t* source = param->input->data<int8_t>();
        if (source == nullptr) return -1;
        const float scale = quantization.scale;
        const int32_t zero_point = quantization.zero_point;
        if (param->to == DataType::FP32) {
            float* destination = param->out->mutable_data<float>();
            if (destination == nullptr) return -1;
            ParallelForElements(count, [&](size_t index) {
                destination[index] =
                    (static_cast<int32_t>(source[index]) - zero_point) * scale;
            });
            param->out->Resize(param->input->dims().data());
            param->out->set_quantization(QuantizationParams{});
            param->out->set_data_type(DataType::FP32);
            return 0;
        }
        if (param->to == DataType::FP16 || param->to == DataType::BF16) {
            auto* destination = static_cast<uint8_t*>(param->out->mutable_data(count * sizeof(uint16_t)));
            if (destination == nullptr) return -1;
            ParallelForElements(count, [&](size_t index) {
                const float value =
                    (static_cast<int32_t>(source[index]) - zero_point) * scale;
                const uint16_t bits = param->to == DataType::FP16 ? EncodeFloat16(value) : EncodeBFloat16(value);
                std::memcpy(destination + index * sizeof(uint16_t), &bits, sizeof(bits));
            });
            param->out->Resize(param->input->dims().data());
            param->out->set_quantization(QuantizationParams{});
            param->out->set_data_type(param->to);
            return 0;
        }
    }

    std::vector<float> values(count);
    const int8_t* source = static_cast<const int8_t*>(param->input->raw_data());
    if (DequantizeInt8(source, values.data(), param->input->dims().data(), quantization) != 0) return -1;
    if (!WriteFloatValues(*param->out, param->to, values)) return -1;
    param->out->Resize(param->input->dims().data());
    param->out->set_quantization(QuantizationParams{});
    return 0;
}

}  // namespace quantize_linear_detail
}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_COMMON_QUANTIZE_LINEAR_UTILS_H
