#ifndef FEATHER_INFER_UTIL_TYPES_H
#define FEATHER_INFER_UTIL_TYPES_H
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <variant>

#include "util/fp8.h"
namespace feather {

struct BFloat16;

enum class DataType {
    INT4 = 0,
    INT8 = 1,
    UINT8 = 2,
    FP16 = 3,
    FP32 = 4,
    INT32 = 5,
    INT64 = 6,
    STRING = 7,
    TENSOR = 8,
    BOOL = 9,
    UNKNOWN = 10,
    BF16 = 11,
    FP8E4M3 = 12,
    FP8E5M2 = 13,
};

enum class QuantizationGranularity {
    kPerTensor = 0,
    kPerChannel = 1,
    kPerGroup = 2,
    kPerBlock = 3,
};

struct QuantizationParams {
    bool enabled{false};
    float scale{1.0f};
    QuantizationGranularity granularity{QuantizationGranularity::kPerTensor};
    int64_t axis{-1};
    int64_t block_size{0};
    // Keep the original five aggregate-initializer fields above stable for
    // existing FP8 callers. INT8 extends the metadata after that boundary.
    int32_t zero_point{0};
    std::vector<float> scales{};
    std::vector<int32_t> zero_points{};

    float scale_at(size_t index = 0) const {
        if (!enabled) {
            return 1.0f;
        }
        if (scales.empty() || scales.size() == 1) {
            return scales.empty() ? scale : scales.front();
        }
        return scales.at(index);
    }

    int32_t zero_point_at(size_t index = 0) const {
        if (!enabled) {
            return 0;
        }
        if (zero_points.empty() || zero_points.size() == 1) {
            return zero_points.empty() ? zero_point : zero_points.front();
        }
        return zero_points.at(index);
    }
};

// FP8 kernels consume one scalar scale for an entire tensor. Disabled
// quantization denotes the implicit unit scale; enabled non-scalar schemes
// need a kernel that explicitly implements their indexing semantics.
inline bool HasCompatiblePerTensorQuantization(const QuantizationParams& quantization) {
    return !quantization.enabled || quantization.granularity == QuantizationGranularity::kPerTensor;
}

template<typename T>
struct DataTypeTrait {
    static DataType type() {
        if constexpr (std::is_same<T, float>::value) {
            return DataType::FP32;
        } else if constexpr (std::is_same<T, int32_t>::value) {
            return DataType::INT32;
        } else if constexpr (std::is_same<T, int64_t>::value) {
            return DataType::INT64;
        } else if constexpr (std::is_same<T, int8_t>::value) {
            return DataType::INT8;
        } else if constexpr (std::is_same<T, uint8_t>::value) {
            return DataType::UINT8;
        } else if constexpr (std::is_same<T, bool>::value) {
            return DataType::BOOL;
        } else if constexpr (std::is_same<T, uint16_t>::value) {
            return DataType::FP16;
        } else if constexpr (std::is_same<T, BFloat16>::value) {
            return DataType::BF16;
        } else if constexpr (std::is_same<T, Fp8E4M3>::value) {
            return DataType::FP8E4M3;
        } else if constexpr (std::is_same<T, Fp8E5M2>::value) {
            return DataType::FP8E5M2;
        } else if constexpr (std::is_same<T, std::string>::value) {
            return DataType::STRING;
        } else {
            return DataType::UNKNOWN;

        }
    }
};

enum class DeviceType {
    UNKNOWN,
    COMMON,
    X86,
    ARM32,
    ARM64,
    CUDA,
    ASCEND
};

inline DeviceType GetHostRuntimeDevice() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
    return DeviceType::X86;
#elif defined(__aarch64__)
    return DeviceType::ARM64;
#elif defined(__arm__) || defined(_M_ARM)
    return DeviceType::ARM32;
#else
    return DeviceType::COMMON;
#endif
}


enum class DataLayout { 
    NCHW = 0, 
    NHWC = 1, 
    ND = 2 
};

struct ImageShape4D {
    int64_t n{0};
    int64_t c{0};
    int64_t h{0};
    int64_t w{0};
};

inline bool IsImageLayout(DataLayout layout) {
    return layout == DataLayout::NCHW || layout == DataLayout::NHWC;
}

inline DataLayout NormalizeDataLayout(DataLayout layout) {
    return IsImageLayout(layout) ? layout : DataLayout::NCHW;
}

inline bool IsChannelLastLayout(DataLayout layout) {
    return NormalizeDataLayout(layout) == DataLayout::NHWC;
}

inline int ChannelAxisForLayout(DataLayout layout) {
    return IsChannelLastLayout(layout) ? 3 : 1;
}

inline int HeightAxisForLayout(DataLayout layout) {
    return IsChannelLastLayout(layout) ? 1 : 2;
}

inline int WidthAxisForLayout(DataLayout layout) {
    return IsChannelLastLayout(layout) ? 2 : 3;
}

inline bool DecodeImageShape4D(const std::vector<int64_t>& dims, DataLayout layout, ImageShape4D* shape) {
    if (shape == nullptr || dims.size() != 4) {
        return false;
    }
    const DataLayout normalized = NormalizeDataLayout(layout);
    shape->n = dims[0];
    shape->c = dims[static_cast<size_t>(ChannelAxisForLayout(normalized))];
    shape->h = dims[static_cast<size_t>(HeightAxisForLayout(normalized))];
    shape->w = dims[static_cast<size_t>(WidthAxisForLayout(normalized))];
    return true;
}

inline std::vector<int64_t> EncodeImageShape4D(const ImageShape4D& shape, DataLayout layout) {
    if (IsChannelLastLayout(layout)) {
        return {shape.n, shape.h, shape.w, shape.c};
    }
    return {shape.n, shape.c, shape.h, shape.w};
}

inline int64_t OffsetForImage4D(DataLayout layout, int64_t n, int64_t c, int64_t h, int64_t w,
                                int64_t channels, int64_t height, int64_t width) {
    if (IsChannelLastLayout(layout)) {
        return ((n * height + h) * width + w) * channels + c;
    }
    return ((n * channels + c) * height + h) * width + w;
}

inline size_t DataTypeBytes(DataType dtype) {
    switch (dtype) {
        case DataType::INT8:
        case DataType::UINT8:
        case DataType::BOOL:
        case DataType::FP8E4M3:
        case DataType::FP8E5M2:
            return 1;
        case DataType::FP16:
        case DataType::BF16:
            return 2;
        case DataType::FP32:
        case DataType::INT32:
            return 4;
        case DataType::INT64:
            return 8;
        default:
            return 0;
    }
}

}  // namespace feather

#endif  // FEATHER_INFER_UTIL_TYPES_H
