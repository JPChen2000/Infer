#ifndef FEATHER_INFER_UTIL_TYPES_H
#define FEATHER_INFER_UTIL_TYPES_H
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <variant>
namespace feather {

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
    UNKNOWN
};

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


enum class OpType {
    Conv2D = 0,
    Pool = 1,
    Concat = 2,
    ReLU = 3,
    Gemm = 4,
    Upsample = 5,
    Reshape = 6,
    Split = 7,
    Sigmoid = 8,
    Softmax = 9,
    AvgPool = 10,
    OP_NUMS
};

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
            return 1;
        case DataType::FP16:
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
