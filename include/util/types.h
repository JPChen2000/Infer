#ifndef FEATHER_INFER_UTIL_TYPES_H
#define FEATHER_INFER_UTIL_TYPES_H
#include <cstddef>
#include <iostream>
#include <unordered_map>
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

inline size_t DataTypeBytes(DataType dtype) {
    switch (dtype) {
        case DataType::INT8:
        case DataType::UINT8:
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
