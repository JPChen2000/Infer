#ifndef FEATHER_KERNEL_COMMON_KERNEL_IO_H
#define FEATHER_KERNEL_COMMON_KERNEL_IO_H

#include <cstdint>

#include "core/tensor.h"
#include "util/bf16.h"
#include "util/fp16.h"
#include "util/types.h"

namespace feather {
namespace kernel {

template <DataType dtype>
struct TensorIO;

template <>
struct TensorIO<DataType::FP32> {
    static float Read(const Tensor* tensor, int64_t index) { return tensor->data<float>()[index]; }
    static void Write(Tensor* tensor, int64_t index, float value) { tensor->mutable_data<float>()[index] = value; }
};

template <>
struct TensorIO<DataType::FP16> {
    static float Read(const Tensor* tensor, int64_t index) { return HalfToFloat(tensor->data<uint16_t>()[index]); }
    static void Write(Tensor* tensor, int64_t index, float value) {
        tensor->mutable_data<uint16_t>()[index] = FloatToHalf(value);
    }
};

template <>
struct TensorIO<DataType::BF16> {
    static float Read(const Tensor* tensor, int64_t index) {
        return BFloat16ToFloat(tensor->data<BFloat16>()[index].bits);
    }
    static void Write(Tensor* tensor, int64_t index, float value) {
        tensor->mutable_data<BFloat16>()[index].bits = FloatToBFloat16(value);
    }
};

inline bool ReadScalarFloatTensor(const Tensor* tensor, float* value) {
    if (tensor == nullptr || value == nullptr || !tensor->IsInitialized() || tensor->numel() != 1) {
        return false;
    }
    switch (tensor->data_type()) {
        case DataType::FP16:
            *value = HalfToFloat(tensor->data<uint16_t>()[0]);
            return true;
        case DataType::BF16:
            *value = BFloat16ToFloat(tensor->data<BFloat16>()[0].bits);
            return true;
        case DataType::FP32:
            *value = tensor->data<float>()[0];
            return true;
        case DataType::INT32:
            *value = static_cast<float>(tensor->data<int32_t>()[0]);
            return true;
        case DataType::INT64:
            *value = static_cast<float>(tensor->data<int64_t>()[0]);
            return true;
        default:
            return false;
    }
}

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_COMMON_KERNEL_IO_H
