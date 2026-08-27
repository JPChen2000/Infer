#ifndef FEATHER_KERNEL_COMMON_KERNEL_IO_H
#define FEATHER_KERNEL_COMMON_KERNEL_IO_H

#include <cmath>
#include <cstdint>

#include "core/tensor.h"
#include "util/bf16.h"
#include "util/fp8.h"
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

template <>
struct TensorIO<DataType::FP8E4M3> {
    static float Read(const Tensor* tensor, int64_t index) {
        return Fp8E4M3ToFloat(tensor->data<Fp8E4M3>()[index].bits) * tensor->quantization_scale();
    }
    static void Write(Tensor* tensor, int64_t index, float value) {
        tensor->mutable_data<Fp8E4M3>()[index].bits =
            FloatToFp8E4M3(value / tensor->quantization_scale());
    }
};

template <>
struct TensorIO<DataType::FP8E5M2> {
    static float Read(const Tensor* tensor, int64_t index) {
        return Fp8E5M2ToFloat(tensor->data<Fp8E5M2>()[index].bits) * tensor->quantization_scale();
    }
    static void Write(Tensor* tensor, int64_t index, float value) {
        tensor->mutable_data<Fp8E5M2>()[index].bits =
            FloatToFp8E5M2(value / tensor->quantization_scale());
    }
};

inline bool IsReadableScalarFloatTensor(const Tensor* tensor) {
    if (tensor == nullptr || !tensor->IsInitialized() || tensor->numel() != 1) {
        return false;
    }
    const size_t element_bytes = DataTypeBytes(tensor->data_type());
    if (element_bytes == 0 || tensor->memory_size() < element_bytes) return false;
    switch (tensor->data_type()) {
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
        case DataType::INT32:
        case DataType::INT64:
            return true;
        case DataType::FP8E4M3:
        case DataType::FP8E5M2:
            return HasCompatiblePerTensorQuantization(tensor->quantization()) &&
                   std::isfinite(tensor->quantization_scale()) && tensor->quantization_scale() > 0.0f;
        default:
            return false;
    }
}

inline bool ReadScalarFloatTensor(const Tensor* tensor, float* value) {
    if (value == nullptr || !IsReadableScalarFloatTensor(tensor)) return false;
    switch (tensor->data_type()) {
        case DataType::FP16:
            *value = HalfToFloat(tensor->data<uint16_t>()[0]);
            return true;
        case DataType::BF16:
            *value = BFloat16ToFloat(tensor->data<BFloat16>()[0].bits);
            return true;
        case DataType::FP8E4M3:
            *value = TensorIO<DataType::FP8E4M3>::Read(tensor, 0);
            return true;
        case DataType::FP8E5M2:
            *value = TensorIO<DataType::FP8E5M2>::Read(tensor, 0);
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
