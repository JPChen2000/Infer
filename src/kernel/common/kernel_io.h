#ifndef FEATHER_KERNEL_COMMON_KERNEL_IO_H
#define FEATHER_KERNEL_COMMON_KERNEL_IO_H

#include <cstdint>

#include "core/tensor.h"
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

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_COMMON_KERNEL_IO_H
