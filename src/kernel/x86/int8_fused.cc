#include "src/kernel/common/int8_fused.h"

#include <memory>

#include "src/kernel/resize_concat.h"
#include "src/kernel/shape.h"
#include "src/kernel/yolo_decode.h"

namespace feather {
namespace kernel {

namespace {



}  // namespace

template <>
int32_t ShapeKernel<DeviceType::X86, DataType::INT8>::compute() {
    return common::ComputeShapeInt8(ParamAs<feather::operators::ShapeParam>());
}

template <>
int32_t ResizeConcatKernel<DeviceType::X86, DataType::INT8>::compute() {
    return common::ComputeResizeConcatInt8(ParamAs<feather::operators::ResizeConcatParam>());
}

template <>
int32_t YoloDecodeKernel<DeviceType::X86, DataType::INT8>::compute() {
    return common::ComputeYoloDecodeInt8(ParamAs<feather::operators::YoloDecodeParam>());
}



}  // namespace kernel
}  // namespace feather

