#include "src/kernel/common/int8_fused.h"

#include <memory>

#include "src/kernel/resize_concat.h"
#include "src/kernel/shape.h"
#include "src/kernel/yolo_decode.h"

namespace feather {
namespace kernel {

namespace {

bool g_x86_int8_fused_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::X86, DataType::INT8, "Shape", []() {
        return std::make_unique<ShapeKernel<DeviceType::X86, DataType::INT8>>();
    });
    dispatcher.registerKernel(DeviceType::X86, DataType::INT8, "ResizeConcat", []() {
        return std::make_unique<ResizeConcatKernel<DeviceType::X86, DataType::INT8>>();
    });
    dispatcher.registerKernel(DeviceType::X86, DataType::INT8, "YoloDecode", []() {
        return std::make_unique<YoloDecodeKernel<DeviceType::X86, DataType::INT8>>();
    });
    return true;
}();

}  // namespace

template <>
int32_t ShapeKernel<DeviceType::X86, DataType::INT8>::compute() {
    return common::ComputeShapeInt8(static_cast<feather::operators::ShapeParam*>(param_));
}

template <>
int32_t ResizeConcatKernel<DeviceType::X86, DataType::INT8>::compute() {
    return common::ComputeResizeConcatInt8(static_cast<feather::operators::ResizeConcatParam*>(param_));
}

template <>
int32_t YoloDecodeKernel<DeviceType::X86, DataType::INT8>::compute() {
    return common::ComputeYoloDecodeInt8(static_cast<feather::operators::YoloDecodeParam*>(param_));
}

void EnsureX86Int8FusedKernelsRegistered() {
    (void)g_x86_int8_fused_kernels_registered;
}

}  // namespace kernel
}  // namespace feather

