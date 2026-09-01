#ifndef FEATHER_KERNEL_COMMON_INT8_FUSED_H
#define FEATHER_KERNEL_COMMON_INT8_FUSED_H

#include <cstdint>
#include <memory>

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

struct Int8Quantization {
    float scale{1.0f};
    int32_t zero_point{0};
};

namespace common {

int32_t ComputeShapeInt8(feather::operators::ShapeParam* param);
int32_t ComputeResizeConcatInt8(feather::operators::ResizeConcatParam* param);
int32_t ComputeYoloDecodeInt8(feather::operators::YoloDecodeParam* param);
Int8Quantization GetInt8Quantization(const std::shared_ptr<Tensor>& tensor);

}  // namespace common

void EnsureCommonInt8FusedKernelsRegistered();
void EnsureX86Int8FusedKernelsRegistered();
void EnsureCudaInt8FusedKernelsRegistered();

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_COMMON_INT8_FUSED_H
