#ifndef FEATHER_KERNEL_YOLO_DECODE_H
#define FEATHER_KERNEL_YOLO_DECODE_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureCommonYoloDecodeKernelsRegistered();
void EnsureYoloDecodeKernelsRegistered();

template <DeviceType dev, DataType dtype>
class YoloDecodeKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_YOLO_DECODE_H
