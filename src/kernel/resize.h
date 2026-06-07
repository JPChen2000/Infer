#ifndef FEATHER_KERNEL_RESIZE_H
#define FEATHER_KERNEL_RESIZE_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureCommonResizeKernelsRegistered();
void EnsureX86ResizeKernelsRegistered();
void EnsureResizeKernelsRegistered();

template <DeviceType dev, DataType dtype>
class ResizeKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_RESIZE_H
