#ifndef FEATHER_KERNEL_RESIZE_CONCAT_H
#define FEATHER_KERNEL_RESIZE_CONCAT_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureResizeConcatKernelsRegistered();

template <DeviceType dev, DataType dtype>
class ResizeConcatKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_RESIZE_CONCAT_H
