#ifndef FEATHER_KERNEL_SLICE_H
#define FEATHER_KERNEL_SLICE_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureX86SliceKernelsRegistered();
void EnsureSliceKernelsRegistered();

template <DeviceType dev, DataType dtype>
class SliceKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_SLICE_H
