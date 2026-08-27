#ifndef FEATHER_KERNEL_SQRT_H
#define FEATHER_KERNEL_SQRT_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureSqrtKernelsRegistered();
void EnsureX86SqrtKernelsRegistered();

template <DeviceType dev, DataType dtype>
class SqrtKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_SQRT_H
