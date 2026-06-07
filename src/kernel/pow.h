#ifndef FEATHER_KERNEL_POW_H
#define FEATHER_KERNEL_POW_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureCommonPowKernelsRegistered();
void EnsureX86PowKernelsRegistered();
void EnsurePowKernelsRegistered();

template <DeviceType dev, DataType dtype>
class PowKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_POW_H
