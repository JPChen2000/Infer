#ifndef FEATHER_KERNEL_SILU_H
#define FEATHER_KERNEL_SILU_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureCommonSiluKernelsRegistered();
void EnsureX86SiluKernelsRegistered();
void EnsureSiluKernelsRegistered();

template <DeviceType dev, DataType dtype>
class SiluKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_SILU_H
