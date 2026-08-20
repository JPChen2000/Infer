#ifndef FEATHER_KERNEL_DIV_H
#define FEATHER_KERNEL_DIV_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureDivKernelsRegistered();
void EnsureX86DivKernelsRegistered();

template <DeviceType dev, DataType dtype>
class DivKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_DIV_H
