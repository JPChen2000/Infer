#ifndef FEATHER_KERNEL_EXP_H
#define FEATHER_KERNEL_EXP_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureExpKernelsRegistered();
void EnsureX86ExpKernelsRegistered();

template <DeviceType dev, DataType dtype>
class ExpKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_EXP_H
