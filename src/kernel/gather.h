#ifndef FEATHER_KERNEL_GATHER_H
#define FEATHER_KERNEL_GATHER_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureCommonGatherKernelsRegistered();
void EnsureGatherKernelsRegistered();

template <DeviceType dev, DataType dtype>
class GatherKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_GATHER_H
