#ifndef FEATHER_KERNEL_GLOBAL_AVERAGE_POOL_H
#define FEATHER_KERNEL_GLOBAL_AVERAGE_POOL_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureCommonGlobalAveragePoolKernelsRegistered();
void EnsureGlobalAveragePoolKernelsRegistered();

template <DeviceType dev, DataType dtype>
class GlobalAveragePoolKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_GLOBAL_AVERAGE_POOL_H
