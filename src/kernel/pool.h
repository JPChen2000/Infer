#ifndef FEATHER_KERNEL_POOL_H
#define FEATHER_KERNEL_POOL_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureCommonPoolKernelsRegistered();
void EnsureX86PoolKernelsRegistered();
void EnsurePoolKernelsRegistered();

template <DeviceType dev, DataType dtype>
class AvgPoolKernel : public KernelBase {
   public:
    int32_t compute() override;
};

template <DeviceType dev, DataType dtype>
class MaxPoolKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_POOL_H
