#ifndef FEATHER_KERNEL_EXPAND_H
#define FEATHER_KERNEL_EXPAND_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureCommonExpandKernelsRegistered();
void EnsureX86ExpandKernelsRegistered();
void EnsureExpandKernelsRegistered();

template <DeviceType dev, DataType dtype>
class ExpandKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_EXPAND_H
