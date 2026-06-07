#ifndef FEATHER_KERNEL_SPLIT_H
#define FEATHER_KERNEL_SPLIT_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureCommonSplitKernelsRegistered();
void EnsureX86SplitKernelsRegistered();
void EnsureSplitKernelsRegistered();

template <DeviceType dev, DataType dtype>
class SplitKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_SPLIT_H
