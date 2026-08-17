#ifndef FEATHER_KERNEL_WHERE_H
#define FEATHER_KERNEL_WHERE_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureCommonWhereKernelsRegistered();
void EnsureWhereKernelsRegistered();

template <DeviceType dev, DataType dtype>
class WhereKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_WHERE_H
