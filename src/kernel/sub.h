#ifndef FEATHER_KERNEL_SUB_H
#define FEATHER_KERNEL_SUB_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureSubKernelsRegistered();
void EnsureX86SubKernelsRegistered();

template <DeviceType dev, DataType dtype>
class SubKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_SUB_H
