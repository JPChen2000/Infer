#ifndef FEATHER_KERNEL_EQUAL_H
#define FEATHER_KERNEL_EQUAL_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureCommonEqualKernelsRegistered();
void EnsureEqualKernelsRegistered();

template <DeviceType dev, DataType dtype>
class EqualKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_EQUAL_H
