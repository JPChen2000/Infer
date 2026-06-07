#ifndef FEATHER_KERNEL_ADD_H
#define FEATHER_KERNEL_ADD_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureCommonAddKernelsRegistered();
void EnsureX86AddKernelsRegistered();
void EnsureAddKernelsRegistered();

template <DeviceType dev, DataType dtype>
class AddKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_ADD_H
