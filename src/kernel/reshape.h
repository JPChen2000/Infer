#ifndef FEATHER_KERNEL_RESHAPE_H
#define FEATHER_KERNEL_RESHAPE_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureCommonReshapeKernelsRegistered();
void EnsureX86ReshapeKernelsRegistered();
void EnsureReshapeKernelsRegistered();

template <DeviceType dev, DataType dtype>
class ReshapeKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_RESHAPE_H
