#ifndef FEATHER_KERNEL_SOFTPLUS_H
#define FEATHER_KERNEL_SOFTPLUS_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureSoftplusKernelsRegistered();

template <DeviceType dev, DataType dtype>
class SoftplusKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_SOFTPLUS_H
