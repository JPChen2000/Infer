#ifndef FEATHER_KERNEL_IDENTITY_H
#define FEATHER_KERNEL_IDENTITY_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureIdentityKernelsRegistered();

template <DeviceType dev, DataType dtype>
class IdentityKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_IDENTITY_H
