#ifndef FEATHER_KERNEL_UNSQUEEZE_H
#define FEATHER_KERNEL_UNSQUEEZE_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureCommonUnsqueezeKernelsRegistered();
void EnsureUnsqueezeKernelsRegistered();

template <DeviceType dev, DataType dtype>
class UnsqueezeKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_UNSQUEEZE_H
