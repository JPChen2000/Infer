#ifndef FEATHER_KERNEL_COS_H
#define FEATHER_KERNEL_COS_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureCosKernelsRegistered();

template <DeviceType dev, DataType dtype>
class CosKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_COS_H
