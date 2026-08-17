#ifndef FEATHER_KERNEL_CAST_H
#define FEATHER_KERNEL_CAST_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureCommonCastKernelsRegistered();
void EnsureCastKernelsRegistered();

template <DeviceType dev, DataType dtype>
class CastKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_CAST_H
