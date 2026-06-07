#ifndef FEATHER_KERNEL_FLATTEN_H
#define FEATHER_KERNEL_FLATTEN_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureFlattenKernelsRegistered();

template <DeviceType dev, DataType dtype>
class FlattenKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_FLATTEN_H
