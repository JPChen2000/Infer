#ifndef FEATHER_KERNEL_TANH_H
#define FEATHER_KERNEL_TANH_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureTanhKernelsRegistered();

template <DeviceType dev, DataType dtype>
class TanhKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_TANH_H
