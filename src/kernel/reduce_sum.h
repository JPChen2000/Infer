#ifndef FEATHER_KERNEL_REDUCE_SUM_H
#define FEATHER_KERNEL_REDUCE_SUM_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureReduceSumKernelsRegistered();

template <DeviceType dev, DataType dtype>
class ReduceSumKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_REDUCE_SUM_H
