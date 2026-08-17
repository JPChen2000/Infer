#ifndef FEATHER_KERNEL_REDUCE_MEAN_H
#define FEATHER_KERNEL_REDUCE_MEAN_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureCommonReduceMeanKernelsRegistered();
void EnsureReduceMeanKernelsRegistered();

template <DeviceType dev, DataType dtype>
class ReduceMeanKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_REDUCE_MEAN_H
