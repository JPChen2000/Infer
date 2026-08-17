#ifndef FEATHER_KERNEL_BATCH_NORMALIZATION_H
#define FEATHER_KERNEL_BATCH_NORMALIZATION_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureCommonBatchNormalizationKernelsRegistered();
void EnsureBatchNormalizationKernelsRegistered();

template <DeviceType dev, DataType dtype>
class BatchNormalizationKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_BATCH_NORMALIZATION_H
