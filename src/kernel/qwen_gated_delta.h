#ifndef FEATHER_KERNEL_QWEN_GATED_DELTA_H
#define FEATHER_KERNEL_QWEN_GATED_DELTA_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureCommonQwenGatedDeltaKernelsRegistered();
void EnsureX86QwenGatedDeltaKernelsRegistered();
void EnsureQwenGatedDeltaKernelsRegistered();

template <DeviceType dev, DataType dtype>
class QwenGatedDeltaStateKernel : public KernelBase {
   public:
    int32_t compute() override;
};

template <DeviceType dev, DataType dtype>
class QwenGatedDeltaOutputKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_QWEN_GATED_DELTA_H
