#ifndef FEATHER_KERNEL_QWEN_RMS_NORM_H
#define FEATHER_KERNEL_QWEN_RMS_NORM_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureCommonQwenRmsNormKernelsRegistered();
void EnsureX86QwenRmsNormKernelsRegistered();
#ifdef FEATHER_WITH_CUDA
void EnsureCudaQwenRmsNormKernelsRegistered();
#endif
void EnsureQwenRmsNormKernelsRegistered();

template <DeviceType dev, DataType dtype>
class QwenRmsNormKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_QWEN_RMS_NORM_H
