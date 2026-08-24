#ifndef FEATHER_KERNEL_QWEN_DEPTHWISE_CONV_H
#define FEATHER_KERNEL_QWEN_DEPTHWISE_CONV_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureX86QwenDepthwiseConvStateKernelsRegistered();
#ifdef FEATHER_WITH_CUDA
void EnsureCudaQwenDepthwiseConvStateKernelsRegistered();
#endif
void EnsureQwenDepthwiseConvStateKernelsRegistered();

template <DeviceType dev, DataType dtype>
class QwenDepthwiseConvStateKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_QWEN_DEPTHWISE_CONV_H
