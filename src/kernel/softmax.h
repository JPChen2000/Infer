#ifndef FEATHER_KERNEL_SOFTMAX_H
#define FEATHER_KERNEL_SOFTMAX_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureX86SoftmaxKernelsRegistered();
void EnsureSoftmaxKernelsRegistered();

template <DeviceType dev, DataType dtype>
class SoftmaxKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_SOFTMAX_H
