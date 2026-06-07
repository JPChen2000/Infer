#ifndef FEATHER_KERNEL_SIGMOID_H
#define FEATHER_KERNEL_SIGMOID_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureCommonSigmoidKernelsRegistered();
void EnsureX86SigmoidKernelsRegistered();
void EnsureSigmoidKernelsRegistered();

template <DeviceType dev, DataType dtype>
class SigmoidKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_SIGMOID_H
