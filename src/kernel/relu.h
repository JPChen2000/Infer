#ifndef FEATHER_KERNEL_RELU_H
#define FEATHER_KERNEL_RELU_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureCommonReluKernelsRegistered();
void EnsureX86ReluKernelsRegistered();
void EnsureReluKernelsRegistered();

template <DeviceType dev, DataType dtype>
class ReluKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_RELU_H
