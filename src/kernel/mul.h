#ifndef FEATHER_KERNEL_MUL_H
#define FEATHER_KERNEL_MUL_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureCommonMulKernelsRegistered();
void EnsureX86MulKernelsRegistered();
void EnsureMulKernelsRegistered();

template <DeviceType dev, DataType dtype>
class MulKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_MUL_H
