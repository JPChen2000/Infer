#ifndef FEATHER_KERNEL_TRANSPOSE_H
#define FEATHER_KERNEL_TRANSPOSE_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureCommonTransposeKernelsRegistered();
void EnsureX86TransposeKernelsRegistered();
void EnsureTransposeKernelsRegistered();

template <DeviceType dev, DataType dtype>
class TransposeKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_TRANSPOSE_H
