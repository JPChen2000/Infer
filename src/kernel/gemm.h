#ifndef FEATHER_KERNEL_GEMM_H
#define FEATHER_KERNEL_GEMM_H

#include "core/kernel.h"

namespace feather {
namespace kernel {

void EnsureCommonGemmKernelsRegistered();
void EnsureX86GemmKernelsRegistered();
void EnsureGemmKernelsRegistered();

template <DeviceType dev, DataType dtype>
class GemmKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_GEMM_H
