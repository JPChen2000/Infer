#ifndef FEATHER_KERNEL_MATMUL_H
#define FEATHER_KERNEL_MATMUL_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureCommonMatMulKernelsRegistered();
void EnsureX86MatMulKernelsRegistered();
void EnsureMatMulKernelsRegistered();

template <DeviceType dev, DataType dtype>
class MatMulKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_MATMUL_H
