#ifndef FEATHER_KERNEL_MATMUL_H
#define FEATHER_KERNEL_MATMUL_H

#include "core/kernel.h"
#include "src/operator/params.h"
#include "src/kernel/x86/linear.h"

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

template <>
class MatMulKernel<DeviceType::X86, DataType::BF16> : public KernelBase {
   public:
    int32_t compute() override;

   private:
    x86::Bf16LinearWorkspace workspace_;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_MATMUL_H
