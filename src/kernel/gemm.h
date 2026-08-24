#ifndef FEATHER_KERNEL_GEMM_H
#define FEATHER_KERNEL_GEMM_H

#include "core/kernel.h"
#include "src/kernel/x86/linear.h"

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

template <>
class GemmKernel<DeviceType::X86, DataType::BF16> : public KernelBase {
   public:
    int32_t Prepare() override;
   int32_t compute() override;

   private:
    x86::PackedBf16Rhs packed_rhs_;
    x86::PackedBf16TransposedRhs packed_transposed_rhs_;
    x86::Bf16LinearWorkspace workspace_;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_GEMM_H
