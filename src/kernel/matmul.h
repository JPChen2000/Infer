#ifndef FEATHER_KERNEL_MATMUL_H
#define FEATHER_KERNEL_MATMUL_H

#include "core/kernel.h"
#include "src/operator/params.h"
#include "src/kernel/x86/linear.h"
#include "src/kernel/x86/linear_fp8.h"

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
    int32_t Prepare() override;
    int32_t compute() override;

   private:
    x86::PackedBf16Rhs packed_rhs_;
    x86::Bf16LinearWorkspace workspace_;
};

template <>
class MatMulKernel<DeviceType::X86, DataType::FP8E4M3> : public KernelBase {
   public:
    int32_t Prepare() override;
    int32_t compute() override;

   private:
    x86::PackedFp8Rhs packed_rhs_;
    x86::Fp8LinearWorkspace workspace_;
};

template <>
class MatMulKernel<DeviceType::X86, DataType::FP8E5M2> : public KernelBase {
   public:
    int32_t Prepare() override;
    int32_t compute() override;

   private:
    x86::PackedFp8Rhs packed_rhs_;
    x86::Fp8LinearWorkspace workspace_;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_MATMUL_H
