#ifndef FEATHER_KERNEL_FC_COMPUTE_H
#define FEATHER_KERNEL_FC_COMPUTE_H
#include "core/kernel.h"
#include "core/tensor.h"
#include "src/kernel/x86/linear_fp8.h"
#include "util/logger.h"
using feather::Tensor;
namespace feather {
namespace kernel {

void EnsureCommonFcKernelsRegistered();
void EnsureFcKernelsRegistered();
void EnsureX86FcKernelsRegistered();

template <DeviceType dev, DataType dtype>
class FcKernel : public KernelBase {
   public:
    int32_t compute();
};

template <>
class FcKernel<DeviceType::X86, DataType::FP8E4M3> : public KernelBase {
   public:
    int32_t Prepare() override;
    int32_t compute() override;

   private:
    x86::PackedFp8Rhs packed_rhs_;
    x86::Fp8LinearWorkspace workspace_;
};

template <>
class FcKernel<DeviceType::X86, DataType::FP8E5M2> : public KernelBase {
   public:
    int32_t Prepare() override;
    int32_t compute() override;

   private:
    x86::PackedFp8Rhs packed_rhs_;
    x86::Fp8LinearWorkspace workspace_;
};
}  // namespace kernel
}  // namespace feather
// template class feather::kernel::FcCUDAKernel<DataType::FP32>;
#endif  // FEATHER_KERNEL_FC_COMPUTE_H
