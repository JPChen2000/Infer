#ifndef FEATHER_KERNEL_QWEN_GEMM_ARGMAX_H
#define FEATHER_KERNEL_QWEN_GEMM_ARGMAX_H

#include "core/kernel.h"
#include "src/operator/params.h"
#include "src/kernel/x86/linear.h"

namespace feather {
namespace kernel {

void EnsureCommonQwenGemmArgmaxKernelsRegistered();
void EnsureX86QwenGemmArgmaxKernelsRegistered();
#ifdef FEATHER_WITH_CUDA
void EnsureCudaQwenGemmArgmaxKernelsRegistered();
#endif
void EnsureQwenGemmArgmaxKernelsRegistered();

#ifdef FEATHER_WITH_CUDA
enum class CudaQwenGemmArgmaxBackend {
    kUnknown = 0,
    kFusedGemv,
    kCublasFallback,
};

void ResetLastCudaQwenGemmArgmaxBackend();
CudaQwenGemmArgmaxBackend LastCudaQwenGemmArgmaxBackend();
#endif

template <DeviceType dev, DataType dtype>
class QwenGemmArgmaxKernel : public KernelBase {
   public:
    int32_t compute() override;
};

template <>
class QwenGemmArgmaxKernel<DeviceType::X86, DataType::BF16> : public KernelBase {
   public:
    int32_t Prepare() override;
    int32_t compute() override;

   private:
    x86::PackedBf16TransposedRhs packed_rhs_;
    x86::Bf16LinearWorkspace workspace_;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_QWEN_GEMM_ARGMAX_H
