#ifndef FEATHER_KERNEL_SILU_H
#define FEATHER_KERNEL_SILU_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

enum class CudaSiluBackend {
    kUnknown,
    kFallback,
    kDirect,
    kCudnn,
};

namespace detail {
inline thread_local CudaSiluBackend g_last_cuda_silu_backend = CudaSiluBackend::kUnknown;
}

inline CudaSiluBackend LastCudaSiluBackend() { return detail::g_last_cuda_silu_backend; }
inline void ResetLastCudaSiluBackend() { detail::g_last_cuda_silu_backend = CudaSiluBackend::kUnknown; }
inline void SetLastCudaSiluBackend(CudaSiluBackend backend) { detail::g_last_cuda_silu_backend = backend; }

void EnsureCommonSiluKernelsRegistered();
void EnsureX86SiluKernelsRegistered();
void EnsureSiluKernelsRegistered();

template <DeviceType dev, DataType dtype>
class SiluKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_SILU_H
