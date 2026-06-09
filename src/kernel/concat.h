#ifndef FEATHER_KERNEL_CONCAT_H
#define FEATHER_KERNEL_CONCAT_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

enum class CudaConcatBackend {
    kUnknown,
    kKernel,
    kMemcpy2D,
};

namespace detail {
inline thread_local CudaConcatBackend g_last_cuda_concat_backend = CudaConcatBackend::kUnknown;
}

inline CudaConcatBackend LastCudaConcatBackend() { return detail::g_last_cuda_concat_backend; }
inline void ResetLastCudaConcatBackend() { detail::g_last_cuda_concat_backend = CudaConcatBackend::kUnknown; }
inline void SetLastCudaConcatBackend(CudaConcatBackend backend) { detail::g_last_cuda_concat_backend = backend; }

void EnsureCommonConcatKernelsRegistered();
void EnsureX86ConcatKernelsRegistered();
void EnsureConcatKernelsRegistered();

template <DeviceType dev, DataType dtype>
class ConcatKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_CONCAT_H
