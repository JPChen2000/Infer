#ifndef FEATHER_KERNEL_POOL_H
#define FEATHER_KERNEL_POOL_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

enum class CudaPoolBackend {
    kUnknown,
    kFallback,
    kCudnn,
};

namespace detail {
inline thread_local CudaPoolBackend g_last_cuda_pool_backend = CudaPoolBackend::kUnknown;
}

inline CudaPoolBackend LastCudaPoolBackend() { return detail::g_last_cuda_pool_backend; }
inline void ResetLastCudaPoolBackend() { detail::g_last_cuda_pool_backend = CudaPoolBackend::kUnknown; }
inline void SetLastCudaPoolBackend(CudaPoolBackend backend) { detail::g_last_cuda_pool_backend = backend; }

void EnsureCommonPoolKernelsRegistered();
void EnsureX86PoolKernelsRegistered();
void EnsurePoolKernelsRegistered();

template <DeviceType dev, DataType dtype>
class AvgPoolKernel : public KernelBase {
   public:
    int32_t compute() override;
};

template <DeviceType dev, DataType dtype>
class MaxPoolKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_POOL_H
