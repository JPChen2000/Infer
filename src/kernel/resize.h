#ifndef FEATHER_KERNEL_RESIZE_H
#define FEATHER_KERNEL_RESIZE_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

enum class CudaResizeBackend {
    kUnknown,
    kGeneric,
    kDirect4D,
};

namespace detail {
inline thread_local CudaResizeBackend g_last_cuda_resize_backend = CudaResizeBackend::kUnknown;
}

inline CudaResizeBackend LastCudaResizeBackend() { return detail::g_last_cuda_resize_backend; }
inline void ResetLastCudaResizeBackend() { detail::g_last_cuda_resize_backend = CudaResizeBackend::kUnknown; }
inline void SetLastCudaResizeBackend(CudaResizeBackend backend) { detail::g_last_cuda_resize_backend = backend; }

void EnsureCommonResizeKernelsRegistered();
void EnsureX86ResizeKernelsRegistered();
void EnsureResizeKernelsRegistered();

template <DeviceType dev, DataType dtype>
class ResizeKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_RESIZE_H
