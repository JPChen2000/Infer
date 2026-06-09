#ifndef FEATHER_KERNEL_CUDA_RUNTIME_H
#define FEATHER_KERNEL_CUDA_RUNTIME_H

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>

#include "core/tensor.h"

namespace feather {
namespace kernel {
namespace cuda_detail {

struct TensorCacheStats {
    size_t active_tensor_count{0};
    size_t persistent_tensor_count{0};
    size_t free_block_count{0};
    size_t active_bytes{0};
    size_t pooled_bytes{0};
};

void SetDeferredHostSync(bool enabled);
bool DeferredHostSyncEnabled();
int WarmupCudaRuntime();
cudaStream_t InferenceStream();
int SynchronizeInferenceStream();
void InvalidateTensorDevice(const Tensor* tensor);
void MarkTensorDevicePersistent(const Tensor* tensor, bool persistent);
bool IsTensorDevicePersistent(const Tensor* tensor);
void ReleaseTensorDevice(const Tensor* tensor);
int AcquireTemporaryDeviceBuffer(size_t bytes, void** device_ptr);
void ReleaseTemporaryDeviceBuffer(void* device_ptr, size_t bytes);
TensorCacheStats GetTensorCacheStats();
void ClearTensorCache();
int AcquireTensorDevice(const Tensor* tensor, size_t bytes, const void* host_data, void** device_ptr);
int AcquireOutputTensorDevice(Tensor* tensor, size_t bytes, void** device_ptr);
int SyncTensorToHost(Tensor* tensor, size_t bytes, void* host_data);
int SyncTensorToHostIfNeeded(Tensor* tensor, size_t bytes, void* host_data);
cublasHandle_t CublasHandle();

class DeferredHostSyncScope {
   public:
    DeferredHostSyncScope();
    DeferredHostSyncScope(const DeferredHostSyncScope&) = delete;
    DeferredHostSyncScope& operator=(const DeferredHostSyncScope&) = delete;
    ~DeferredHostSyncScope();

   private:
    bool previous_{false};
};

}  // namespace cuda_detail
}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_CUDA_RUNTIME_H
