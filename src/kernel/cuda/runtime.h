#ifndef FEATHER_KERNEL_CUDA_RUNTIME_H
#define FEATHER_KERNEL_CUDA_RUNTIME_H

#include <cuda_runtime.h>

#include <cstddef>

#include "core/tensor.h"

namespace feather {
namespace kernel {
namespace cuda_detail {

void SetDeferredHostSync(bool enabled);
bool DeferredHostSyncEnabled();
int WarmupCudaRuntime();
cudaStream_t InferenceStream();
int SynchronizeInferenceStream();
void InvalidateTensorDevice(const Tensor* tensor);
void ClearTensorCache();
int AcquireTensorDevice(const Tensor* tensor, size_t bytes, const void* host_data, void** device_ptr);
int AcquireOutputTensorDevice(Tensor* tensor, size_t bytes, void** device_ptr);
int SyncTensorToHost(Tensor* tensor, size_t bytes, void* host_data);
int SyncTensorToHostIfNeeded(Tensor* tensor, size_t bytes, void* host_data);

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
