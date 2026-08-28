#ifndef FEATHER_KERNEL_CUDA_RUNTIME_H
#define FEATHER_KERNEL_CUDA_RUNTIME_H

#include <cublas_v2.h>
#ifdef FEATHER_WITH_CUBLASLT
#include <cublasLt.h>
#endif
#include <cuda_runtime.h>
#ifdef FEATHER_WITH_CUDNN
#include <cudnn.h>
#endif

#include <cstddef>
#include <cstdint>
#include <string>

#include "core/tensor.h"

namespace feather {
namespace kernel {
namespace cuda_detail {

inline int64_t DivUp(int64_t value, int64_t divisor) {
    if (value <= 0 || divisor <= 0) {
        return 0;
    }
    return value / divisor + (value % divisor == 0 ? 0 : 1);
}

struct TensorCacheStats {
    size_t active_tensor_count{0};
    size_t persistent_tensor_count{0};
    size_t free_block_count{0};
    size_t active_bytes{0};
    size_t pooled_bytes{0};
};

enum class CudaFp8MatmulBackend {
    kUnknown = 0,
    kCublasLt,
    kBf16TensorCoreFallback,
    kScalarFallback,
};

void SetDeferredHostSync(bool enabled);
bool DeferredHostSyncEnabled();
int WarmupCudaRuntime();
std::string CudaLastErrorMessage();
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
int AliasTensorDeviceStorage(const Tensor* input, Tensor* output, size_t bytes);
int SwapTensorDeviceStorage(Tensor* input, Tensor* output);
int AppendTensorStateOnDevice(Tensor* input, const Tensor* output, int axis);
cublasHandle_t CublasHandle();
#ifdef FEATHER_WITH_CUBLASLT
cublasLtHandle_t CublasLtHandle();
#endif
void ResetLastCudaFp8MatmulBackend();
void SetLastCudaFp8MatmulBackend(CudaFp8MatmulBackend backend);
CudaFp8MatmulBackend LastCudaFp8MatmulBackend();
#ifdef FEATHER_WITH_CUDNN
cudnnHandle_t CudnnHandle();
#endif

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
