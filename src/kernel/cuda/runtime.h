// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
#include <memory>
#include <vector>

#include "core/tensor.h"

namespace feather {
namespace kernel {
namespace cuda_detail {

// A graph/session owns its device resources. Scopes propagate the active
// context through existing kernel APIs, including graph worker threads.
class CudaExecutionContext;
std::shared_ptr<CudaExecutionContext> CreateExecutionContext();

class ExecutionContextScope {
   public:
    explicit ExecutionContextScope(std::shared_ptr<CudaExecutionContext> context);
    ~ExecutionContextScope();
    ExecutionContextScope(const ExecutionContextScope&) = delete;
    ExecutionContextScope& operator=(const ExecutionContextScope&) = delete;
   private:
    std::shared_ptr<CudaExecutionContext> previous_;
    int previous_device_{-1};
};

// Records allocator use without changing reuse during stream capture. A
// retained lease reserves the captured addresses until its graph is destroyed.
class CaptureAllocationSnapshot;
class CaptureAllocationScope {
 public:
    CaptureAllocationScope();
    ~CaptureAllocationScope();
    CaptureAllocationScope(const CaptureAllocationScope&) = delete;
    CaptureAllocationScope& operator=(const CaptureAllocationScope&) = delete;
    bool valid() const { return recording_; }
    std::shared_ptr<void> Retain();
 private:
    std::shared_ptr<CudaExecutionContext> context_;
    std::unique_ptr<CaptureAllocationSnapshot> snapshot_;
    bool recording_{false};
};

inline int64_t DivUp(int64_t value, int64_t divisor) {
    if (value <= 0 || divisor <= 0) {
        return 0;
    }
    return value / divisor + (value % divisor == 0 ? 0 : 1);
}

struct TensorCacheStats {
    // Counts represent unique Storage/offset/byte-range replicas. Multiple
    // registered Tensor handles for an equal view share one replica.
    size_t active_tensor_count{0};
    size_t persistent_tensor_count{0};
    size_t free_block_count{0};
    size_t active_bytes{0};
    size_t pooled_bytes{0};
    size_t fp8_weight_cache_count{0};
    size_t fp8_weight_cache_bytes{0};
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
// Tensor addresses are handle registrations only; device state is keyed by
// stable Storage identity and view byte range. Shared handles remain registered
// until their final graph/session user unregisters.
void RegisterTensorDeviceUser(const Tensor* tensor);
void UnregisterTensorDeviceUsers(const std::vector<const Tensor*>& tensors);
std::shared_ptr<const Storage> GetTensorDeviceStorage(const Tensor* tensor);
int AcquireTemporaryDeviceBuffer(size_t bytes, void** device_ptr);
void SetWorkspaceLimit(size_t bytes);
// Optional resident BF16 copies of immutable FP8 weights. Zero disables caching.
void SetFp8WeightCacheLimit(size_t bytes);
// Returns 0 on a ready cache hit/build, 1 when the caller should use scratch.
int TryAcquireFp8Bf16Weight(const Tensor* tensor, const void* source, size_t count, float scale,
                            const void** prepared);
void ReleaseTemporaryDeviceBuffer(void* device_ptr, size_t bytes);
TensorCacheStats GetTensorCacheStats();
void ClearTensorCache();
int AcquireTensorDevice(const Tensor* tensor, size_t bytes, const void* host_data, void** device_ptr);
int ShareImmutableTensorDevice(const std::shared_ptr<CudaExecutionContext>& source_context,
                               const Tensor* source, const Tensor* target);
int AcquireOutputTensorDevice(Tensor* tensor, size_t bytes, void** device_ptr);
// host_data identifies the tensor's current host view. A download acquires a
// tracked writable view internally, preserving copy-on-write and versions.
int SyncTensorToHost(Tensor* tensor, size_t bytes, const void* host_data);
int SyncTensorToHostIfNeeded(Tensor* tensor, size_t bytes, const void* host_data);
int AliasTensorDeviceStorage(const Tensor* input, Tensor* output, size_t bytes);
int SwapTensorDeviceStorage(Tensor* input, Tensor* output);
int CopyTensorStateOnDevice(Tensor* input, const Tensor* output);
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
