#include "src/kernel/cuda/runtime.h"

#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace feather {
namespace kernel {
namespace cuda_detail {

namespace {

struct CachedTensorDeviceState {
    void* ptr{nullptr};
    size_t bytes{0};
    size_t capacity{0};
    const void* host_ptr{nullptr};
    bool device_valid{false};
    bool host_valid{true};
    bool persistent{false};
};

std::mutex& CacheMutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<const Tensor*, CachedTensorDeviceState>& TensorCache() {
    static std::unordered_map<const Tensor*, CachedTensorDeviceState> cache;
    return cache;
}

std::unordered_map<size_t, std::vector<void*>>& FreeDeviceBlocks() {
    static std::unordered_map<size_t, std::vector<void*>> blocks;
    return blocks;
}

bool& DeferredSyncFlag() {
    static bool enabled = false;
    return enabled;
}

cudaStream_t& StreamStorage() {
    static cudaStream_t stream = nullptr;
    return stream;
}

std::once_flag& StreamCreateOnceFlag() {
    static std::once_flag once;
    return once;
}

int& StreamCreateStatus() {
    static int status = -1;
    return status;
}

cublasHandle_t& CublasHandleStorage() {
    static cublasHandle_t handle = nullptr;
    return handle;
}

std::once_flag& CublasCreateOnceFlag() {
    static std::once_flag once;
    return once;
}

int& CublasCreateStatus() {
    static int status = -1;
    return status;
}

#ifdef FEATHER_WITH_CUDNN
cudnnHandle_t& CudnnHandleStorage() {
    static cudnnHandle_t handle = nullptr;
    return handle;
}

std::once_flag& CudnnCreateOnceFlag() {
    static std::once_flag once;
    return once;
}

int& CudnnCreateStatus() {
    static int status = -1;
    return status;
}
#endif

int CudaStatus(cudaError_t status) {
    return status == cudaSuccess ? 0 : -1;
}

int CublasStatus(cublasStatus_t status) {
    return status == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}

#ifdef FEATHER_WITH_CUDNN
int CudnnStatus(cudnnStatus_t status) {
    return status == CUDNN_STATUS_SUCCESS ? 0 : -1;
}
#endif

int EnsureInferenceStreamCreated();

size_t NormalizeAllocationSize(size_t bytes) {
    if (bytes == 0) {
        return 0;
    }
    constexpr size_t kMinClassSize = 256;
    constexpr size_t kMaxChunkSize = 8 * 1024;
    constexpr size_t kLargeAlignSize = 4 * 1024;

    if (bytes <= kMinClassSize) {
        return kMinClassSize;
    }
    if (bytes <= kMaxChunkSize) {
        size_t klass = kMinClassSize;
        while (klass < bytes) {
            klass <<= 1;
        }
        return klass;
    }
    return ((bytes + kLargeAlignSize - 1) / kLargeAlignSize) * kLargeAlignSize;
}

void ReleaseBlockToPool(CachedTensorDeviceState* state) {
    if (state == nullptr || state->ptr == nullptr) {
        return;
    }
    FreeDeviceBlocks()[state->capacity].push_back(state->ptr);
    state->ptr = nullptr;
    state->bytes = 0;
    state->capacity = 0;
    state->device_valid = false;
    state->host_valid = true;
}

void CreateInferenceStreamOnce() {
    StreamCreateStatus() = CudaStatus(cudaStreamCreateWithFlags(&StreamStorage(), cudaStreamNonBlocking));
}

void CreateCublasHandleOnce() {
    if (EnsureInferenceStreamCreated() != 0) {
        CublasCreateStatus() = -1;
        return;
    }
    CublasCreateStatus() = CublasStatus(cublasCreate(&CublasHandleStorage()));
    if (CublasCreateStatus() == 0) {
        CublasCreateStatus() = CublasStatus(cublasSetStream(CublasHandleStorage(), StreamStorage()));
    }
}

#ifdef FEATHER_WITH_CUDNN
void CreateCudnnHandleOnce() {
    if (EnsureInferenceStreamCreated() != 0) {
        CudnnCreateStatus() = -1;
        return;
    }
    CudnnCreateStatus() = CudnnStatus(cudnnCreate(&CudnnHandleStorage()));
    if (CudnnCreateStatus() == 0) {
        CudnnCreateStatus() = CudnnStatus(cudnnSetStream(CudnnHandleStorage(), StreamStorage()));
    }
}
#endif

int EnsureInferenceStreamCreated() {
    std::call_once(StreamCreateOnceFlag(), CreateInferenceStreamOnce);
    return StreamCreateStatus();
}

int EnsureCublasHandleCreated() {
    std::call_once(CublasCreateOnceFlag(), CreateCublasHandleOnce);
    if (CublasCreateStatus() != 0) {
        return CublasCreateStatus();
    }
    return CublasStatus(cublasSetStream(CublasHandleStorage(), StreamStorage()));
}

#ifdef FEATHER_WITH_CUDNN
int EnsureCudnnHandleCreated() {
    std::call_once(CudnnCreateOnceFlag(), CreateCudnnHandleOnce);
    if (CudnnCreateStatus() != 0) {
        return CudnnCreateStatus();
    }
    return CudnnStatus(cudnnSetStream(CudnnHandleStorage(), StreamStorage()));
}
#endif

int EnsureDeviceAllocation(CachedTensorDeviceState* state, size_t bytes) {
    if (state == nullptr) {
        return -1;
    }
    const size_t capacity = NormalizeAllocationSize(bytes);
    if (state->ptr != nullptr && state->capacity >= capacity) {
        state->bytes = bytes;
        return 0;
    }
    if (state->ptr != nullptr) {
        ReleaseBlockToPool(state);
    }
    state->bytes = 0;
    state->capacity = 0;
    if (bytes == 0) {
        return 0;
    }
    auto& free_blocks = FreeDeviceBlocks();
    auto free_it = free_blocks.find(capacity);
    if (free_it != free_blocks.end() && !free_it->second.empty()) {
        state->ptr = free_it->second.back();
        free_it->second.pop_back();
    } else if (CudaStatus(cudaMalloc(&state->ptr, capacity)) != 0) {
        return -1;
    }
    state->bytes = bytes;
    state->capacity = capacity;
    return 0;
}

}  // namespace

void SetDeferredHostSync(bool enabled) {
    std::lock_guard<std::mutex> lock(CacheMutex());
    DeferredSyncFlag() = enabled;
}

bool DeferredHostSyncEnabled() {
    std::lock_guard<std::mutex> lock(CacheMutex());
    return DeferredSyncFlag();
}

int WarmupCudaRuntime() {
    if (CudaStatus(cudaFree(nullptr)) != 0) {
        return -1;
    }
    return EnsureInferenceStreamCreated();
}

cudaStream_t InferenceStream() {
    if (EnsureInferenceStreamCreated() != 0) {
        return nullptr;
    }
    return StreamStorage();
}

int SynchronizeInferenceStream() {
    if (EnsureInferenceStreamCreated() != 0) {
        return -1;
    }
    return CudaStatus(cudaStreamSynchronize(StreamStorage()));
}

void InvalidateTensorDevice(const Tensor* tensor) {
    if (tensor == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(CacheMutex());
    auto it = TensorCache().find(tensor);
    if (it != TensorCache().end()) {
        it->second.device_valid = false;
        it->second.host_valid = true;
    }
}

void MarkTensorDevicePersistent(const Tensor* tensor, bool persistent) {
    if (tensor == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(CacheMutex());
    TensorCache()[tensor].persistent = persistent;
}

bool IsTensorDevicePersistent(const Tensor* tensor) {
    if (tensor == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(CacheMutex());
    auto it = TensorCache().find(tensor);
    return it != TensorCache().end() && it->second.persistent;
}

void ReleaseTensorDevice(const Tensor* tensor) {
    if (tensor == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(CacheMutex());
    auto it = TensorCache().find(tensor);
    if (it == TensorCache().end() || it->second.persistent) {
        return;
    }
    ReleaseBlockToPool(&it->second);
}

int AcquireTemporaryDeviceBuffer(size_t bytes, void** device_ptr) {
    if (device_ptr == nullptr) {
        return -1;
    }
    *device_ptr = nullptr;
    if (bytes == 0) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(CacheMutex());
    const size_t capacity = NormalizeAllocationSize(bytes);
    auto& free_blocks = FreeDeviceBlocks();
    auto free_it = free_blocks.find(capacity);
    if (free_it != free_blocks.end() && !free_it->second.empty()) {
        *device_ptr = free_it->second.back();
        free_it->second.pop_back();
        return 0;
    }
    return CudaStatus(cudaMalloc(device_ptr, capacity));
}

void ReleaseTemporaryDeviceBuffer(void* device_ptr, size_t bytes) {
    if (device_ptr == nullptr || bytes == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(CacheMutex());
    FreeDeviceBlocks()[NormalizeAllocationSize(bytes)].push_back(device_ptr);
}

TensorCacheStats GetTensorCacheStats() {
    std::lock_guard<std::mutex> lock(CacheMutex());
    TensorCacheStats stats;
    for (const auto& item : TensorCache()) {
        if (item.second.persistent) {
            ++stats.persistent_tensor_count;
        }
        if (item.second.ptr != nullptr) {
            ++stats.active_tensor_count;
            stats.active_bytes += item.second.capacity;
        }
    }
    for (const auto& blocks : FreeDeviceBlocks()) {
        stats.free_block_count += blocks.second.size();
        stats.pooled_bytes += blocks.first * blocks.second.size();
    }
    return stats;
}

void ClearTensorCache() {
    (void)SynchronizeInferenceStream();
    std::lock_guard<std::mutex> lock(CacheMutex());
    for (auto& item : TensorCache()) {
        if (item.second.ptr != nullptr) {
            cudaFree(item.second.ptr);
        }
    }
    for (auto& blocks : FreeDeviceBlocks()) {
        for (void* ptr : blocks.second) {
            if (ptr != nullptr) {
                cudaFree(ptr);
            }
        }
    }
    FreeDeviceBlocks().clear();
    TensorCache().clear();
}

int AcquireTensorDevice(const Tensor* tensor, size_t bytes, const void* host_data, void** device_ptr) {
    if (tensor == nullptr || device_ptr == nullptr || (bytes != 0 && host_data == nullptr)) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(CacheMutex());
    auto& state = TensorCache()[tensor];
    if (EnsureDeviceAllocation(&state, bytes) != 0) {
        return -1;
    }
    const bool must_copy_host = !DeferredSyncFlag() || !state.device_valid || state.host_ptr != host_data;
    if (bytes != 0 && must_copy_host) {
        if (CudaStatus(cudaMemcpyAsync(state.ptr, host_data, bytes, cudaMemcpyHostToDevice, InferenceStream())) != 0) {
            return -1;
        }
    }
    state.host_ptr = host_data;
    state.device_valid = true;
    state.host_valid = true;
    *device_ptr = state.ptr;
    return 0;
}

int AcquireOutputTensorDevice(Tensor* tensor, size_t bytes, void** device_ptr) {
    if (tensor == nullptr || device_ptr == nullptr) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(CacheMutex());
    auto& state = TensorCache()[tensor];
    if (EnsureDeviceAllocation(&state, bytes) != 0) {
        return -1;
    }
    state.host_ptr = tensor->raw_data();
    state.device_valid = true;
    state.host_valid = false;
    *device_ptr = state.ptr;
    return 0;
}

int SyncTensorToHost(Tensor* tensor, size_t bytes, void* host_data) {
    if (tensor == nullptr || (bytes != 0 && host_data == nullptr)) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(CacheMutex());
    auto it = TensorCache().find(tensor);
    if (it == TensorCache().end() || !it->second.device_valid) {
        return 0;
    }
    auto& state = it->second;
    if (bytes != 0) {
        if (CudaStatus(cudaMemcpyAsync(host_data, state.ptr, bytes, cudaMemcpyDeviceToHost, InferenceStream())) != 0 ||
            SynchronizeInferenceStream() != 0) {
            return -1;
        }
    }
    state.host_ptr = host_data;
    state.host_valid = true;
    return 0;
}

int SyncTensorToHostIfNeeded(Tensor* tensor, size_t bytes, void* host_data) {
    if (tensor == nullptr || (bytes != 0 && host_data == nullptr)) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(CacheMutex());
    auto it = TensorCache().find(tensor);
    if (it == TensorCache().end() || !it->second.device_valid || it->second.host_valid) {
        return 0;
    }
    auto& state = it->second;
    if (bytes != 0) {
        if (CudaStatus(cudaMemcpyAsync(host_data, state.ptr, bytes, cudaMemcpyDeviceToHost, InferenceStream())) != 0 ||
            SynchronizeInferenceStream() != 0) {
            return -1;
        }
    }
    state.host_ptr = host_data;
    state.host_valid = true;
    return 0;
}

cublasHandle_t CublasHandle() {
    if (EnsureCublasHandleCreated() != 0) {
        return nullptr;
    }
    return CublasHandleStorage();
}

#ifdef FEATHER_WITH_CUDNN
cudnnHandle_t CudnnHandle() {
    if (EnsureCudnnHandleCreated() != 0) {
        return nullptr;
    }
    return CudnnHandleStorage();
}
#endif

DeferredHostSyncScope::DeferredHostSyncScope() : previous_(DeferredHostSyncEnabled()) {
    SetDeferredHostSync(true);
}

DeferredHostSyncScope::~DeferredHostSyncScope() {
    SetDeferredHostSync(previous_);
}

}  // namespace cuda_detail
}  // namespace kernel
}  // namespace feather
