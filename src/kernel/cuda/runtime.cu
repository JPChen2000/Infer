#include "src/kernel/cuda/runtime.h"

#include <cuda_runtime.h>

#include <mutex>
#include <unordered_map>

namespace feather {
namespace kernel {
namespace cuda_detail {

namespace {

struct CachedTensorDeviceState {
    void* ptr{nullptr};
    size_t bytes{0};
    const void* host_ptr{nullptr};
    bool device_valid{false};
    bool host_valid{true};
};

std::mutex& CacheMutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<const Tensor*, CachedTensorDeviceState>& TensorCache() {
    static std::unordered_map<const Tensor*, CachedTensorDeviceState> cache;
    return cache;
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

int CudaStatus(cudaError_t status) {
    return status == cudaSuccess ? 0 : -1;
}

void CreateInferenceStreamOnce() {
    StreamCreateStatus() = CudaStatus(cudaStreamCreateWithFlags(&StreamStorage(), cudaStreamNonBlocking));
}

int EnsureInferenceStreamCreated() {
    std::call_once(StreamCreateOnceFlag(), CreateInferenceStreamOnce);
    return StreamCreateStatus();
}

int EnsureDeviceAllocation(CachedTensorDeviceState* state, size_t bytes) {
    if (state == nullptr) {
        return -1;
    }
    if (state->ptr != nullptr && state->bytes >= bytes) {
        state->bytes = bytes;
        return 0;
    }
    if (state->ptr != nullptr) {
        cudaFree(state->ptr);
        state->ptr = nullptr;
    }
    state->bytes = 0;
    if (bytes == 0) {
        return 0;
    }
    if (CudaStatus(cudaMalloc(&state->ptr, bytes)) != 0) {
        return -1;
    }
    state->bytes = bytes;
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

void ClearTensorCache() {
    (void)SynchronizeInferenceStream();
    std::lock_guard<std::mutex> lock(CacheMutex());
    for (auto& item : TensorCache()) {
        if (item.second.ptr != nullptr) {
            cudaFree(item.second.ptr);
        }
    }
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

DeferredHostSyncScope::DeferredHostSyncScope() : previous_(DeferredHostSyncEnabled()) {
    SetDeferredHostSync(true);
}

DeferredHostSyncScope::~DeferredHostSyncScope() {
    SetDeferredHostSync(previous_);
}

}  // namespace cuda_detail
}  // namespace kernel
}  // namespace feather
