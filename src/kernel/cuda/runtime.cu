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

#include "src/kernel/cuda/runtime.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "util/bf16.h"
#include "core/device_replica.h"
#include "src/kernel/cuda/kernel_io.cuh"

namespace feather {
namespace kernel {
namespace cuda_detail {

// The only untracked write access here is to backend-owned device allocations.
class StorageAccess {
 public:
    static void* MutableData(const Storage& storage) { return storage.MutableData(); }
};

namespace {
using detail::DeviceViewIdentity;



struct CachedBf16Weight {
    std::shared_ptr<Storage> storage;
    size_t count{0};
    uint64_t generation{0};
    DataType dtype{DataType::UNKNOWN};
    float scale{0};
};

struct CachedDeviceAllocation {
    std::shared_ptr<Storage> storage;
    void* ptr{nullptr};
    size_t capacity{0};
    std::atomic<bool> shared_immutable{false};
    uint64_t generation{0};
    std::mutex weight_mutex;
    std::weak_ptr<CachedBf16Weight> bf16_weight;
};

struct CachedTensorDeviceState {
    void* ptr{nullptr};
    size_t bytes{0};
    size_t capacity{0};
    std::shared_ptr<CachedDeviceAllocation> allocation;
    const void* host_ptr{nullptr};
    uint64_t host_mutation_version{0};
    bool device_valid{false};
    bool host_valid{true};
    bool persistent{false};
};

using ReplicaIndex = detail::DeviceReplicaIndex<CachedTensorDeviceState>;

struct CachedWeightUse {
    std::weak_ptr<CachedDeviceAllocation> source;
    std::shared_ptr<CachedBf16Weight> weight;
};

struct CapturedAllocation {
    std::shared_ptr<Storage> storage;
    size_t capacity{0};
    bool reusable{true};
    bool pinned{false};
    std::vector<std::weak_ptr<CachedDeviceAllocation>> wrappers;
};

}  // namespace

class CudaExecutionContext {
   public:
    ~CudaExecutionContext();
    std::mutex mutex;
    ReplicaIndex replicas;
    std::unordered_map<const Tensor*, size_t> tensor_users;
    std::unordered_set<const void*> persistent_handles;
    bool capture_recording{false};
    std::unordered_map<void*, CapturedAllocation> captured_allocations;
    std::unordered_map<void*, size_t> capture_pins;
    std::unordered_map<const CachedDeviceAllocation*, size_t> capture_snapshot_references;
    std::unordered_map<size_t, std::vector<void*>> free_blocks;
    std::unordered_map<void*, std::shared_ptr<Storage>> allocations;
    size_t workspace_limit{static_cast<size_t>(-1)};
    size_t fp8_weight_cache_limit{0};
    std::unordered_map<const CachedDeviceAllocation*, CachedWeightUse> fp8_weights;
    size_t workspace_active_bytes{0};
    std::unordered_map<void*, size_t> workspaces;
    bool deferred_sync{false};
    int device{-1};
    cudaStream_t stream{nullptr};
    std::once_flag stream_once, cublas_once;
    int stream_status{-1}, cublas_status{-1};
    cublasHandle_t cublas{nullptr};
#ifdef FEATHER_WITH_CUBLASLT
    cublasLtHandle_t cublaslt{nullptr};
    std::once_flag cublaslt_once;
    int cublaslt_status{-1};
#endif
#ifdef FEATHER_WITH_CUDNN
    cudnnHandle_t cudnn{nullptr};
    std::once_flag cudnn_once;
    int cudnn_status{-1};
#endif
};

class CaptureAllocationSnapshot {
 public:
    explicit CaptureAllocationSnapshot(const CudaExecutionContext& context)
        : replicas(context.replicas), persistent_handles(context.persistent_handles),
          free_blocks(context.free_blocks), allocations(context.allocations),
          workspaces(context.workspaces), workspace_active_bytes(context.workspace_active_bytes) {
        for (const auto& item : replicas)
            if (item.second.allocation) ++wrapper_references[item.second.allocation.get()];
    }
    ReplicaIndex replicas;
    std::unordered_set<const void*> persistent_handles;
    std::unordered_map<size_t, std::vector<void*>> free_blocks;
    std::unordered_map<void*, std::shared_ptr<Storage>> allocations;
    std::unordered_map<void*, size_t> workspaces;
    size_t workspace_active_bytes{0};
    std::unordered_map<const CachedDeviceAllocation*, size_t> wrapper_references;
};

namespace {
std::shared_ptr<CudaExecutionContext>& ActiveContext() {
    static thread_local std::shared_ptr<CudaExecutionContext> context = std::make_shared<CudaExecutionContext>();
    return context;
}
std::mutex& CacheMutex() { return ActiveContext()->mutex; }
auto& ReplicaCache() { return ActiveContext()->replicas; }
auto& FreeDeviceBlocks() { return ActiveContext()->free_blocks; }
bool& DeferredSyncFlag() { return ActiveContext()->deferred_sync; }
cudaStream_t& StreamStorage() { return ActiveContext()->stream; }
std::once_flag& StreamCreateOnceFlag() { return ActiveContext()->stream_once; }
int& StreamCreateStatus() { return ActiveContext()->stream_status; }
cublasHandle_t& CublasHandleStorage() { return ActiveContext()->cublas; }
std::once_flag& CublasCreateOnceFlag() { return ActiveContext()->cublas_once; }
int& CublasCreateStatus() { return ActiveContext()->cublas_status; }
#ifdef FEATHER_WITH_CUBLASLT
cublasLtHandle_t& CublasLtHandleStorage() { return ActiveContext()->cublaslt; }
std::once_flag& CublasLtCreateOnceFlag() { return ActiveContext()->cublaslt_once; }
int& CublasLtCreateStatus() { return ActiveContext()->cublaslt_status; }
#endif
std::atomic<int>& LastFp8MatmulBackendStorage() {
    static thread_local std::atomic<int> backend{static_cast<int>(CudaFp8MatmulBackend::kUnknown)};
    return backend;
}
#ifdef FEATHER_WITH_CUDNN
cudnnHandle_t& CudnnHandleStorage() { return ActiveContext()->cudnn; }
std::once_flag& CudnnCreateOnceFlag() { return ActiveContext()->cudnn_once; }
int& CudnnCreateStatus() { return ActiveContext()->cudnn_status; }
#endif

std::string& LastRuntimeError() {
    static thread_local std::string error;
    return error;
}

int CudaStatus(cudaError_t status) {
    if (status != cudaSuccess) LastRuntimeError() = cudaGetErrorString(status);
    return status == cudaSuccess ? 0 : -1;
}

int CublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) LastRuntimeError() = "cuBLAS status=" + std::to_string(static_cast<int>(status));
    return status == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}

#ifdef FEATHER_WITH_CUDNN
int CudnnStatus(cudnnStatus_t status) {
    if (status != CUDNN_STATUS_SUCCESS) LastRuntimeError() = cudnnGetErrorString(status);
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
    if (bytes > std::numeric_limits<size_t>::max() - (kLargeAlignSize - 1)) return 0;
    return ((bytes + kLargeAlignSize - 1) / kLargeAlignSize) * kLargeAlignSize;
}

void OwnDeviceAllocation(void* ptr, size_t capacity) {
    int device = -1;
    CHECK(cudaGetDevice(&device) == cudaSuccess);
    if (ActiveContext()->device < 0) ActiveContext()->device = device;
    auto owner = std::shared_ptr<void>(ptr, [device](void* memory) {
        int previous = -1;
        (void)cudaGetDevice(&previous);
        if (previous != device) (void)cudaSetDevice(device);
        (void)cudaFree(memory);
        if (previous >= 0 && previous != device) (void)cudaSetDevice(previous);
    });
    auto buffer = std::make_shared<Buffer>(ptr, capacity, std::move(owner));
    ActiveContext()->allocations[ptr] = std::make_shared<Storage>(std::move(buffer), DeviceType::CUDA, device);
}

void RecordCapturedAllocation(void* ptr, size_t capacity, const std::shared_ptr<Storage>& storage,
                              const std::shared_ptr<CachedDeviceAllocation>& wrapper = nullptr) {
    if (!ptr || !ActiveContext()->capture_recording) return;
    auto& recorded = ActiveContext()->captured_allocations[ptr];
    recorded.storage = storage;
    recorded.capacity = capacity;
    if (wrapper) {
        recorded.reusable = recorded.reusable && !wrapper->shared_immutable.load(std::memory_order_relaxed);
        if (recorded.wrappers.empty() || recorded.wrappers.back().lock() != wrapper)
            recorded.wrappers.emplace_back(wrapper);
    }
}

void RecordCapturedAllocation(const CachedTensorDeviceState& state) {
    if (state.allocation) RecordCapturedAllocation(state.ptr, state.capacity, state.allocation->storage, state.allocation);
}

void ReleaseBlockToPool(CachedTensorDeviceState* state) {
    if (state == nullptr || state->ptr == nullptr) {
        return;
    }
    const auto snapshot = ActiveContext()->capture_snapshot_references.find(state->allocation.get());
    const size_t snapshot_references = snapshot == ActiveContext()->capture_snapshot_references.end() ? 0 : snapshot->second;
    if (state->allocation == nullptr ||
        (!state->allocation->shared_immutable.load(std::memory_order_relaxed) &&
         static_cast<size_t>(state->allocation.use_count()) == 1 + snapshot_references)) {
        if (!ActiveContext()->capture_pins.count(state->ptr))
            FreeDeviceBlocks()[state->capacity].push_back(state->ptr);
    }
    if (!ActiveContext()->capture_recording)
        ActiveContext()->fp8_weights.erase(state->allocation.get());
    state->allocation.reset();
    state->ptr = nullptr;
    state->bytes = 0;
    state->capacity = 0;
    state->host_mutation_version = 0;
    state->device_valid = false;
    state->host_valid = true;
}

DeviceViewIdentity TensorView(const Tensor* tensor, size_t bytes) {
    return {tensor->storage()->id(), tensor->offset(), bytes};
}

void RefreshReplicaPersistence(DeviceViewIdentity view) {
    auto found = ReplicaCache().find(view);
    if (found == ReplicaCache().end()) return;
    found->second.persistent = false;
    for (const auto* handle : ReplicaCache().Handles(view))
        if (ActiveContext()->persistent_handles.count(handle)) { found->second.persistent = true; break; }
}

// All Tensor* use ends at this registration adapter. Replica state and alias
// traversal below use stable Storage identities even after handles are gone.
CachedTensorDeviceState& BindTensorReplica(const Tensor* tensor, size_t bytes) {
    auto previous = ReplicaCache().FindHandle(tensor);
    const auto view = TensorView(tensor, bytes);
    const auto previous_view = previous == ReplicaCache().end() ? view : previous->first;
    auto& state = ReplicaCache().Bind(tensor, view,
        [](CachedTensorDeviceState& orphan) { ReleaseBlockToPool(&orphan); });
    RefreshReplicaPersistence(view);
    if (previous_view != view) RefreshReplicaPersistence(previous_view);
    return state;
}

ReplicaIndex::iterator FindTensorReplica(const Tensor* tensor) {
    auto& replicas = ReplicaCache();
    if (!tensor || !tensor->storage()) return replicas.end();
    auto found = replicas.FindHandle(tensor);
    if (found != replicas.end() && found->first.storage_id == tensor->storage()->id() &&
        found->first.offset == tensor->offset() && found->first.bytes <= tensor->memory_size()) return found;
    const auto view = TensorView(tensor, tensor->memory_size());
    if (replicas.find(view) == replicas.end()) return replicas.end();
    BindTensorReplica(tensor, view.bytes);
    return replicas.find(view);
}

void CreateInferenceStreamOnce() {
    if (CudaStatus(cudaGetDevice(&ActiveContext()->device)) != 0) return;
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

#ifdef FEATHER_WITH_CUBLASLT
void CreateCublasLtHandleOnce() {
    if (EnsureInferenceStreamCreated() != 0) {
        CublasLtCreateStatus() = -1;
        return;
    }
    CublasLtCreateStatus() = CublasStatus(cublasLtCreate(&CublasLtHandleStorage()));
}
#endif

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

#ifdef FEATHER_WITH_CUBLASLT
int EnsureCublasLtHandleCreated() {
    std::call_once(CublasLtCreateOnceFlag(), CreateCublasLtHandleOnce);
    return CublasLtCreateStatus();
}
#endif

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
    if (bytes && !capacity) return -1;
    if (state->ptr != nullptr && state->allocation != nullptr && state->capacity >= capacity) {
        state->bytes = bytes;
        RecordCapturedAllocation(*state);
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
    void* device_ptr = nullptr;
    auto free_it = free_blocks.find(capacity);
    if (free_it != free_blocks.end() && !free_it->second.empty()) {
        device_ptr = free_it->second.back();
        free_it->second.pop_back();
    } else {
        if (CudaStatus(cudaMalloc(&device_ptr, capacity)) != 0) return -1;
        OwnDeviceAllocation(device_ptr, capacity);
    }
    state->allocation = std::make_shared<CachedDeviceAllocation>();
    state->allocation->storage = ActiveContext()->allocations.at(device_ptr);
    state->allocation->ptr = device_ptr;
    state->allocation->capacity = capacity;
    state->ptr = device_ptr;
    state->bytes = bytes;
    state->capacity = capacity;
    RecordCapturedAllocation(*state);
    return 0;
}

}  // namespace

CudaExecutionContext::~CudaExecutionContext() {
    // External Storage handles may extend an allocation lifetime. Drop this
    // session's references only after its stream has finished using them.
    int previous = -1;
    if (device >= 0) {
        (void)cudaGetDevice(&previous);
        (void)cudaSetDevice(device);
    }
    if (stream != nullptr) (void)cudaStreamSynchronize(stream);
    replicas.clear();
    fp8_weights.clear();
    free_blocks.clear();
    allocations.clear();
#ifdef FEATHER_WITH_CUDNN
    if (cudnn != nullptr) cudnnDestroy(cudnn);
#endif
#ifdef FEATHER_WITH_CUBLASLT
    if (cublaslt != nullptr) cublasLtDestroy(cublaslt);
#endif
    if (cublas != nullptr) cublasDestroy(cublas);
    if (stream != nullptr) cudaStreamDestroy(stream);
    if (previous >= 0 && previous != device) (void)cudaSetDevice(previous);
}

std::shared_ptr<CudaExecutionContext> CreateExecutionContext() {
    return std::make_shared<CudaExecutionContext>();
}
ExecutionContextScope::ExecutionContextScope(std::shared_ptr<CudaExecutionContext> context)
    : previous_(ActiveContext()) {
    CHECK(context != nullptr);
    if (context != previous_ && context->device >= 0) {
        int current_device = -1;
        CHECK(cudaGetDevice(&current_device) == cudaSuccess);
        if (current_device != context->device) {
            CHECK(cudaSetDevice(context->device) == cudaSuccess);
            previous_device_ = current_device;
        }
    }
    ActiveContext() = std::move(context);
}
ExecutionContextScope::~ExecutionContextScope() {
    ActiveContext() = std::move(previous_);
    if (previous_device_ >= 0) (void)cudaSetDevice(previous_device_);
}

namespace {
struct CaptureAllocationLease {
    std::shared_ptr<CudaExecutionContext> context;
    std::unordered_map<void*, CapturedAllocation> allocations;
    ~CaptureAllocationLease() noexcept {
        try {
            ExecutionContextScope scope(context);
            if (StreamStorage() != nullptr) (void)SynchronizeInferenceStream();
            std::lock_guard<std::mutex> lock(CacheMutex());
            for (const auto& item : allocations) {
                // Retain can fail partway through acquiring pins. Roll back
                // exactly this lease's increments, including shared addresses.
                if (!item.second.pinned) continue;
                auto pin = context->capture_pins.find(item.first);
                if (pin == context->capture_pins.end() || --pin->second != 0) continue;
                context->capture_pins.erase(pin);
                if (!item.second.reusable || context->workspaces.count(item.first)) continue;
                bool referenced = false;
                for (const auto& tensor : ReplicaCache())
                    if (tensor.second.ptr == item.first && tensor.second.allocation) { referenced = true; break; }
                // Exported DeviceStorage handles also prevent recycling.
                if (!referenced) for (const auto& wrapper : item.second.wrappers)
                    if (!wrapper.expired()) { referenced = true; break; }
                if (referenced) continue;
                try {
                    context->allocations.emplace(item.first, item.second.storage);
                    auto& free = FreeDeviceBlocks()[item.second.capacity];
                    if (std::find(free.begin(), free.end(), item.first) == free.end()) free.push_back(item.first);
                } catch (...) {
                    // Recycling is optional. The context allocation map, if
                    // present, retains the block; otherwise this unused lease
                    // storage can safely release it instead of terminating.
                }
            }
        } catch (...) {
            // Never propagate from graph teardown, including an unavailable
            // runtime or mutex failure. Unreleased pins conservatively prevent
            // these addresses from entering another graph's allocation pool.
        }
    }
};
}  // namespace

CaptureAllocationScope::CaptureAllocationScope() : context_(ActiveContext()) {
    std::lock_guard<std::mutex> lock(CacheMutex());
    if (context_->capture_recording) return;
    // Snapshot creation can allocate, and therefore precedes stream capture.
    // Its wrapper references do not change allocator reuse during recording.
    snapshot_ = std::make_unique<CaptureAllocationSnapshot>(*context_);
    context_->capture_snapshot_references.swap(snapshot_->wrapper_references);
    context_->capture_recording = recording_ = true;
    context_->captured_allocations.clear();
}
CaptureAllocationScope::~CaptureAllocationScope() {
    if (!snapshot_) return;
    ExecutionContextScope scope(context_);
    std::lock_guard<std::mutex> lock(CacheMutex());
    if (recording_) {
        // An abandoned graph never executed its writes or uploads. Restore
        // coherence flags, alias identities, and free/workspace lists together.
        context_->replicas.swap(snapshot_->replicas);
        context_->persistent_handles.swap(snapshot_->persistent_handles);
        context_->free_blocks.swap(snapshot_->free_blocks);
        context_->allocations.swap(snapshot_->allocations);
        context_->workspaces.swap(snapshot_->workspaces);
        context_->workspace_active_bytes = snapshot_->workspace_active_bytes;
    }
    context_->capture_recording = false;
    context_->captured_allocations.clear();
    context_->capture_snapshot_references.clear();
}
std::shared_ptr<void> CaptureAllocationScope::Retain() {
    if (!recording_) return nullptr;
    ExecutionContextScope scope(context_);
    auto lease = std::make_shared<CaptureAllocationLease>();
    lease->context = context_;
    std::lock_guard<std::mutex> lock(CacheMutex());
    lease->allocations = std::move(context_->captured_allocations);
    for (auto& item : lease->allocations) {
        ++context_->capture_pins[item.first];
        item.second.pinned = true;
        const auto found = FreeDeviceBlocks().find(item.second.capacity);
        if (found != FreeDeviceBlocks().end()) {
            auto& blocks = found->second;
            blocks.erase(std::remove(blocks.begin(), blocks.end(), item.first), blocks.end());
        }
    }
    context_->capture_recording = recording_ = false;
    context_->capture_snapshot_references.clear();
    snapshot_.reset();
    return lease;
}

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

std::string CudaLastErrorMessage() {
    const auto recorded = std::exchange(LastRuntimeError(), std::string{});
    const auto status = cudaGetLastError();
    if (!recorded.empty()) return recorded;
    if (status == cudaSuccess) {
        return "no CUDA error";
    }
    return cudaGetErrorString(status);
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
    auto it = FindTensorReplica(tensor);
    if (it != ReplicaCache().end()) {
        it->second.device_valid = false;
        it->second.host_valid = true;
        it->second.host_mutation_version = tensor->mutation_version();
    }
}

void MarkTensorDevicePersistent(const Tensor* tensor, bool persistent) {
    if (tensor == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(CacheMutex());
    auto& context = *ActiveContext();
    if (persistent) context.persistent_handles.insert(tensor);
    else context.persistent_handles.erase(tensor);
    if (!tensor->storage()) return;
    BindTensorReplica(tensor, tensor->memory_size());
}

bool IsTensorDevicePersistent(const Tensor* tensor) {
    if (tensor == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(CacheMutex());
    return ActiveContext()->persistent_handles.count(tensor) != 0;
}

void ReleaseTensorDevice(const Tensor* tensor) {
    if (tensor == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(CacheMutex());
    auto it = FindTensorReplica(tensor);
    if (it == ReplicaCache().end() || it->second.persistent) {
        return;
    }
    ReplicaCache().EraseHandle(tensor,
        [](CachedTensorDeviceState& orphan) { ReleaseBlockToPool(&orphan); });
}

void RegisterTensorDeviceUser(const Tensor* tensor) {
    if (!tensor) return;
    std::lock_guard<std::mutex> lock(CacheMutex());
    ++ActiveContext()->tensor_users[tensor];
}

void UnregisterTensorDeviceUsers(const std::vector<const Tensor*>& tensors) {
    if (tensors.empty()) return;
    // Drop allocations only after outstanding graph work has finished. Do not
    // initialize CUDA for graphs that have only executed host kernels.
    if (StreamStorage() != nullptr) (void)SynchronizeInferenceStream();
    std::lock_guard<std::mutex> lock(CacheMutex());
    for (const auto* tensor : tensors) {
        auto user = ActiveContext()->tensor_users.find(tensor);
        if (user == ActiveContext()->tensor_users.end() || --user->second != 0) continue;
        ActiveContext()->tensor_users.erase(user);
        ActiveContext()->persistent_handles.erase(tensor);
        const auto previous = ReplicaCache().FindHandle(tensor);
        if (previous == ReplicaCache().end()) continue;
        const auto view = previous->first;
        // A persistent state can be forgotten only when its final graph user
        // disappears. Other tensor aliases and exported Storage handles retain
        // their own allocation ownership; the pointer must never enter a pool.
        ReplicaCache().EraseHandle(tensor, [](CachedTensorDeviceState& state) {
            const auto ptr = state.ptr;
            const auto& allocation = state.allocation;
            const bool releasable = !allocation || allocation.use_count() == 1 ||
                allocation->shared_immutable.load(std::memory_order_relaxed);
            if (ptr && releasable) ActiveContext()->allocations.erase(ptr);
        });
        RefreshReplicaPersistence(view);
    }
    auto& weights = ActiveContext()->fp8_weights;
    for (auto it = weights.begin(); it != weights.end();) {
        bool referenced = false;
        for (const auto& item : ReplicaCache())
            if (item.second.allocation.get() == it->first) { referenced = true; break; }
        if (!referenced) it = weights.erase(it); else ++it;
    }
}

std::shared_ptr<const Storage> GetTensorDeviceStorage(const Tensor* tensor) {
    std::lock_guard<std::mutex> lock(CacheMutex());
    const auto found = FindTensorReplica(tensor);
    if (found == ReplicaCache().end() || !found->second.allocation) return nullptr;
    return std::shared_ptr<const Storage>(found->second.allocation, found->second.allocation->storage.get());
}

void SetWorkspaceLimit(size_t bytes) {
    std::lock_guard<std::mutex> lock(CacheMutex());
    ActiveContext()->workspace_limit = bytes;
}

void SetFp8WeightCacheLimit(size_t bytes) {
    // Session setup changes policy before execution/capture. Retained graphs
    // keep their own Storage handles if a later policy releases cache entries.
    std::lock_guard<std::mutex> lock(CacheMutex());
    auto& context = *ActiveContext();
    if (context.fp8_weight_cache_limit == bytes) return;
    if (context.capture_recording) return;
    if (StreamStorage() != nullptr && SynchronizeInferenceStream() != 0) return;
    context.fp8_weight_cache_limit = bytes;
    context.fp8_weights.clear();
}

namespace {
template <typename T>
__global__ void CacheFp8WeightAsBf16(const T* source, BFloat16* out, size_t count, float scale) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) WriteDevice(out, i, ReadDevice(source, i, scale));
}

void RecordCachedWeight(const std::shared_ptr<CachedBf16Weight>& weight) {
    void* ptr = StorageAccess::MutableData(*weight->storage);
    RecordCapturedAllocation(ptr, weight->storage->size(), weight->storage);
    if (ActiveContext()->capture_recording)
        ActiveContext()->captured_allocations.at(ptr).reusable = false;
}
}  // namespace

int TryAcquireFp8Bf16Weight(const Tensor* tensor, const void* source, size_t count, float scale,
                            const void** prepared) {
    if (!prepared) return 1;
    *prepared = nullptr;
    if (!tensor || !tensor->is_immutable() || !source || count == 0 ||
        count > std::numeric_limits<size_t>::max() / sizeof(BFloat16) ||
        (tensor->data_type() != DataType::FP8E4M3 && tensor->data_type() != DataType::FP8E5M2) ||
        !std::isfinite(scale) || scale <= 0) return 1;
    std::lock_guard<std::mutex> lock(CacheMutex());
    auto& context = *ActiveContext();
    const size_t bytes = count * sizeof(BFloat16);
    if (bytes > context.fp8_weight_cache_limit) return 1;
    auto state = FindTensorReplica(tensor);
    if (state == ReplicaCache().end() || !state->second.device_valid || state->second.ptr != source ||
        state->second.bytes != count || !state->second.allocation) return 1;
    const auto allocation = state->second.allocation;
    std::lock_guard<std::mutex> weight_lock(allocation->weight_mutex);
    auto weight = allocation->bf16_weight.lock();
    const bool matches = weight && weight->generation == allocation->generation && weight->count == count &&
                         weight->dtype == tensor->data_type() && weight->scale == scale;
    auto existing = context.fp8_weights.find(allocation.get());
    if (matches && existing != context.fp8_weights.end() && existing->second.weight == weight) {
        RecordCachedWeight(weight);
        *prepared = weight->storage->data();
        return 0;
    }
    // Never allocate, synchronize or publish an unexecuted conversion during
    // capture. A cold capture uses the ordinary temporary conversion instead.
    cudaStreamCaptureStatus capturing{};
    if (context.capture_recording || cudaStreamIsCapturing(InferenceStream(), &capturing) != cudaSuccess ||
        capturing != cudaStreamCaptureStatusNone) return 1;
    size_t used = 0;
    for (auto it = context.fp8_weights.begin(); it != context.fp8_weights.end();) {
        if (it->second.source.expired()) it = context.fp8_weights.erase(it);
        else {
            if (it->first != allocation.get()) used += it->second.weight->storage->size();
            ++it;
        }
    }
    if (used > context.fp8_weight_cache_limit || bytes > context.fp8_weight_cache_limit - used) return 1;
    if (!matches) {
        void* ptr = nullptr;
        if (cudaMalloc(&ptr, bytes) != cudaSuccess) {
            (void)cudaGetLastError();
            return 1;  // Caching is optional; scratch/scalar fallback can still run.
        }
        int device = context.device;
        auto owner = std::shared_ptr<void>(ptr, [device](void* memory) {
            int previous = -1;
            (void)cudaGetDevice(&previous);
            if (previous != device) (void)cudaSetDevice(device);
            (void)cudaFree(memory);
            if (previous >= 0 && previous != device) (void)cudaSetDevice(previous);
        });
        auto buffer = std::make_shared<Buffer>(ptr, bytes, std::move(owner));
        weight = std::make_shared<CachedBf16Weight>();
        weight->storage = std::make_shared<Storage>(std::move(buffer), DeviceType::CUDA, device);
        weight->generation = allocation->generation;
        weight->count = count; weight->dtype = tensor->data_type(); weight->scale = scale;
        const auto blocks = static_cast<unsigned>(DivUp(count, kCudaThreads));
        if (tensor->data_type() == DataType::FP8E4M3)
            CacheFp8WeightAsBf16<<<blocks, kCudaThreads, 0, InferenceStream()>>>(
                static_cast<const Fp8E4M3*>(source), static_cast<BFloat16*>(ptr), count, scale);
        else
            CacheFp8WeightAsBf16<<<blocks, kCudaThreads, 0, InferenceStream()>>>(
                static_cast<const Fp8E5M2*>(source), static_cast<BFloat16*>(ptr), count, scale);
        // Publish only completed read-only data so another context/stream can
        // use the shared representation without a per-token event or barrier.
        if (CudaStatus(cudaGetLastError()) != 0 || SynchronizeInferenceStream() != 0) return 1;
        allocation->bf16_weight = weight;
    }
    context.fp8_weights[allocation.get()] = {allocation, weight};
    *prepared = weight->storage->data();
    return 0;
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
    auto& context = *ActiveContext();
    if (!capacity) return -1;
    if (context.workspace_active_bytes > context.workspace_limit ||
        capacity > context.workspace_limit - context.workspace_active_bytes) {
        LastRuntimeError() = "CUDA workspace budget exceeded";
        return -1;
    }
    auto& free_blocks = FreeDeviceBlocks();
    auto free_it = free_blocks.find(capacity);
    if (free_it != free_blocks.end() && !free_it->second.empty()) {
        *device_ptr = free_it->second.back();
        free_it->second.pop_back();
        context.workspaces[*device_ptr] = capacity;
        context.workspace_active_bytes += capacity;
        RecordCapturedAllocation(*device_ptr, capacity, context.allocations.at(*device_ptr));
        return 0;
    }
    const auto status = CudaStatus(cudaMalloc(device_ptr, capacity));
    if (status == 0) {
        OwnDeviceAllocation(*device_ptr, capacity);
        context.workspaces[*device_ptr] = capacity;
        context.workspace_active_bytes += capacity;
        RecordCapturedAllocation(*device_ptr, capacity, context.allocations.at(*device_ptr));
    }
    return status;
}

void ReleaseTemporaryDeviceBuffer(void* device_ptr, size_t bytes) {
    if (device_ptr == nullptr || bytes == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(CacheMutex());
    const auto found = ActiveContext()->workspaces.find(device_ptr);
    if (found == ActiveContext()->workspaces.end()) return;
    const auto capacity = found->second;
    ActiveContext()->workspace_active_bytes -= capacity;
    ActiveContext()->workspaces.erase(found);
    if (!ActiveContext()->capture_pins.count(device_ptr)) FreeDeviceBlocks()[capacity].push_back(device_ptr);
}

TensorCacheStats GetTensorCacheStats() {
    std::lock_guard<std::mutex> lock(CacheMutex());
    TensorCacheStats stats;
    std::unordered_set<void*> active_ptrs;
    for (const auto& item : ReplicaCache()) {
        if (item.second.persistent) {
            ++stats.persistent_tensor_count;
        }
        if (item.second.ptr != nullptr) {
            ++stats.active_tensor_count;
            if (active_ptrs.insert(item.second.ptr).second) {
                stats.active_bytes += item.second.capacity;
            }
        }
    }
    for (const auto& weight : ActiveContext()->fp8_weights) {
        if (weight.second.source.expired()) continue;
        ++stats.fp8_weight_cache_count;
        stats.fp8_weight_cache_bytes += weight.second.weight->storage->size();
    }
    for (const auto& blocks : FreeDeviceBlocks()) {
        stats.free_block_count += blocks.second.size();
        stats.pooled_bytes += blocks.first * blocks.second.size();
    }
    return stats;
}

void ClearTensorCache() {
    if (StreamStorage() != nullptr) (void)SynchronizeInferenceStream();
    std::lock_guard<std::mutex> lock(CacheMutex());
    ReplicaCache().clear();
    ActiveContext()->persistent_handles.clear();
    ActiveContext()->fp8_weights.clear();
    FreeDeviceBlocks().clear();
    ActiveContext()->allocations.clear();
    ActiveContext()->workspaces.clear();
    ActiveContext()->workspace_active_bytes = 0;
}

int AcquireTensorDevice(const Tensor* tensor, size_t bytes, const void* host_data, void** device_ptr) {
    if (tensor == nullptr || !tensor->storage() || bytes > tensor->memory_size() ||
        device_ptr == nullptr || (bytes != 0 && host_data == nullptr)) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(CacheMutex());
    auto& state = BindTensorReplica(tensor, bytes);
    const bool host_changed = !state.device_valid || state.bytes != bytes || state.host_ptr != host_data ||
                              state.host_mutation_version != tensor->mutation_version();
    const bool shared = state.allocation && state.allocation->shared_immutable.load(std::memory_order_relaxed);
    if (shared && host_changed) ReleaseBlockToPool(&state);
    if (EnsureDeviceAllocation(&state, bytes) != 0) return -1;
    const bool must_copy_host = (!DeferredSyncFlag() && !shared && !tensor->is_immutable()) || host_changed;
    if (bytes != 0 && must_copy_host) {
        ++state.allocation->generation;
        if (CudaStatus(cudaMemcpyAsync(state.ptr, host_data, bytes, cudaMemcpyHostToDevice, InferenceStream())) != 0) {
            return -1;
        }
    }
    state.host_ptr = host_data;
    state.host_mutation_version = tensor->mutation_version();
    state.device_valid = true;
    // A device-side read does not refresh a stale host copy. Only an actual
    // upload establishes host/device agreement here.
    if (must_copy_host) state.host_valid = true;
    *device_ptr = state.ptr;
    return 0;
}

int ShareImmutableTensorDevice(const std::shared_ptr<CudaExecutionContext>& source_context,
                               const Tensor* source, const Tensor* target) {
    if (!source_context || !source || !target || !source->is_immutable() || !target->is_immutable() ||
        source->memory_size() != target->memory_size()) return -1;
    if (source_context == ActiveContext() && source == target) return 0;
    const auto target_context = ActiveContext();
    if (EnsureInferenceStreamCreated() != 0) return -1;
    CachedTensorDeviceState imported;
    {
        ExecutionContextScope scope(source_context);
        DeferredHostSyncScope deferred;
        void* ptr = nullptr;
        if (AcquireTensorDevice(source, source->memory_size(), source->raw_data(), &ptr) != 0 ||
            SynchronizeInferenceStream() != 0 || source_context->device != target_context->device) return -1;
        std::lock_guard<std::mutex> lock(CacheMutex());
        imported = FindTensorReplica(source)->second;
        if (!imported.allocation && source->memory_size()) return -1;
        if (imported.allocation) imported.allocation->shared_immutable.store(true, std::memory_order_relaxed);
    }
    std::lock_guard<std::mutex> lock(CacheMutex());
    auto& state = BindTensorReplica(target, target->memory_size());
    ReleaseBlockToPool(&state);
    imported.host_ptr = target->raw_data();
    imported.host_mutation_version = target->mutation_version();
    imported.host_valid = true; imported.persistent = true;
    ActiveContext()->persistent_handles.insert(target);
    state = std::move(imported);
    return 0;
}

int AcquireOutputTensorDevice(Tensor* tensor, size_t bytes, void** device_ptr) {
    if (tensor == nullptr || !tensor->storage() || bytes > tensor->memory_size() || device_ptr == nullptr) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(CacheMutex());
    auto& state = BindTensorReplica(tensor, bytes);
    if (state.allocation && state.allocation->shared_immutable.load(std::memory_order_relaxed)) ReleaseBlockToPool(&state);
    if (EnsureDeviceAllocation(&state, bytes) != 0) {
        return -1;
    }
    state.host_ptr = tensor->raw_data();
    state.host_mutation_version = tensor->mutation_version();
    state.device_valid = true;
    state.host_valid = false;
    if (state.allocation) ++state.allocation->generation;
    *device_ptr = state.ptr;
    return 0;
}

namespace {

bool CanDownloadStorageView(const Tensor& tensor, size_t bytes, const CachedTensorDeviceState& downloaded) {
    const DeviceViewIdentity destination{tensor.storage()->id(), tensor.offset(), bytes};
    const auto version = tensor.mutation_version();
    for (const auto& view : ReplicaCache().StorageViews(destination.storage_id)) {
        const auto& cached = ReplicaCache().find(view)->second;
        if (&cached == &downloaded || cached.host_mutation_version != version || !cached.device_valid ||
            cached.host_valid || !cached.bytes ||
            (cached.allocation && cached.allocation == downloaded.allocation)) continue;
        if (destination.Overlaps(view) && !destination.Contains(view)) {
            LastRuntimeError() = "CUDA download partially overlaps a pending device view";
            return false;
        }
    }
    return true;
}

// A tracked download changes the Storage version and may move its entire COW
// buffer. Visit only this Storage's indexed views: disjoint pending results
// retain device authority, while independent overlapping views follow the new
// host value. No former Tensor handle is dereferenced or host pointer inferred.
void RefreshDownloadedStorageAliases(const Tensor& tensor, uint64_t previous_version,
                                     size_t downloaded_bytes, const CachedTensorDeviceState& downloaded) {
    const auto* current_base = static_cast<const char*>(tensor.storage()->data());
    const DeviceViewIdentity destination{tensor.storage()->id(), tensor.offset(), downloaded_bytes};
    for (const auto& view : ReplicaCache().StorageViews(destination.storage_id)) {
        auto& cached = ReplicaCache().find(view)->second;
        if (cached.host_mutation_version != previous_version) continue;
        cached.host_ptr = current_base + view.offset;
        if (&cached != &downloaded && destination.Overlaps(view) &&
            (!cached.allocation || cached.allocation != downloaded.allocation)) {
            // Partially covered pending views were rejected before host mutation.
            // A true device alias already contains the downloaded result.
            cached.device_valid = false;
            cached.host_valid = true;
        }
        cached.host_mutation_version = tensor.mutation_version();
    }
}

}  // namespace

int SyncTensorToHost(Tensor* tensor, size_t bytes, const void* host_data) {
    if (tensor == nullptr || bytes > tensor->memory_size() ||
        (bytes != 0 && (host_data == nullptr || host_data != tensor->raw_data()))) {
        return -1;
    }
    if (bytes == 0) return 0;
    std::lock_guard<std::mutex> lock(CacheMutex());
    auto it = FindTensorReplica(tensor);
    if (it == ReplicaCache().end() || !it->second.device_valid) {
        return 0;
    }
    auto& state = it->second;
    if (bytes != state.bytes || !CanDownloadStorageView(*tensor, bytes, state)) return -1;
    if (bytes != 0) {
        const auto previous_version = tensor->mutation_version();
        void* destination = tensor->mutable_data(tensor->memory_size());
        if (CudaStatus(cudaMemcpyAsync(destination, state.ptr, bytes, cudaMemcpyDeviceToHost, InferenceStream())) != 0 ||
            SynchronizeInferenceStream() != 0) {
            return -1;
        }
        RefreshDownloadedStorageAliases(*tensor, previous_version, bytes, state);
    }
    state.host_ptr = tensor->raw_data();
    state.host_mutation_version = tensor->mutation_version();
    state.host_valid = true;
    return 0;
}

int SyncTensorToHostIfNeeded(Tensor* tensor, size_t bytes, const void* host_data) {
    if (tensor == nullptr || bytes > tensor->memory_size() ||
        (bytes != 0 && (host_data == nullptr || host_data != tensor->raw_data()))) {
        return -1;
    }
    if (bytes == 0) return 0;
    std::lock_guard<std::mutex> lock(CacheMutex());
    auto it = FindTensorReplica(tensor);
    if (it == ReplicaCache().end() || !it->second.device_valid || it->second.host_valid) {
        return 0;
    }
    auto& state = it->second;
    if (bytes != state.bytes) return -1;
    if (state.host_mutation_version != tensor->mutation_version()) {
        // A host-side writer ran after the CUDA producer. The host buffer is
        // now authoritative for a Common/X86 fallback, so never overwrite it
        // with the stale device result.
        state.host_ptr = host_data;
        state.host_mutation_version = tensor->mutation_version();
        state.device_valid = false;
        state.host_valid = true;
        return 0;
    }
    if (!CanDownloadStorageView(*tensor, bytes, state)) return -1;
    if (bytes != 0) {
        const auto previous_version = tensor->mutation_version();
        void* destination = tensor->mutable_data(tensor->memory_size());
        if (CudaStatus(cudaMemcpyAsync(destination, state.ptr, bytes, cudaMemcpyDeviceToHost, InferenceStream())) != 0 ||
            SynchronizeInferenceStream() != 0) {
            return -1;
        }
        RefreshDownloadedStorageAliases(*tensor, previous_version, bytes, state);
    }
    state.host_ptr = tensor->raw_data();
    state.host_mutation_version = tensor->mutation_version();
    state.host_valid = true;
    return 0;
}

namespace {

size_t TensorByteSize(const Tensor* tensor) {
    if (tensor == nullptr || tensor->numel() <= 0) {
        return 0;
    }
    const size_t element_bytes = DataTypeBytes(tensor->data_type());
    return element_bytes == 0 ? 0 : static_cast<size_t>(tensor->numel()) * element_bytes;
}

int AcquireExistingTensorDevice(const Tensor* tensor, size_t bytes, void** device_ptr) {
    if (tensor == nullptr || device_ptr == nullptr) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(CacheMutex());
    const auto it = FindTensorReplica(tensor);
    if (it == ReplicaCache().end() || it->second.ptr == nullptr || it->second.allocation == nullptr ||
        it->second.allocation->ptr != it->second.ptr || !it->second.device_valid ||
        it->second.bytes < bytes || it->second.host_mutation_version != tensor->mutation_version()) {
        return -1;
    }
    RecordCapturedAllocation(it->second);
    *device_ptr = it->second.ptr;
    return 0;
}

}  // namespace

int AliasTensorDeviceStorage(const Tensor* input, Tensor* output, size_t bytes) {
    if (input == nullptr || output == nullptr) {
        return -1;
    }
    if (input == output) {
        return bytes <= input->memory_size() ? 0 : -1;
    }
    if (!input->IsInitialized() || !output->IsInitialized() || input->numel() != output->numel() ||
        input->memory_size() < bytes || output->memory_size() < bytes) {
        return -1;
    }

    if (bytes != 0) {
        void* input_ptr = nullptr;
        if (AcquireExistingTensorDevice(input, bytes, &input_ptr) != 0) {
            return -1;
        }
    }

    std::lock_guard<std::mutex> lock(CacheMutex());
    const auto input_it = FindTensorReplica(input);
    if (input_it == ReplicaCache().end()) {
        return -1;
    }
    const auto input_allocation = input_it->second.allocation;
    if (bytes != 0 && (input_it->second.ptr == nullptr || input_allocation == nullptr ||
                       !input_it->second.device_valid || input_it->second.bytes < bytes ||
                       input_it->second.host_mutation_version != input->mutation_version())) {
        return -1;
    }

    const auto& input_state = input_it->second;
    auto& output_state = BindTensorReplica(output, bytes);
    const bool persistent = output_state.persistent;
    if (output_state.ptr != nullptr && output_state.allocation != input_allocation) {
        ReleaseBlockToPool(&output_state);
    }
    output_state.persistent = persistent;
    output_state.allocation = input_allocation;
    output_state.ptr = bytes == 0 ? nullptr : input_state.ptr;
    output_state.bytes = bytes;
    output_state.capacity = bytes == 0 ? 0 : input_state.capacity;
    output_state.host_ptr = output->raw_data();
    output_state.host_mutation_version = output->mutation_version();
    output_state.device_valid = true;
    output_state.host_valid = false;
    return 0;
}

template <typename T>
__global__ void AppendTensorStateKernelCuda(T* input, const T* output, int64_t outer, int64_t slots,
                                             int64_t inner) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = outer * inner;
    if (index >= total) {
        return;
    }
    const int64_t outer_index = index / inner;
    const int64_t inner_index = index % inner;
    const int64_t base = outer_index * slots * inner + inner_index;
    for (int64_t slot = 0; slot + 1 < slots; ++slot) {
        input[base + slot * inner] = input[base + (slot + 1) * inner];
    }
    input[base + (slots - 1) * inner] = output[outer_index * inner + inner_index];
}

template <typename T>
int LaunchAppendTensorState(T* input, const T* output, int64_t outer, int64_t slots, int64_t inner) {
    constexpr int kAppendThreads = 256;
    const int64_t total = outer * inner;
    AppendTensorStateKernelCuda<T>
        <<<static_cast<int>((total + kAppendThreads - 1) / kAppendThreads), kAppendThreads, 0,
           InferenceStream()>>>(input, output, outer, slots, inner);
    return CudaStatus(cudaGetLastError());
}

int CopyTensorStateOnDevice(Tensor* input, const Tensor* output) {
    if (!input || !output || !input->IsInitialized() || !output->IsInitialized() || input->is_immutable() ||
        input->dims() != output->dims() || input->data_type() != output->data_type()) return -1;
    if (input == output) return 0;
    size_t bytes = 0;
    if (!CheckedTensorBytes(input->dims().data(), input->data_type(), &bytes) ||
        input->memory_size() < bytes || output->memory_size() < bytes) return -1;
    if (!bytes) return 0;
    void* input_ptr = nullptr;
    void* output_ptr = nullptr;
    if ((AcquireExistingTensorDevice(input, bytes, &input_ptr) != 0 &&
         AcquireTensorDevice(input, bytes, input->raw_data(), &input_ptr) != 0) ||
        AcquireExistingTensorDevice(output, bytes, &output_ptr) != 0) return -1;
    if (input_ptr != output_ptr &&
        CudaStatus(cudaMemcpyAsync(input_ptr, output_ptr, bytes, cudaMemcpyDeviceToDevice, InferenceStream())) != 0) return -1;
    std::lock_guard<std::mutex> lock(CacheMutex());
    auto& state = FindTensorReplica(input)->second;
    state.device_valid = true;
    state.host_valid = false;
    return 0;
}

int SwapTensorDeviceStorage(Tensor* input, Tensor* output) {
    if (input == nullptr || output == nullptr || input == output || !input->IsInitialized() ||
        !output->IsInitialized() || input->data_type() != output->data_type() || input->dims() != output->dims() ||
        TensorByteSize(input) == 0 || TensorByteSize(input) != TensorByteSize(output)) {
        return input == output ? 0 : -1;
    }
    if (SynchronizeInferenceStream() != 0) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(CacheMutex());
    auto input_it = FindTensorReplica(input);
    auto output_it = FindTensorReplica(output);
    if (input_it == ReplicaCache().end() || output_it == ReplicaCache().end() || input_it->second.ptr == nullptr ||
        output_it->second.ptr == nullptr || !input_it->second.device_valid || !output_it->second.device_valid) {
        return -1;
    }
    std::swap(input_it->second, output_it->second);
    const bool persistent = input_it->second.persistent || output_it->second.persistent;
    if (persistent) {
        ActiveContext()->persistent_handles.insert(input);
        ActiveContext()->persistent_handles.insert(output);
    }
    input_it->second.persistent = persistent;
    output_it->second.persistent = persistent;
    input_it->second.host_ptr = input->raw_data();
    input_it->second.host_mutation_version = input->mutation_version();
    input_it->second.host_valid = false;
    input_it->second.device_valid = true;
    output_it->second.host_ptr = output->raw_data();
    output_it->second.host_mutation_version = output->mutation_version();
    output_it->second.host_valid = false;
    output_it->second.device_valid = true;
    return 0;
}

int AppendTensorStateOnDevice(Tensor* input, const Tensor* output, int axis) {
    if (input == nullptr || output == nullptr || !input->IsInitialized() || !output->IsInitialized() ||
        input->data_type() != output->data_type() || input->dims().size() != output->dims().size() || axis < 0 ||
        axis >= static_cast<int>(input->dims().size()) || input->dims()[axis] <= 1 || output->dims()[axis] != 1) {
        return -1;
    }
    int64_t outer = 1;
    int64_t inner = 1;
    for (int index = 0; index < axis; ++index) {
        if (input->dims()[index] != output->dims()[index] || input->dims()[index] <= 0) {
            return -1;
        }
        outer *= input->dims()[index];
    }
    for (size_t index = static_cast<size_t>(axis + 1); index < input->dims().size(); ++index) {
        if (input->dims()[index] != output->dims()[index] || input->dims()[index] <= 0) {
            return -1;
        }
        inner *= input->dims()[index];
    }
    const int64_t slots = input->dims()[axis];
    if (outer <= 0 || inner <= 0 || slots <= 1 || output->numel() != outer * inner ||
        TensorByteSize(input) == 0 || TensorByteSize(output) == 0) {
        return -1;
    }
    void* input_ptr = nullptr;
    void* output_ptr = nullptr;
    const size_t input_bytes = TensorByteSize(input);
    const size_t output_bytes = TensorByteSize(output);
    // State advances can run outside deferred graph execution. Preserve the
    // device history there instead of uploading its stale host copy again.
    // A missing device value or a newer host mutation still needs an upload.
    if ((AcquireExistingTensorDevice(input, input_bytes, &input_ptr) != 0 &&
         AcquireTensorDevice(input, input_bytes, input->raw_data(), &input_ptr) != 0) ||
        AcquireExistingTensorDevice(output, output_bytes, &output_ptr) != 0) {
        return -1;
    }
    int status = -1;
    switch (input->data_type()) {
        case DataType::FP32:
            status = LaunchAppendTensorState(static_cast<float*>(input_ptr), static_cast<const float*>(output_ptr),
                                             outer, slots, inner);
            break;
        case DataType::FP16:
            status = LaunchAppendTensorState(static_cast<uint16_t*>(input_ptr),
                                             static_cast<const uint16_t*>(output_ptr), outer, slots, inner);
            break;
        case DataType::BF16:
            status = LaunchAppendTensorState(static_cast<BFloat16*>(input_ptr),
                                             static_cast<const BFloat16*>(output_ptr), outer, slots, inner);
            break;
        case DataType::INT32:
            status = LaunchAppendTensorState(static_cast<int32_t*>(input_ptr), static_cast<const int32_t*>(output_ptr),
                                             outer, slots, inner);
            break;
        case DataType::INT64:
            status = LaunchAppendTensorState(static_cast<int64_t*>(input_ptr), static_cast<const int64_t*>(output_ptr),
                                             outer, slots, inner);
            break;
        default:
            return -1;
    }
    if (status != 0) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(CacheMutex());
    auto input_it = FindTensorReplica(input);
    if (input_it != ReplicaCache().end()) {
        input_it->second.device_valid = true;
        input_it->second.host_valid = false;
        input_it->second.host_ptr = input->raw_data();
        input_it->second.host_mutation_version = input->mutation_version();
    }
    return 0;
}

cublasHandle_t CublasHandle() {
    if (EnsureCublasHandleCreated() != 0) {
        return nullptr;
    }
    return CublasHandleStorage();
}

#ifdef FEATHER_WITH_CUBLASLT
cublasLtHandle_t CublasLtHandle() {
    if (EnsureCublasLtHandleCreated() != 0) {
        return nullptr;
    }
    return CublasLtHandleStorage();
}
#endif

void ResetLastCudaFp8MatmulBackend() {
    LastFp8MatmulBackendStorage().store(static_cast<int>(CudaFp8MatmulBackend::kUnknown),
                                        std::memory_order_relaxed);
}

void SetLastCudaFp8MatmulBackend(CudaFp8MatmulBackend backend) {
    LastFp8MatmulBackendStorage().store(static_cast<int>(backend), std::memory_order_relaxed);
}

CudaFp8MatmulBackend LastCudaFp8MatmulBackend() {
    return static_cast<CudaFp8MatmulBackend>(LastFp8MatmulBackendStorage().load(std::memory_order_relaxed));
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
