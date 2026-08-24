#include "src/kernel/cuda/runtime.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "util/bf16.h"

namespace feather {
namespace kernel {
namespace cuda_detail {

namespace {

struct CachedDeviceAllocation {
    void* ptr{nullptr};
    size_t capacity{0};
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
    if (state->allocation == nullptr || state->allocation.use_count() == 1) {
        FreeDeviceBlocks()[state->capacity].push_back(state->ptr);
    }
    state->allocation.reset();
    state->ptr = nullptr;
    state->bytes = 0;
    state->capacity = 0;
    state->host_mutation_version = 0;
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
    if (state->ptr != nullptr && state->allocation != nullptr && state->capacity >= capacity) {
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
    void* device_ptr = nullptr;
    auto free_it = free_blocks.find(capacity);
    if (free_it != free_blocks.end() && !free_it->second.empty()) {
        device_ptr = free_it->second.back();
        free_it->second.pop_back();
    } else if (CudaStatus(cudaMalloc(&device_ptr, capacity)) != 0) {
        return -1;
    }
    state->allocation = std::make_shared<CachedDeviceAllocation>();
    state->allocation->ptr = device_ptr;
    state->allocation->capacity = capacity;
    state->ptr = device_ptr;
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

std::string CudaLastErrorMessage() {
    const auto status = cudaGetLastError();
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
    auto it = TensorCache().find(tensor);
    if (it != TensorCache().end()) {
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
    std::unordered_set<void*> active_ptrs;
    for (const auto& item : TensorCache()) {
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
    for (const auto& blocks : FreeDeviceBlocks()) {
        stats.free_block_count += blocks.second.size();
        stats.pooled_bytes += blocks.first * blocks.second.size();
    }
    return stats;
}

void ClearTensorCache() {
    (void)SynchronizeInferenceStream();
    std::lock_guard<std::mutex> lock(CacheMutex());
    std::unordered_set<void*> released_ptrs;
    for (auto& item : TensorCache()) {
        if (item.second.ptr != nullptr && released_ptrs.insert(item.second.ptr).second) {
            cudaFree(item.second.ptr);
        }
    }
    for (auto& blocks : FreeDeviceBlocks()) {
        for (void* ptr : blocks.second) {
            if (ptr != nullptr && released_ptrs.insert(ptr).second) {
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
    const bool must_copy_host = !DeferredSyncFlag() || !state.device_valid || state.host_ptr != host_data ||
                                state.host_mutation_version != tensor->mutation_version();
    if (bytes != 0 && must_copy_host) {
        if (CudaStatus(cudaMemcpyAsync(state.ptr, host_data, bytes, cudaMemcpyHostToDevice, InferenceStream())) != 0) {
            return -1;
        }
    }
    state.host_ptr = host_data;
    state.host_mutation_version = tensor->mutation_version();
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
    state.host_mutation_version = tensor->mutation_version();
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
    state.host_mutation_version = tensor->mutation_version();
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
    if (bytes != 0) {
        if (CudaStatus(cudaMemcpyAsync(host_data, state.ptr, bytes, cudaMemcpyDeviceToHost, InferenceStream())) != 0 ||
            SynchronizeInferenceStream() != 0) {
            return -1;
        }
    }
    state.host_ptr = host_data;
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
    const auto it = TensorCache().find(tensor);
    if (it == TensorCache().end() || it->second.ptr == nullptr || it->second.allocation == nullptr ||
        it->second.allocation->ptr != it->second.ptr || !it->second.device_valid ||
        it->second.bytes < bytes || it->second.host_mutation_version != tensor->mutation_version()) {
        return -1;
    }
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
    const auto input_it = TensorCache().find(input);
    if (input_it == TensorCache().end()) {
        return -1;
    }
    const auto input_allocation = input_it->second.allocation;
    if (bytes != 0 && (input_it->second.ptr == nullptr || input_allocation == nullptr ||
                       !input_it->second.device_valid || input_it->second.bytes < bytes ||
                       input_it->second.host_mutation_version != input->mutation_version())) {
        return -1;
    }

    auto& output_state = TensorCache()[output];
    const bool persistent = output_state.persistent;
    if (output_state.ptr != nullptr && output_state.allocation != input_allocation) {
        ReleaseBlockToPool(&output_state);
    }
    output_state.persistent = persistent;
    output_state.allocation = input_allocation;
    output_state.ptr = bytes == 0 ? nullptr : input_it->second.ptr;
    output_state.bytes = bytes;
    output_state.capacity = bytes == 0 ? 0 : input_it->second.capacity;
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
    auto input_it = TensorCache().find(input);
    auto output_it = TensorCache().find(output);
    if (input_it == TensorCache().end() || output_it == TensorCache().end() || input_it->second.ptr == nullptr ||
        output_it->second.ptr == nullptr || !input_it->second.device_valid || !output_it->second.device_valid) {
        return -1;
    }
    std::swap(input_it->second, output_it->second);
    const bool persistent = input_it->second.persistent || output_it->second.persistent;
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
    if (AcquireTensorDevice(input, input_bytes, input->raw_data(), &input_ptr) != 0 ||
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
    auto input_it = TensorCache().find(input);
    if (input_it != TensorCache().end()) {
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
