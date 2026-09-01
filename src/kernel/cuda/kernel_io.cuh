#ifndef FEATHER_KERNEL_CUDA_KERNEL_IO_CUH
#define FEATHER_KERNEL_CUDA_KERNEL_IO_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "core/tensor.h"
#include "src/kernel/cuda/runtime.h"
#include "util/bf16.h"
#include "util/fp8.h"
#include "util/types.h"

namespace feather {
namespace kernel {
namespace cuda_detail {

constexpr int kCudaThreads = 256;
constexpr int kMaxCudaRank = 8;

inline int CudaCheck(cudaError_t status) {
    return status == cudaSuccess ? 0 : -1;
}

inline int CublasCheck(cublasStatus_t status) {
    return status == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}

inline std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& dims) {
    for (const int64_t dim : dims) {
        if (dim <= 0) return {};
    }
    std::vector<int64_t> strides(dims.size(), 1);
    int64_t suffix = 1;
    for (int64_t i = static_cast<int64_t>(dims.size()) - 1; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = suffix;
        if (suffix > std::numeric_limits<int64_t>::max() / dims[static_cast<size_t>(i)]) return {};
        suffix *= dims[static_cast<size_t>(i)];
    }
    return strides;
}

inline bool HasValidCudaShape(const std::vector<int64_t>& dims) {
    if (dims.size() > kMaxCudaRank) return false;
    if (dims.empty()) return true;
    return !ComputeStrides(dims).empty();
}

inline int64_t ComputeProduct(const std::vector<int64_t>& dims, size_t begin, size_t end) {
    if (begin > end || end > dims.size()) return -1;
    int64_t product = 1;
    for (size_t i = begin; i < end; ++i) {
        if (dims[i] <= 0 || product > std::numeric_limits<int64_t>::max() / dims[i]) {
            return -1;
        }
        product *= dims[i];
    }
    return product;
}

inline bool SafeByteCount(int64_t count, size_t element_bytes, size_t* bytes) {
    if (bytes == nullptr || count < 0 || element_bytes == 0) return false;
    const auto unsigned_count = static_cast<uint64_t>(count);
    if (unsigned_count > std::numeric_limits<size_t>::max() / element_bytes) return false;
    *bytes = static_cast<size_t>(unsigned_count) * element_bytes;
    return true;
}

struct CudaShape {
    int rank{0};
    int64_t dims[kMaxCudaRank]{};
    int64_t strides[kMaxCudaRank]{};
};

inline bool MakeCudaShape(const std::vector<int64_t>& dims, CudaShape* shape) {
    if (shape == nullptr || !HasValidCudaShape(dims)) {
        return false;
    }
    *shape = CudaShape();
    shape->rank = static_cast<int>(dims.size());
    const auto strides = ComputeStrides(dims);
    for (size_t i = 0; i < dims.size(); ++i) {
        shape->dims[i] = dims[i];
        shape->strides[i] = strides[i];
    }
    return true;
}

inline bool InferBroadcastShape(const std::vector<int64_t>& lhs_dims, const std::vector<int64_t>& rhs_dims,
                                std::vector<int64_t>* out_dims) {
    if (out_dims == nullptr) {
        return false;
    }
    const size_t out_rank = std::max(lhs_dims.size(), rhs_dims.size());
    out_dims->assign(out_rank, 1);
    for (size_t i = 0; i < out_rank; ++i) {
        const int64_t lhs_dim = i < out_rank - lhs_dims.size() ? 1 : lhs_dims[i - (out_rank - lhs_dims.size())];
        const int64_t rhs_dim = i < out_rank - rhs_dims.size() ? 1 : rhs_dims[i - (out_rank - rhs_dims.size())];
        if (lhs_dim <= 0 || rhs_dim <= 0 || (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1)) {
            return false;
        }
        (*out_dims)[i] = std::max(lhs_dim, rhs_dim);
    }
    return true;
}

template <typename T>
class DeviceBuffer {
   public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_), owns_memory_(other.owns_memory_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
        other.owns_memory_ = true;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            count_ = other.count_;
            owns_memory_ = other.owns_memory_;
            other.ptr_ = nullptr;
            other.count_ = 0;
            other.owns_memory_ = true;
        }
        return *this;
    }

    ~DeviceBuffer() { reset(); }

    int allocate(size_t count) {
        reset();
        if (count == 0) {
            return 0;
        }
        T* ptr = nullptr;
        if (CudaCheck(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T))) != 0) {
            return -1;
        }
        ptr_ = ptr;
        count_ = count;
        return 0;
    }

    void reset() {
        if (ptr_ != nullptr && owns_memory_) {
            cudaFree(ptr_);
        }
        ptr_ = nullptr;
        count_ = 0;
        owns_memory_ = true;
    }

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t count() const { return count_; }

    void attach(T* ptr, size_t count) {
        reset();
        ptr_ = ptr;
        count_ = count;
        owns_memory_ = false;
    }

   private:
    T* ptr_{nullptr};
    size_t count_{0};
    bool owns_memory_{true};
};

template <DataType dtype>
struct CudaStorage;

template <>
struct CudaStorage<DataType::FP32> {
    using Type = float;
};

template <>
struct CudaStorage<DataType::FP16> {
    using Type = uint16_t;
};

template <>
struct CudaStorage<DataType::BF16> {
    using Type = BFloat16;
};

template <>
struct CudaStorage<DataType::FP8E4M3> {
    using Type = Fp8E4M3;
};

template <>
struct CudaStorage<DataType::FP8E5M2> {
    using Type = Fp8E5M2;
};

template <>
struct CudaStorage<DataType::INT32> {
    using Type = int32_t;
};

template <>
struct CudaStorage<DataType::INT8> {
    using Type = int8_t;
};

template <>
struct CudaStorage<DataType::INT64> {
    using Type = int64_t;
};

template <DataType dtype>
using StorageT = typename CudaStorage<dtype>::Type;

template <DataType dtype>
inline bool HasValidQuantizationScale(const Tensor* tensor) {
    if (tensor == nullptr) return false;
    if constexpr (dtype == DataType::FP8E4M3 || dtype == DataType::FP8E5M2) {
        const float scale = tensor->quantization_scale();
        return HasCompatiblePerTensorQuantization(tensor->quantization()) && std::isfinite(scale) && scale > 0.0f;
    }
    return true;
}

template <DataType dtype>
inline bool IsTensorReady(const Tensor* tensor) {
    if (tensor == nullptr || !tensor->IsInitialized() || tensor->data_type() != dtype ||
        !HasValidCudaShape(tensor->dims().data())) {
        return false;
    }
    const int64_t safe_numel = ComputeProduct(tensor->dims().data(), 0, tensor->dims().size());
    if (safe_numel < 0 || tensor->numel() != safe_numel) return false;
    const auto element_bytes = DataTypeBytes(dtype);
    size_t required_bytes = 0;
    return HasValidQuantizationScale<dtype>(tensor) && SafeByteCount(safe_numel, element_bytes, &required_bytes) &&
           tensor->memory_size() >= required_bytes;
}

template <DataType dtype>
inline bool IsOutputReady(const Tensor* tensor, const std::vector<int64_t>* expected_dims = nullptr) {
    if (tensor == nullptr || !tensor->IsInitialized() || !HasValidCudaShape(tensor->dims().data()) ||
        (tensor->data_type() != DataType::UNKNOWN && tensor->data_type() != dtype)) {
        return false;
    }
    if (expected_dims != nullptr &&
        (!HasValidCudaShape(*expected_dims) || tensor->dims().data() != *expected_dims)) return false;
    const int64_t safe_numel = ComputeProduct(tensor->dims().data(), 0, tensor->dims().size());
    if (safe_numel < 0 || tensor->numel() != safe_numel) return false;
    const auto element_bytes = DataTypeBytes(dtype);
    size_t required_bytes = 0;
    return HasValidQuantizationScale<dtype>(tensor) && SafeByteCount(safe_numel, element_bytes, &required_bytes) &&
           tensor->memory_size() >= required_bytes;
}

template <typename T>
int CopyTensorToDevice(const Tensor* tensor, DeviceBuffer<T>* device) {
    if (tensor == nullptr || device == nullptr || !tensor->IsInitialized() ||
        !HasValidCudaShape(tensor->dims().data()) || tensor->numel() < 0) {
        return -1;
    }
    const int64_t safe_numel = ComputeProduct(tensor->dims().data(), 0, tensor->dims().size());
    size_t bytes = 0;
    if (safe_numel < 0 || tensor->numel() != safe_numel || !SafeByteCount(safe_numel, sizeof(T), &bytes) ||
        tensor->memory_size() < bytes) return -1;
    const size_t count = static_cast<size_t>(safe_numel);
    void* ptr = nullptr;
    if (AcquireTensorDevice(tensor, bytes, count == 0 ? nullptr : tensor->data<T>(), &ptr) != 0) {
        return -1;
    }
    device->attach(reinterpret_cast<T*>(ptr), count);
    return 0;
}

template <typename T>
int AllocateTensorOnDevice(Tensor* tensor, DeviceBuffer<T>* device) {
    if (tensor == nullptr || device == nullptr || !tensor->IsInitialized() ||
        !HasValidCudaShape(tensor->dims().data())) {
        return -1;
    }
    const int64_t safe_numel = ComputeProduct(tensor->dims().data(), 0, tensor->dims().size());
    size_t bytes = 0;
    if (safe_numel < 0 || tensor->numel() != safe_numel || !SafeByteCount(safe_numel, sizeof(T), &bytes)) return -1;
    tensor->set_data_type(DataTypeTrait<T>::type());
    const size_t count = static_cast<size_t>(safe_numel);
    void* ptr = nullptr;
    if (AcquireOutputTensorDevice(tensor, bytes, &ptr) != 0) {
        return -1;
    }
    device->attach(reinterpret_cast<T*>(ptr), count);
    return 0;
}

template <typename T>
int CopyDeviceToTensor(DeviceBuffer<T>* device, Tensor* tensor) {
    if (device == nullptr || tensor == nullptr || !tensor->IsInitialized() ||
        !HasValidCudaShape(tensor->dims().data())) {
        return -1;
    }
    const int64_t safe_numel = ComputeProduct(tensor->dims().data(), 0, tensor->dims().size());
    size_t required_bytes = 0;
    if (safe_numel < 0 || tensor->numel() != safe_numel || !SafeByteCount(safe_numel, sizeof(T), &required_bytes) ||
        device->count() != static_cast<size_t>(safe_numel) || tensor->memory_size() < required_bytes) return -1;
    if (DeferredHostSyncEnabled()) {
        return 0;
    }
    T* host = tensor->mutable_data<T>();
    if (device->count() == 0) {
        return 0;
    }
    return SyncTensorToHost(tensor, required_bytes, host);
}

template <typename T>
int CopyHostVectorToDevice(const std::vector<T>& host, DeviceBuffer<T>* device) {
    if (device == nullptr) {
        return -1;
    }
    if (device->allocate(host.size()) != 0) {
        return -1;
    }
    if (host.empty()) {
        return 0;
    }
    size_t bytes = 0;
    if (!SafeByteCount(static_cast<int64_t>(host.size()), sizeof(T), &bytes)) return -1;
    return CudaCheck(cudaMemcpyAsync(device->get(), host.data(), bytes, cudaMemcpyHostToDevice, InferenceStream()));
}

template <typename T>
int RunDeviceCopy(Tensor* input, Tensor* out) {
    if (input == nullptr || out == nullptr) {
        return -1;
    }
    DeviceBuffer<T> input_device;
    DeviceBuffer<T> output_device;
    if (CopyTensorToDevice(input, &input_device) != 0 || AllocateTensorOnDevice(out, &output_device) != 0) {
        return -1;
    }
    if (input_device.count() != output_device.count()) {
        return -1;
    }
    if (input_device.count() != 0 &&
        CudaCheck(cudaMemcpyAsync(output_device.get(), input_device.get(), input_device.count() * sizeof(T),
                                  cudaMemcpyDeviceToDevice, InferenceStream())) != 0) {
        return -1;
    }
    return CopyDeviceToTensor(&output_device, out);
}

template <typename T>
int RunDeviceAlias(Tensor* input, Tensor* out) {
    if (input == nullptr || out == nullptr || !input->IsInitialized() || !out->IsInitialized() ||
        input->numel() != out->numel()) {
        return -1;
    }
    out->set_data_type(DataTypeTrait<T>::type());
    DeviceBuffer<T> input_device;
    if (CopyTensorToDevice(input, &input_device) != 0) {
        return -1;
    }
    const size_t bytes = input_device.count() * sizeof(T);
    if (AliasTensorDeviceStorage(input, out, bytes) != 0) {
        return -1;
    }
    if (!DeferredHostSyncEnabled()) {
        return SyncTensorToHost(out, bytes, out->raw_data());
    }
    return 0;
}

__device__ inline float ReadDevice(const float* data, int64_t idx) {
    return data[idx];
}

__device__ inline float ReadDevice(const uint16_t* data, int64_t idx) {
    return __half2float(reinterpret_cast<const __half*>(data)[idx]);
}

__device__ inline float ReadDevice(const BFloat16* data, int64_t idx) {
    return __uint_as_float(static_cast<uint32_t>(data[idx].bits) << 16);
}

__device__ inline float ReadDevice(const Fp8E4M3* data, int64_t idx, float scale) {
    return Fp8E4M3ToFloat(data[idx].bits) * scale;
}

__device__ inline float ReadDevice(const Fp8E5M2* data, int64_t idx, float scale) {
    return Fp8E5M2ToFloat(data[idx].bits) * scale;
}

__device__ inline float ReadDevice(const int32_t* data, int64_t idx) {
    return static_cast<float>(data[idx]);
}

__device__ inline float ReadDevice(const int64_t* data, int64_t idx) {
    return static_cast<float>(data[idx]);
}

__device__ inline void WriteDevice(float* data, int64_t idx, float value) {
    data[idx] = value;
}

__device__ inline void WriteDevice(uint16_t* data, int64_t idx, float value) {
    reinterpret_cast<__half*>(data)[idx] = __float2half(value);
}

__device__ inline void WriteDevice(BFloat16* data, int64_t idx, float value) {
    const uint32_t float_bits = __float_as_uint(value);
    const uint32_t exponent = float_bits & 0x7f800000u;
    const uint32_t mantissa = float_bits & 0x007fffffu;
    if (exponent == 0x7f800000u && mantissa != 0) {
        data[idx].bits = static_cast<uint16_t>((float_bits >> 16) | 0x0040u);
        return;
    }
    const uint32_t round_bias = 0x7fffu + ((float_bits >> 16) & 1u);
    data[idx].bits = static_cast<uint16_t>((float_bits + round_bias) >> 16);
}

__device__ inline void WriteDevice(Fp8E4M3* data, int64_t idx, float value, float scale) {
    data[idx].bits = FloatToFp8E4M3(value / scale);
}

__device__ inline void WriteDevice(Fp8E5M2* data, int64_t idx, float value, float scale) {
    data[idx].bits = FloatToFp8E5M2(value / scale);
}

template <typename T>
__device__ inline float ReadDevice(const T* data, int64_t idx, float) {
    return ReadDevice(data, idx);
}

template <typename T>
__device__ inline void WriteDevice(T* data, int64_t idx, float value, float) {
    WriteDevice(data, idx, value);
}

}  // namespace cuda_detail
}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_CUDA_KERNEL_IO_CUH
