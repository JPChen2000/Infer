#ifndef FEATHER_KERNEL_CUDA_LINEAR_KERNELS_CUH
#define FEATHER_KERNEL_CUDA_LINEAR_KERNELS_CUH

#include <cuda_fp16.h>
#ifdef FEATHER_WITH_CUBLASLT
#include <cuda_fp8.h>
#endif

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "src/kernel/cuda/kernel_io.cuh"

namespace feather {
namespace kernel {
namespace cuda_detail {

constexpr int kLinearTile = 16;

template <typename T>
__global__ void MatMulTiledKernelCuda(const T* a, const T* b, const T* bias, T* out, int64_t m, int64_t k, int64_t n,
                                      int bias_mode, float a_scale, float b_scale, float bias_scale, float out_scale,
                                      float alpha, float beta, bool trans_b) {
    __shared__ float a_tile[kLinearTile][kLinearTile];
    __shared__ float b_tile[kLinearTile][kLinearTile];

    const int64_t row = static_cast<int64_t>(blockIdx.y) * kLinearTile + threadIdx.y;
    const int64_t col = static_cast<int64_t>(blockIdx.x) * kLinearTile + threadIdx.x;
    float sum = 0.0f;

    for (int64_t tile = 0; tile < k; tile += kLinearTile) {
        const int64_t a_col = tile + threadIdx.x;
        const int64_t b_row = tile + threadIdx.y;
        a_tile[threadIdx.y][threadIdx.x] =
            (row < m && a_col < k) ? ReadDevice(a, row * k + a_col, a_scale) : 0.0f;
        const int64_t b_offset = trans_b ? col * k + b_row : b_row * n + col;
        b_tile[threadIdx.y][threadIdx.x] = (b_row < k && col < n) ? ReadDevice(b, b_offset, b_scale) : 0.0f;
        __syncthreads();

        for (int i = 0; i < kLinearTile; ++i) {
            sum += a_tile[threadIdx.y][i] * b_tile[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        const int64_t idx = row * n + col;
        if (bias != nullptr) {
            sum = alpha * sum + beta * ReadDevice(bias, bias_mode == 1 ? col : idx, bias_scale);
        } else {
            sum *= alpha;
        }
        WriteDevice(out, idx, sum, out_scale);
    }
}

template <typename T>
inline void LaunchMatMulKernelCuda(const T* a, const T* b, const T* bias, T* out, int64_t m, int64_t k, int64_t n,
                                   int bias_mode, float a_scale = 1.0f, float b_scale = 1.0f,
                                   float bias_scale = 1.0f, float out_scale = 1.0f, float alpha = 1.0f,
                                   float beta = 1.0f, bool trans_b = false) {
    dim3 block(kLinearTile, kLinearTile);
    dim3 grid(static_cast<unsigned int>(DivUp(n, kLinearTile)), static_cast<unsigned int>(DivUp(m, kLinearTile)));
    MatMulTiledKernelCuda<T><<<grid, block, 0, InferenceStream()>>>(a, b, bias, out, m, k, n, bias_mode, a_scale,
                                                                     b_scale, bias_scale, out_scale, alpha, beta,
                                                                     trans_b);
}

template <typename T>
__global__ void AddBiasKernelCuda(T* out, const T* bias, int64_t m, int64_t n, int bias_mode) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = m * n;
    if (idx >= total) {
        return;
    }
    const float bias_value = ReadDevice(bias, bias_mode == 1 ? (idx % n) : idx);
    WriteDevice(out, idx, ReadDevice(out, idx) + bias_value);
}

template <typename T>
__global__ void AddScaledBiasKernelCuda(T* out, const T* bias, int64_t m, int64_t n, int bias_mode, float beta) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = m * n;
    if (idx >= total) {
        return;
    }
    const float bias_value = beta * ReadDevice(bias, bias_mode == 1 ? (idx % n) : idx);
    WriteDevice(out, idx, ReadDevice(out, idx) + bias_value);
}

inline int LaunchCublasMatMulFp32(const float* a, const float* b, float* out, int64_t m, int64_t k, int64_t n,
                                  float alpha_value, bool trans_b) {
    auto handle = CublasHandle();
    if (handle == nullptr) {
        return -1;
    }
    const float alpha = alpha_value;
    const float beta = 0.0f;
    const auto b_operation = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    const int b_leading_dimension = trans_b ? static_cast<int>(k) : static_cast<int>(n);
    return CublasCheck(cublasSgemm(handle, b_operation, CUBLAS_OP_N, static_cast<int>(n), static_cast<int>(m),
                                   static_cast<int>(k), &alpha, b, b_leading_dimension, a, static_cast<int>(k), &beta,
                                   out, static_cast<int>(n)));
}

inline int LaunchCublasMatMulFp16(const uint16_t* a, const uint16_t* b, uint16_t* out, int64_t m, int64_t k,
                                  int64_t n, float alpha_value, bool trans_b) {
    auto handle = CublasHandle();
    if (handle == nullptr) {
        return -1;
    }
    const float alpha = alpha_value;
    const float beta = 0.0f;
    const auto b_operation = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    const int b_leading_dimension = trans_b ? static_cast<int>(k) : static_cast<int>(n);
    return CublasCheck(cublasGemmEx(handle, b_operation, CUBLAS_OP_N, static_cast<int>(n), static_cast<int>(m),
                                    static_cast<int>(k), &alpha, b, CUDA_R_16F, b_leading_dimension, a, CUDA_R_16F,
                                    static_cast<int>(k), &beta, out, CUDA_R_16F, static_cast<int>(n),
                                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

inline int LaunchCublasMatMulBf16(const BFloat16* a, const BFloat16* b, BFloat16* out, int64_t m, int64_t k,
                                  int64_t n, float alpha_value, bool trans_b) {
    auto handle = CublasHandle();
    if (handle == nullptr) {
        return -1;
    }
    const float alpha = alpha_value;
    const float beta = 0.0f;
    const auto b_operation = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    const int b_leading_dimension = trans_b ? static_cast<int>(k) : static_cast<int>(n);
    return CublasCheck(cublasGemmEx(handle, b_operation, CUBLAS_OP_N, static_cast<int>(n), static_cast<int>(m),
                                    static_cast<int>(k), &alpha, b, CUDA_R_16BF, b_leading_dimension, a, CUDA_R_16BF,
                                    static_cast<int>(k), &beta, out, CUDA_R_16BF, static_cast<int>(n),
                                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

inline int LaunchCublasMatMulBf16ToFp32(const BFloat16* a, const BFloat16* b, float* out, int64_t m, int64_t k,
                                        int64_t n, bool trans_b) {
    auto handle = CublasHandle();
    if (handle == nullptr) {
        return -1;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const auto b_operation = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    const int b_leading_dimension = trans_b ? static_cast<int>(k) : static_cast<int>(n);
    return CublasCheck(cublasGemmEx(handle, b_operation, CUBLAS_OP_N, static_cast<int>(n), static_cast<int>(m),
                                    static_cast<int>(k), &alpha, b, CUDA_R_16BF, b_leading_dimension, a, CUDA_R_16BF,
                                    static_cast<int>(k), &beta, out, CUDA_R_32F, static_cast<int>(n),
                                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

template <DataType dtype>
int LaunchCublasMatMul(const StorageT<dtype>* a, const StorageT<dtype>* b, StorageT<dtype>* out, int64_t m, int64_t k,
                       int64_t n, float alpha_value, bool trans_b);

template <>
inline int LaunchCublasMatMul<DataType::FP32>(const StorageT<DataType::FP32>* a, const StorageT<DataType::FP32>* b,
                                              StorageT<DataType::FP32>* out, int64_t m, int64_t k, int64_t n,
                                              float alpha_value, bool trans_b) {
    return LaunchCublasMatMulFp32(a, b, out, m, k, n, alpha_value, trans_b);
}

template <>
inline int LaunchCublasMatMul<DataType::FP16>(const StorageT<DataType::FP16>* a, const StorageT<DataType::FP16>* b,
                                              StorageT<DataType::FP16>* out, int64_t m, int64_t k, int64_t n,
                                              float alpha_value, bool trans_b) {
    return LaunchCublasMatMulFp16(a, b, out, m, k, n, alpha_value, trans_b);
}

template <>
inline int LaunchCublasMatMul<DataType::BF16>(const StorageT<DataType::BF16>* a, const StorageT<DataType::BF16>* b,
                                              StorageT<DataType::BF16>* out, int64_t m, int64_t k, int64_t n,
                                              float alpha_value, bool trans_b) {
    return LaunchCublasMatMulBf16(a, b, out, m, k, n, alpha_value, trans_b);
}

template <typename T>
struct CudaFp8DataType;

template <>
struct CudaFp8DataType<Fp8E4M3> {
    static constexpr DataType value = DataType::FP8E4M3;
};

template <>
struct CudaFp8DataType<Fp8E5M2> {
    static constexpr DataType value = DataType::FP8E5M2;
};

template <typename T>
__global__ void PackFp8WeightColumnMajorKernelCuda(const T* input, T* output, int64_t k, int64_t n) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = k * n;
    if (index >= total) {
        return;
    }
    const int64_t row = index / n;
    const int64_t column = index % n;
    output[row + k * column] = input[index];
}

template <typename T>
__global__ void ConvertFp8ToBf16KernelCuda(const T* input, BFloat16* output, int64_t count, float scale) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    WriteDevice(output, index, ReadDevice(input, index, scale));
}

template <typename T>
__global__ void ConvertBf16ToFp8KernelCuda(const BFloat16* input, T* output, int64_t count, float scale) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    WriteDevice(output, index, ReadDevice(input, index), scale);
}

template <typename T>
__global__ void QuantizeFp8MatmulOutputKernelCuda(const float* input, const T* bias, T* output, int64_t m, int64_t n,
                                                    int bias_mode, float bias_scale, float output_scale, float alpha,
                                                    float beta) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = m * n;
    if (index >= total) {
        return;
    }
    float value = alpha * input[index];
    if (bias != nullptr) {
        value += beta * ReadDevice(bias, bias_mode == 1 ? index % n : index, bias_scale);
    }
    WriteDevice(output, index, value, output_scale);
}

template <typename T>
__global__ void QuantizeFp8MatmulOutputToBf16KernelCuda(const float* input, BFloat16* output, int64_t count,
                                                         float output_scale) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    T quantized{};
    WriteDevice(&quantized, 0, input[index], output_scale);
    WriteDevice(output, index, ReadDevice(&quantized, 0, output_scale));
}

template <typename T>
__global__ void Fp8MatMulToBf16TiledKernelCuda(const T* a, const T* b, BFloat16* out, int64_t m, int64_t k,
                                                int64_t n, float a_scale, float b_scale, float output_scale) {
    __shared__ float a_tile[kLinearTile][kLinearTile];
    __shared__ float b_tile[kLinearTile][kLinearTile];

    const int64_t row = static_cast<int64_t>(blockIdx.y) * kLinearTile + threadIdx.y;
    const int64_t col = static_cast<int64_t>(blockIdx.x) * kLinearTile + threadIdx.x;
    float sum = 0.0f;
    for (int64_t tile = 0; tile < k; tile += kLinearTile) {
        const int64_t a_col = tile + threadIdx.x;
        const int64_t b_row = tile + threadIdx.y;
        a_tile[threadIdx.y][threadIdx.x] = row < m && a_col < k ? ReadDevice(a, row * k + a_col, a_scale) : 0.0f;
        b_tile[threadIdx.y][threadIdx.x] = b_row < k && col < n ? ReadDevice(b, b_row * n + col, b_scale) : 0.0f;
        __syncthreads();
        for (int index = 0; index < kLinearTile; ++index) {
            sum += a_tile[threadIdx.y][index] * b_tile[index][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < m && col < n) {
        T quantized{};
        WriteDevice(&quantized, 0, sum, output_scale);
        WriteDevice(out, row * n + col, ReadDevice(&quantized, 0, output_scale));
    }
}

static __global__ void SetFp8MatmulScalesKernelCuda(float* scales, float weight_scale, float activation_scale) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        scales[0] = weight_scale;
        scales[1] = activation_scale;
    }
}

#ifdef FEATHER_WITH_CUBLASLT

inline bool HasNativeFp8TensorCores() {
    static const bool supported = []() {
        cudaDeviceProp properties{};
        return cudaGetDeviceProperties(&properties, 0) == cudaSuccess &&
               (properties.major > 8 || (properties.major == 8 && properties.minor >= 9));
    }();
    return supported;
}

inline bool SameFp8Value(float expected, float actual) {
    if (std::isnan(expected) || std::isnan(actual)) {
        return std::isnan(expected) && std::isnan(actual);
    }
    return expected == actual && (expected != 0.0f || std::signbit(expected) == std::signbit(actual));
}

template <DataType dtype>
inline bool NativeFp8EncodingCompatible() {
    static const bool compatible = []() {
        for (int code = 0; code < 256; ++code) {
            const auto bits = static_cast<uint8_t>(code);
            float native = 0.0f;
            float project = 0.0f;
            if constexpr (dtype == DataType::FP8E4M3) {
                __nv_fp8_e4m3 value{};
                value.__x = bits;
                native = static_cast<float>(value);
                project = Fp8E4M3ToFloat(bits);
            } else {
                __nv_fp8_e5m2 value{};
                value.__x = bits;
                native = static_cast<float>(value);
                project = Fp8E5M2ToFloat(bits);
            }
            if (!SameFp8Value(project, native)) {
                return false;
            }
        }
        return true;
    }();
    return compatible;
}

template <DataType dtype>
inline cudaDataType NativeFp8CudaType() {
    if constexpr (dtype == DataType::FP8E4M3) {
        return CUDA_R_8F_E4M3;
    }
    return CUDA_R_8F_E5M2;
}

struct Fp8LtPlanKey {
    int64_t m{0};
    int64_t k{0};
    int64_t n{0};

    bool operator==(const Fp8LtPlanKey& other) const {
        return m == other.m && k == other.k && n == other.n;
    }
};

struct Fp8LtPlanKeyHash {
    size_t operator()(const Fp8LtPlanKey& key) const {
        size_t value = std::hash<int64_t>{}(key.m);
        value ^= std::hash<int64_t>{}(key.k) + 0x9e3779b9U + (value << 6U) + (value >> 2U);
        value ^= std::hash<int64_t>{}(key.n) + 0x9e3779b9U + (value << 6U) + (value >> 2U);
        return value;
    }
};

struct Fp8LtPlan {
    cublasLtMatmulDesc_t operation{nullptr};
    cublasLtMatrixLayout_t weight_layout{nullptr};
    cublasLtMatrixLayout_t activation_layout{nullptr};
    cublasLtMatrixLayout_t output_layout{nullptr};
    cublasLtMatmulAlgo_t algorithm{};
    size_t workspace_bytes{0};
    bool supported{false};
};

inline void DestroyFp8LtPlan(Fp8LtPlan* plan) {
    if (plan == nullptr) {
        return;
    }
    if (plan->output_layout != nullptr) cublasLtMatrixLayoutDestroy(plan->output_layout);
    if (plan->activation_layout != nullptr) cublasLtMatrixLayoutDestroy(plan->activation_layout);
    if (plan->weight_layout != nullptr) cublasLtMatrixLayoutDestroy(plan->weight_layout);
    if (plan->operation != nullptr) cublasLtMatmulDescDestroy(plan->operation);
    *plan = Fp8LtPlan();
}

template <DataType dtype>
inline std::unique_ptr<Fp8LtPlan> BuildFp8LtPlan(int64_t m, int64_t k, int64_t n) {
    auto plan = std::make_unique<Fp8LtPlan>();
    const auto handle = CublasLtHandle();
    if (handle == nullptr) {
        return plan;
    }
    const auto type = NativeFp8CudaType<dtype>();
    const cublasOperation_t transpose = CUBLAS_OP_T;
    const cublasOperation_t no_transpose = CUBLAS_OP_N;
    cublasStatus_t status = cublasLtMatmulDescCreate(&plan->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose,
                                                sizeof(transpose));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_TRANSB, &no_transpose,
                                                sizeof(no_transpose));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(&plan->weight_layout, type, static_cast<uint64_t>(k),
                                            static_cast<uint64_t>(n), k);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(&plan->activation_layout, type, static_cast<uint64_t>(k),
                                            static_cast<uint64_t>(m), k);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(&plan->output_layout, CUDA_R_32F, static_cast<uint64_t>(n),
                                            static_cast<uint64_t>(m), n);
    }
    cublasLtMatmulPreference_t preference = nullptr;
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatmulPreferenceCreate(&preference);
    }
    constexpr size_t kMaxWorkspaceBytes = 8 * 1024 * 1024;
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                      &kMaxWorkspaceBytes, sizeof(kMaxWorkspaceBytes));
    }
    cublasLtMatmulHeuristicResult_t heuristic{};
    int result_count = 0;
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatmulAlgoGetHeuristic(handle, plan->operation, plan->weight_layout, plan->activation_layout,
                                                plan->output_layout, plan->output_layout, preference, 1, &heuristic,
                                                &result_count);
    }
    if (preference != nullptr) {
        cublasLtMatmulPreferenceDestroy(preference);
    }
    if (status != CUBLAS_STATUS_SUCCESS || result_count == 0 || heuristic.state != CUBLAS_STATUS_SUCCESS) {
        DestroyFp8LtPlan(plan.get());
        return plan;
    }
    plan->algorithm = heuristic.algo;
    plan->workspace_bytes = heuristic.workspaceSize;
    plan->supported = true;
    return plan;
}

template <DataType dtype>
inline Fp8LtPlan* FindFp8LtPlan(int64_t m, int64_t k, int64_t n) {
    struct Cache {
        std::mutex mutex;
        std::unordered_map<Fp8LtPlanKey, std::unique_ptr<Fp8LtPlan>, Fp8LtPlanKeyHash> plans;
    };
    static Cache cache;
    const Fp8LtPlanKey key{m, k, n};
    std::lock_guard<std::mutex> lock(cache.mutex);
    const auto found = cache.plans.find(key);
    if (found != cache.plans.end()) {
        return found->second.get();
    }
    auto plan = BuildFp8LtPlan<dtype>(m, k, n);
    auto [inserted, unused] = cache.plans.emplace(key, std::move(plan));
    (void)unused;
    return inserted->second.get();
}

struct Fp8LtScaleStorage {
    std::once_flag create_once;
    float* device{nullptr};
    int status{-1};
};

inline Fp8LtScaleStorage& GetFp8LtScaleStorage() {
    static Fp8LtScaleStorage storage;
    return storage;
}

inline float* Fp8LtScaleDeviceBuffer() {
    auto& storage = GetFp8LtScaleStorage();
    std::call_once(storage.create_once, [&storage]() {
        storage.status = CudaCheck(cudaMalloc(&storage.device, 2 * sizeof(float)));
    });
    return storage.status == 0 ? storage.device : nullptr;
}

template <typename T>
struct PackedFp8Weight {
    const void* source_device{nullptr};
    uint64_t mutation_version{0};
    int64_t k{0};
    int64_t n{0};
    T* device{nullptr};
};

template <typename T>
inline int PrepareFp8WeightColumnMajor(const Tensor* tensor, const T* source, int64_t k, int64_t n, bool trans_b,
                                       const T** prepared, void** temporary, size_t* temporary_bytes) {
    if (tensor == nullptr || source == nullptr || prepared == nullptr || temporary == nullptr || temporary_bytes == nullptr) {
        return -1;
    }
    *prepared = source;
    *temporary = nullptr;
    *temporary_bytes = 0;
    if (trans_b) {
        return 0;
    }
    size_t bytes = 0;
    if (!SafeByteCount(k * n, sizeof(T), &bytes)) {
        return -1;
    }
    const int blocks = static_cast<int>(DivUp(k * n, kCudaThreads));
    if (blocks <= 0) {
        return -1;
    }
    if (!tensor->is_immutable()) {
        if (AcquireTemporaryDeviceBuffer(bytes, temporary) != 0) {
            return -1;
        }
        PackFp8WeightColumnMajorKernelCuda<T><<<blocks, kCudaThreads, 0, InferenceStream()>>>(
            source, static_cast<T*>(*temporary), k, n);
        if (CudaCheck(cudaGetLastError()) != 0) {
            ReleaseTemporaryDeviceBuffer(*temporary, bytes);
            *temporary = nullptr;
            return -1;
        }
        *prepared = static_cast<T*>(*temporary);
        *temporary_bytes = bytes;
        return 0;
    }

    struct Cache {
        std::mutex mutex;
        std::unordered_map<const Tensor*, PackedFp8Weight<T>> weights;
    };
    static Cache cache;
    std::lock_guard<std::mutex> lock(cache.mutex);
    auto& entry = cache.weights[tensor];
    if (entry.device != nullptr && entry.source_device == source && entry.mutation_version == tensor->mutation_version() &&
        entry.k == k && entry.n == n) {
        *prepared = entry.device;
        return 0;
    }
    T* packed = nullptr;
    if (CudaCheck(cudaMalloc(&packed, bytes)) != 0) {
        return -1;
    }
    PackFp8WeightColumnMajorKernelCuda<T><<<blocks, kCudaThreads, 0, InferenceStream()>>>(source, packed, k, n);
    if (CudaCheck(cudaGetLastError()) != 0) {
        cudaFree(packed);
        return -1;
    }
    entry.source_device = source;
    entry.mutation_version = tensor->mutation_version();
    entry.k = k;
    entry.n = n;
    entry.device = packed;
    *prepared = packed;
    return 0;
}

template <typename T>
inline int LaunchFp8Bf16TensorCoreFallback(const T* a, const T* b, const T* bias, T* out, int64_t m, int64_t k,
                                           int64_t n, int bias_mode, float a_scale, float b_scale, float bias_scale,
                                           float out_scale, float alpha, float beta, bool trans_b) {
    size_t a_bytes = 0;
    size_t b_bytes = 0;
    size_t output_bytes = 0;
    if (!SafeByteCount(m * k, sizeof(BFloat16), &a_bytes) || !SafeByteCount(k * n, sizeof(BFloat16), &b_bytes) ||
        !SafeByteCount(m * n, sizeof(float), &output_bytes)) {
        return -1;
    }
    void* a_raw = nullptr;
    void* b_raw = nullptr;
    void* output_raw = nullptr;
    if (AcquireTemporaryDeviceBuffer(a_bytes, &a_raw) != 0 || AcquireTemporaryDeviceBuffer(b_bytes, &b_raw) != 0 ||
        AcquireTemporaryDeviceBuffer(output_bytes, &output_raw) != 0) {
        if (output_raw != nullptr) ReleaseTemporaryDeviceBuffer(output_raw, output_bytes);
        if (b_raw != nullptr) ReleaseTemporaryDeviceBuffer(b_raw, b_bytes);
        if (a_raw != nullptr) ReleaseTemporaryDeviceBuffer(a_raw, a_bytes);
        return -1;
    }
    const int a_blocks = static_cast<int>(DivUp(m * k, kCudaThreads));
    const int b_blocks = static_cast<int>(DivUp(k * n, kCudaThreads));
    ConvertFp8ToBf16KernelCuda<T><<<a_blocks, kCudaThreads, 0, InferenceStream()>>>(
        a, static_cast<BFloat16*>(a_raw), m * k, a_scale);
    ConvertFp8ToBf16KernelCuda<T><<<b_blocks, kCudaThreads, 0, InferenceStream()>>>(
        b, static_cast<BFloat16*>(b_raw), k * n, b_scale);
    int status = CudaCheck(cudaGetLastError());
    if (status == 0) {
        status = LaunchCublasMatMulBf16ToFp32(static_cast<BFloat16*>(a_raw), static_cast<BFloat16*>(b_raw),
                                              static_cast<float*>(output_raw), m, k, n, trans_b);
    }
    if (status == 0) {
        const int output_blocks = static_cast<int>(DivUp(m * n, kCudaThreads));
        QuantizeFp8MatmulOutputKernelCuda<T><<<output_blocks, kCudaThreads, 0, InferenceStream()>>>(
            static_cast<float*>(output_raw), bias, out, m, n, bias_mode, bias_scale, out_scale, alpha, beta);
        status = CudaCheck(cudaGetLastError());
    }
    ReleaseTemporaryDeviceBuffer(output_raw, output_bytes);
    ReleaseTemporaryDeviceBuffer(b_raw, b_bytes);
    ReleaseTemporaryDeviceBuffer(a_raw, a_bytes);
    return status;
}

template <typename T>
inline int LaunchFp8Bf16TensorCoreFallbackToBf16(const T* a, const T* b, BFloat16* out, int64_t m, int64_t k,
                                                 int64_t n, float a_scale, float b_scale, float output_scale) {
    size_t a_bytes = 0;
    size_t b_bytes = 0;
    size_t output_bytes = 0;
    if (!SafeByteCount(m * k, sizeof(BFloat16), &a_bytes) || !SafeByteCount(k * n, sizeof(BFloat16), &b_bytes) ||
        !SafeByteCount(m * n, sizeof(float), &output_bytes)) {
        return -1;
    }
    void* a_raw = nullptr;
    void* b_raw = nullptr;
    void* output_raw = nullptr;
    if (AcquireTemporaryDeviceBuffer(a_bytes, &a_raw) != 0 || AcquireTemporaryDeviceBuffer(b_bytes, &b_raw) != 0 ||
        AcquireTemporaryDeviceBuffer(output_bytes, &output_raw) != 0) {
        if (output_raw != nullptr) ReleaseTemporaryDeviceBuffer(output_raw, output_bytes);
        if (b_raw != nullptr) ReleaseTemporaryDeviceBuffer(b_raw, b_bytes);
        if (a_raw != nullptr) ReleaseTemporaryDeviceBuffer(a_raw, a_bytes);
        return -1;
    }
    const int a_blocks = static_cast<int>(DivUp(m * k, kCudaThreads));
    const int b_blocks = static_cast<int>(DivUp(k * n, kCudaThreads));
    ConvertFp8ToBf16KernelCuda<T><<<a_blocks, kCudaThreads, 0, InferenceStream()>>>(
        a, static_cast<BFloat16*>(a_raw), m * k, a_scale);
    ConvertFp8ToBf16KernelCuda<T><<<b_blocks, kCudaThreads, 0, InferenceStream()>>>(
        b, static_cast<BFloat16*>(b_raw), k * n, b_scale);
    int status = CudaCheck(cudaGetLastError());
    if (status == 0) {
        status = LaunchCublasMatMulBf16ToFp32(static_cast<BFloat16*>(a_raw), static_cast<BFloat16*>(b_raw),
                                              static_cast<float*>(output_raw), m, k, n, false);
    }
    if (status == 0) {
        const int output_blocks = static_cast<int>(DivUp(m * n, kCudaThreads));
        QuantizeFp8MatmulOutputToBf16KernelCuda<T><<<output_blocks, kCudaThreads, 0, InferenceStream()>>>(
            static_cast<float*>(output_raw), out, m * n, output_scale);
        status = CudaCheck(cudaGetLastError());
    }
    ReleaseTemporaryDeviceBuffer(output_raw, output_bytes);
    ReleaseTemporaryDeviceBuffer(b_raw, b_bytes);
    ReleaseTemporaryDeviceBuffer(a_raw, a_bytes);
    return status;
}

template <DataType dtype>
inline int LaunchNativeFp8MatMul(const StorageT<dtype>* a, const StorageT<dtype>* b, const Tensor* b_tensor,
                                 const StorageT<dtype>* bias, StorageT<dtype>* out, int64_t m, int64_t k, int64_t n,
                                 int bias_mode, float a_scale, float b_scale, float bias_scale, float out_scale,
                                 float alpha, float beta, bool trans_b) {
    using T = StorageT<dtype>;
    if (!HasNativeFp8TensorCores() || !NativeFp8EncodingCompatible<dtype>()) {
        return -1;
    }
    auto* plan = FindFp8LtPlan<dtype>(m, k, n);
    if (plan == nullptr || !plan->supported) {
        return -1;
    }
    const T* prepared_weight = nullptr;
    void* temporary_weight = nullptr;
    size_t temporary_weight_bytes = 0;
    if (PrepareFp8WeightColumnMajor(b_tensor, b, k, n, trans_b, &prepared_weight, &temporary_weight,
                                    &temporary_weight_bytes) != 0) {
        return -1;
    }
    size_t output_bytes = 0;
    if (!SafeByteCount(m * n, sizeof(float), &output_bytes)) {
        if (temporary_weight != nullptr) ReleaseTemporaryDeviceBuffer(temporary_weight, temporary_weight_bytes);
        return -1;
    }
    void* output_raw = nullptr;
    if (AcquireTemporaryDeviceBuffer(output_bytes, &output_raw) != 0) {
        if (temporary_weight != nullptr) ReleaseTemporaryDeviceBuffer(temporary_weight, temporary_weight_bytes);
        return -1;
    }
    void* workspace = nullptr;
    if (plan->workspace_bytes != 0 && AcquireTemporaryDeviceBuffer(plan->workspace_bytes, &workspace) != 0) {
        ReleaseTemporaryDeviceBuffer(output_raw, output_bytes);
        if (temporary_weight != nullptr) ReleaseTemporaryDeviceBuffer(temporary_weight, temporary_weight_bytes);
        return -1;
    }
    float* scales = Fp8LtScaleDeviceBuffer();
    int status = scales == nullptr ? -1 : 0;
    if (status == 0) {
        SetFp8MatmulScalesKernelCuda<<<1, 1, 0, InferenceStream()>>>(scales, b_scale, a_scale);
        status = CudaCheck(cudaGetLastError());
    }
    const void* weight_scale = scales;
    const void* activation_scale = scales == nullptr ? nullptr : scales + 1;
    if (status == 0) {
        status = CublasCheck(cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                            &weight_scale, sizeof(weight_scale)));
    }
    if (status == 0) {
        status = CublasCheck(cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                            &activation_scale, sizeof(activation_scale)));
    }
    if (status == 0) {
        const float cublas_alpha = 1.0f;
        const float cublas_beta = 0.0f;
        status = CublasCheck(cublasLtMatmul(CublasLtHandle(), plan->operation, &cublas_alpha, prepared_weight,
                                            plan->weight_layout, a, plan->activation_layout, &cublas_beta, output_raw,
                                            plan->output_layout, output_raw, plan->output_layout, &plan->algorithm,
                                            workspace, plan->workspace_bytes, InferenceStream()));
    }
    if (status == 0) {
        const int output_blocks = static_cast<int>(DivUp(m * n, kCudaThreads));
        QuantizeFp8MatmulOutputKernelCuda<T><<<output_blocks, kCudaThreads, 0, InferenceStream()>>>(
            static_cast<float*>(output_raw), bias, out, m, n, bias_mode, bias_scale, out_scale, alpha, beta);
        status = CudaCheck(cudaGetLastError());
    }
    if (workspace != nullptr) ReleaseTemporaryDeviceBuffer(workspace, plan->workspace_bytes);
    ReleaseTemporaryDeviceBuffer(output_raw, output_bytes);
    if (temporary_weight != nullptr) ReleaseTemporaryDeviceBuffer(temporary_weight, temporary_weight_bytes);
    return status;
}

template <DataType dtype>
inline int LaunchNativeFp8MatMulToBf16(const StorageT<dtype>* a, const StorageT<dtype>* b, const Tensor* b_tensor,
                                       BFloat16* out, int64_t m, int64_t k, int64_t n, float a_scale,
                                       float b_scale, float output_scale) {
    using T = StorageT<dtype>;
    if (!HasNativeFp8TensorCores() || !NativeFp8EncodingCompatible<dtype>()) {
        return -1;
    }
    auto* plan = FindFp8LtPlan<dtype>(m, k, n);
    if (plan == nullptr || !plan->supported) {
        return -1;
    }
    const T* prepared_weight = nullptr;
    void* temporary_weight = nullptr;
    size_t temporary_weight_bytes = 0;
    if (PrepareFp8WeightColumnMajor(b_tensor, b, k, n, false, &prepared_weight, &temporary_weight,
                                    &temporary_weight_bytes) != 0) {
        return -1;
    }
    size_t output_bytes = 0;
    if (!SafeByteCount(m * n, sizeof(float), &output_bytes)) {
        if (temporary_weight != nullptr) ReleaseTemporaryDeviceBuffer(temporary_weight, temporary_weight_bytes);
        return -1;
    }
    void* output_raw = nullptr;
    if (AcquireTemporaryDeviceBuffer(output_bytes, &output_raw) != 0) {
        if (temporary_weight != nullptr) ReleaseTemporaryDeviceBuffer(temporary_weight, temporary_weight_bytes);
        return -1;
    }
    void* workspace = nullptr;
    if (plan->workspace_bytes != 0 && AcquireTemporaryDeviceBuffer(plan->workspace_bytes, &workspace) != 0) {
        ReleaseTemporaryDeviceBuffer(output_raw, output_bytes);
        if (temporary_weight != nullptr) ReleaseTemporaryDeviceBuffer(temporary_weight, temporary_weight_bytes);
        return -1;
    }
    float* scales = Fp8LtScaleDeviceBuffer();
    int status = scales == nullptr ? -1 : 0;
    if (status == 0) {
        SetFp8MatmulScalesKernelCuda<<<1, 1, 0, InferenceStream()>>>(scales, b_scale, a_scale);
        status = CudaCheck(cudaGetLastError());
    }
    const void* weight_scale = scales;
    const void* activation_scale = scales == nullptr ? nullptr : scales + 1;
    if (status == 0) {
        status = CublasCheck(cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                            &weight_scale, sizeof(weight_scale)));
    }
    if (status == 0) {
        status = CublasCheck(cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                            &activation_scale, sizeof(activation_scale)));
    }
    if (status == 0) {
        const float cublas_alpha = 1.0f;
        const float cublas_beta = 0.0f;
        status = CublasCheck(cublasLtMatmul(CublasLtHandle(), plan->operation, &cublas_alpha, prepared_weight,
                                            plan->weight_layout, a, plan->activation_layout, &cublas_beta, output_raw,
                                            plan->output_layout, output_raw, plan->output_layout, &plan->algorithm,
                                            workspace, plan->workspace_bytes, InferenceStream()));
    }
    if (status == 0) {
        const int output_blocks = static_cast<int>(DivUp(m * n, kCudaThreads));
        QuantizeFp8MatmulOutputToBf16KernelCuda<T><<<output_blocks, kCudaThreads, 0, InferenceStream()>>>(
            static_cast<float*>(output_raw), out, m * n, output_scale);
        status = CudaCheck(cudaGetLastError());
    }
    if (workspace != nullptr) ReleaseTemporaryDeviceBuffer(workspace, plan->workspace_bytes);
    ReleaseTemporaryDeviceBuffer(output_raw, output_bytes);
    if (temporary_weight != nullptr) ReleaseTemporaryDeviceBuffer(temporary_weight, temporary_weight_bytes);
    return status;
}

#endif  // FEATHER_WITH_CUBLASLT

template <typename T>
inline int LaunchFp8MatMulKernelCuda(const T* a, const T* b, const Tensor* b_tensor, const T* bias, T* out, int64_t m,
                                     int64_t k, int64_t n, int bias_mode, float a_scale, float b_scale,
                                     float bias_scale, float out_scale, float alpha = 1.0f, float beta = 1.0f,
                                     bool trans_b = false) {
    constexpr DataType dtype = CudaFp8DataType<T>::value;
#ifdef FEATHER_WITH_CUBLASLT
    if (LaunchNativeFp8MatMul<dtype>(a, b, b_tensor, bias, out, m, k, n, bias_mode, a_scale, b_scale, bias_scale,
                                     out_scale, alpha, beta, trans_b) == 0) {
        SetLastCudaFp8MatmulBackend(CudaFp8MatmulBackend::kCublasLt);
        return 0;
    }
    if (LaunchFp8Bf16TensorCoreFallback(a, b, bias, out, m, k, n, bias_mode, a_scale, b_scale, bias_scale, out_scale,
                                        alpha, beta, trans_b) == 0) {
        SetLastCudaFp8MatmulBackend(CudaFp8MatmulBackend::kBf16TensorCoreFallback);
        return 0;
    }
#endif
    LaunchMatMulKernelCuda<T>(a, b, bias, out, m, k, n, bias_mode, a_scale, b_scale, bias_scale, out_scale, alpha,
                              beta, trans_b);
    SetLastCudaFp8MatmulBackend(CudaFp8MatmulBackend::kScalarFallback);
    return CudaCheck(cudaGetLastError());
}

// This path retains the exported FP8 quantize/dequantize boundary while
// writing the consumer's BF16 tensor directly. It is used only after the
// Qwen Cast -> MatMul -> Cast pattern has been validated by a graph pass.
template <typename T>
inline int LaunchFp8MatMulToBf16KernelCuda(const T* a, const T* b, const Tensor* b_tensor, BFloat16* out, int64_t m,
                                           int64_t k, int64_t n, float a_scale, float b_scale, float output_scale) {
    constexpr DataType dtype = CudaFp8DataType<T>::value;
#ifdef FEATHER_WITH_CUBLASLT
    if (LaunchNativeFp8MatMulToBf16<dtype>(a, b, b_tensor, out, m, k, n, a_scale, b_scale, output_scale) == 0) {
        SetLastCudaFp8MatmulBackend(CudaFp8MatmulBackend::kCublasLt);
        return 0;
    }
    if (LaunchFp8Bf16TensorCoreFallbackToBf16(a, b, out, m, k, n, a_scale, b_scale, output_scale) == 0) {
        SetLastCudaFp8MatmulBackend(CudaFp8MatmulBackend::kBf16TensorCoreFallback);
        return 0;
    }
#endif
    const dim3 block(kLinearTile, kLinearTile);
    const dim3 grid(static_cast<unsigned int>(DivUp(n, kLinearTile)), static_cast<unsigned int>(DivUp(m, kLinearTile)));
    Fp8MatMulToBf16TiledKernelCuda<T><<<grid, block, 0, InferenceStream()>>>(a, b, out, m, k, n, a_scale, b_scale,
                                                                               output_scale);
    SetLastCudaFp8MatmulBackend(CudaFp8MatmulBackend::kScalarFallback);
    return CudaCheck(cudaGetLastError());
}

}  // namespace cuda_detail
}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_CUDA_LINEAR_KERNELS_CUH
