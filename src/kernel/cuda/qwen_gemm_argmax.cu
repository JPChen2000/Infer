#include "src/kernel/qwen_gemm_argmax.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <memory>

#include "src/kernel/cuda/kernel_io.cuh"
#include "src/kernel/cuda/linear_kernels.cuh"
#include "util/timer.h"

namespace feather {
namespace kernel {
namespace {

constexpr int kFusedWarpsPerBlock = 8;
constexpr int kFusedThreads = kFusedWarpsPerBlock * 32;
constexpr int kFusedMaxBlocks = 256;
constexpr size_t kFusedMaxActivationSharedBytes = 48 * 1024;

struct CudaQwenArgmaxCandidate {
    float value;
    int64_t index;
};

static_assert(sizeof(CudaQwenArgmaxCandidate) == 16, "candidate layout must remain compact");

std::atomic<int> g_last_cuda_qwen_gemm_argmax_backend{
    static_cast<int>(CudaQwenGemmArgmaxBackend::kUnknown)};

bool ForceCublasFallback() {
    const char* configured = std::getenv("FEATHER_CUDA_QWEN_GEMM_ARGMAX");
    return configured != nullptr &&
           (std::strcmp(configured, "cublas") == 0 || std::strcmp(configured, "fallback") == 0);
}

__device__ inline bool Better(float candidate_value, int64_t candidate_index, float best_value, int64_t best_index) {
    if (candidate_index < 0) {
        return false;
    }
    if (best_index < 0 || candidate_value > best_value ||
        (candidate_value == best_value && candidate_index < best_index)) {
        return true;
    }
    return false;
}

__device__ inline float RoundToBf16(float value) {
    const uint32_t float_bits = __float_as_uint(value);
    const uint32_t exponent = float_bits & 0x7f800000u;
    const uint32_t mantissa = float_bits & 0x007fffffu;
    if (exponent == 0x7f800000u && mantissa != 0) {
        return __uint_as_float((float_bits | 0x00400000u) & 0xffff0000u);
    }
    const uint32_t round_bias = 0x7fffu + ((float_bits >> 16) & 1u);
    const uint16_t rounded_bits = static_cast<uint16_t>((float_bits + round_bias) >> 16);
    return __uint_as_float(static_cast<uint32_t>(rounded_bits) << 16);
}

__device__ inline void ReduceWarpSum(float* value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        *value += __shfl_down_sync(0xffffffffu, *value, offset);
    }
}

// Each warp walks several vocab rows. The dot product is accumulated in FP32,
// then rounded to BF16 before greedy selection to match the CPU Qwen path.
__global__ void FusedBf16GemvArgmaxKernel(const BFloat16* activation, const BFloat16* weight,
                                           CudaQwenArgmaxCandidate* candidates, int64_t hidden,
                                           int64_t vocabulary) {
    extern __shared__ BFloat16 activation_shared[];
    __shared__ float block_values[kFusedWarpsPerBlock];
    __shared__ int64_t block_indices[kFusedWarpsPerBlock];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int64_t block = static_cast<int64_t>(blockIdx.x);
    const int64_t block_stride = static_cast<int64_t>(gridDim.x) * kFusedWarpsPerBlock;

    for (int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
        activation_shared[column] = activation[column];
    }
    __syncthreads();

    float best_value = 0.0f;
    int64_t best_index = -1;
    for (int64_t row = block * kFusedWarpsPerBlock + warp; row < vocabulary; row += block_stride) {
        float sum = 0.0f;
        const int64_t row_offset = row * hidden;
        for (int64_t column = lane; column < hidden; column += 32) {
            sum += cuda_detail::ReadDevice(activation_shared, column) * cuda_detail::ReadDevice(weight, row_offset + column);
        }
        ReduceWarpSum(&sum);
        if (lane == 0) {
            const float rounded = RoundToBf16(sum);
            if (!isnan(rounded) && Better(rounded, row, best_value, best_index)) {
                best_value = rounded;
                best_index = row;
            }
        }
    }

    if (lane == 0) {
        block_values[warp] = best_value;
        block_indices[warp] = best_index;
    }
    __syncthreads();

    if (warp == 0) {
        if (lane < kFusedWarpsPerBlock) {
            for (int stride = kFusedWarpsPerBlock / 2; stride > 0; stride >>= 1) {
                if (lane < stride &&
                    Better(block_values[lane + stride], block_indices[lane + stride], block_values[lane],
                           block_indices[lane])) {
                    block_values[lane] = block_values[lane + stride];
                    block_indices[lane] = block_indices[lane + stride];
                }
                __syncwarp(0xffu);
            }
        }
        if (lane == 0) {
            candidates[block].value = block_values[0];
            candidates[block].index = block_indices[0];
        }
    }
}

__global__ void ReduceBf16GemvCandidatesKernel(const CudaQwenArgmaxCandidate* candidates, int64_t* token,
                                                int64_t count) {
    __shared__ float values[cuda_detail::kCudaThreads];
    __shared__ int64_t indices[cuda_detail::kCudaThreads];
    const int lane = static_cast<int>(threadIdx.x);
    float best_value = 0.0f;
    int64_t best_index = -1;
    for (int64_t index = lane; index < count; index += blockDim.x) {
        const CudaQwenArgmaxCandidate candidate = candidates[index];
        if (Better(candidate.value, candidate.index, best_value, best_index)) {
            best_value = candidate.value;
            best_index = candidate.index;
        }
    }
    values[lane] = best_value;
    indices[lane] = best_index;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride && Better(values[lane + stride], indices[lane + stride], values[lane], indices[lane])) {
            values[lane] = values[lane + stride];
            indices[lane] = indices[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        token[0] = indices[0] < 0 ? 0 : indices[0];
    }
}

__global__ void QwenArgmaxKernelCuda(const BFloat16* logits, int64_t* token, int64_t count) {
    __shared__ float values[cuda_detail::kCudaThreads];
    __shared__ int64_t indices[cuda_detail::kCudaThreads];
    const int lane = static_cast<int>(threadIdx.x);
    float best_value = 0.0f;
    int64_t best_index = -1;
    for (int64_t index = static_cast<int64_t>(lane); index < count; index += blockDim.x) {
        const float value = cuda_detail::ReadDevice(logits, index);
        if (!isnan(value) && Better(value, index, best_value, best_index)) {
            best_value = value;
            best_index = index;
        }
    }
    values[lane] = best_value;
    indices[lane] = best_index;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride && Better(values[lane + stride], indices[lane + stride], values[lane], indices[lane])) {
            values[lane] = values[lane + stride];
            indices[lane] = indices[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        token[0] = indices[0] < 0 ? 0 : indices[0];
    }
}

bool Validate(const operators::QwenGemmArgmaxParam* param, int64_t* hidden, int64_t* vocabulary) {
    if (param == nullptr || hidden == nullptr || vocabulary == nullptr || param->a == nullptr || param->b == nullptr ||
        param->out == nullptr || !param->a->IsInitialized() || !param->b->IsInitialized() ||
        !param->out->IsInitialized() || param->a->data_type() != DataType::BF16 ||
        param->b->data_type() != DataType::BF16 || param->a->dims().size() < 2 || param->b->dims().size() != 2 ||
        param->a->dims()[param->a->dims().size() - 1] != param->b->dims()[1] ||
        param->a->numel() != param->a->dims()[param->a->dims().size() - 1] ||
        param->b->dims()[0] <= 0 || param->out->numel() != 1) {
        return false;
    }
    *hidden = param->a->dims()[param->a->dims().size() - 1];
    *vocabulary = param->b->dims()[0];
    return true;
}

int Run(operators::QwenGemmArgmaxParam* param) {
    int64_t hidden = 0;
    int64_t vocabulary = 0;
    if (!Validate(param, &hidden, &vocabulary)) {
        return -1;
    }
    cuda_detail::DeviceBuffer<BFloat16> activation;
    cuda_detail::DeviceBuffer<BFloat16> weight;
    cuda_detail::DeviceBuffer<int64_t> token;
    if (cuda_detail::CopyTensorToDevice(param->a.get(), &activation) != 0 ||
        cuda_detail::CopyTensorToDevice(param->b.get(), &weight) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &token) != 0) {
        return -1;
    }

    const int64_t fused_blocks = std::max<int64_t>(
        1, std::min<int64_t>(kFusedMaxBlocks, cuda_detail::DivUp(vocabulary, kFusedWarpsPerBlock)));
    const size_t activation_shared_bytes = static_cast<size_t>(hidden) * sizeof(BFloat16);
    const size_t candidate_bytes = static_cast<size_t>(fused_blocks) * sizeof(CudaQwenArgmaxCandidate);
    void* candidate_raw = nullptr;
    if (!ForceCublasFallback() && activation_shared_bytes <= kFusedMaxActivationSharedBytes &&
        cuda_detail::AcquireTemporaryDeviceBuffer(candidate_bytes, &candidate_raw) == 0) {
        auto* candidates = static_cast<CudaQwenArgmaxCandidate*>(candidate_raw);
        FusedBf16GemvArgmaxKernel<<<static_cast<int>(fused_blocks), kFusedThreads, activation_shared_bytes,
                                    cuda_detail::InferenceStream()>>>(
            activation.get(), weight.get(), candidates, hidden, vocabulary);
        if (cuda_detail::CudaCheck(cudaGetLastError()) == 0) {
            ReduceBf16GemvCandidatesKernel<<<1, cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
                candidates, token.get(), fused_blocks);
            const int launch_status = cuda_detail::CudaCheck(cudaGetLastError());
            cuda_detail::ReleaseTemporaryDeviceBuffer(candidate_raw, candidate_bytes);
            if (launch_status == 0) {
                g_last_cuda_qwen_gemm_argmax_backend.store(
                    static_cast<int>(CudaQwenGemmArgmaxBackend::kFusedGemv), std::memory_order_relaxed);
                return cuda_detail::CopyDeviceToTensor(&token, param->out.get());
            }
            return -1;
        }
        cuda_detail::ReleaseTemporaryDeviceBuffer(candidate_raw, candidate_bytes);
        return -1;
    }

    // Keep a cuBLAS fallback for environments where the small candidate
    // allocation cannot be serviced by the CUDA allocator.
    void* logits_raw = nullptr;
    const size_t logits_bytes = static_cast<size_t>(vocabulary) * sizeof(BFloat16);
    if (cuda_detail::AcquireTemporaryDeviceBuffer(logits_bytes, &logits_raw) != 0) {
        return -1;
    }
    auto* logits = static_cast<BFloat16*>(logits_raw);
    const int gemm_status = cuda_detail::LaunchCublasMatMul<DataType::BF16>(
        activation.get(), weight.get(), logits, 1, hidden, vocabulary, 1.0f, true);
    if (gemm_status != 0) {
        cuda_detail::ReleaseTemporaryDeviceBuffer(logits_raw, logits_bytes);
        return -1;
    }
    QwenArgmaxKernelCuda<<<1, cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(logits, token.get(),
                                                                                                  vocabulary);
    const int launch_status = cuda_detail::CudaCheck(cudaGetLastError());
    cuda_detail::ReleaseTemporaryDeviceBuffer(logits_raw, logits_bytes);
    if (launch_status != 0) {
        return -1;
    }
    g_last_cuda_qwen_gemm_argmax_backend.store(
        static_cast<int>(CudaQwenGemmArgmaxBackend::kCublasFallback), std::memory_order_relaxed);
    return cuda_detail::CopyDeviceToTensor(&token, param->out.get());
}

bool g_cuda_qwen_gemm_argmax_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::CUDA, DataType::BF16, "QwenGemmArgmax", []() {
        return std::make_unique<QwenGemmArgmaxKernel<DeviceType::CUDA, DataType::BF16>>();
    });
    return true;
}();

}  // namespace

template <>
int32_t QwenGemmArgmaxKernel<DeviceType::CUDA, DataType::BF16>::compute() {
    AutoTimer timer("CUDA::QwenGemmArgmax::BF16");
    return Run(static_cast<operators::QwenGemmArgmaxParam*>(param_));
}

void EnsureCudaQwenGemmArgmaxKernelsRegistered() { (void)g_cuda_qwen_gemm_argmax_kernels_registered; }

void ResetLastCudaQwenGemmArgmaxBackend() {
    g_last_cuda_qwen_gemm_argmax_backend.store(static_cast<int>(CudaQwenGemmArgmaxBackend::kUnknown),
                                                std::memory_order_relaxed);
}

CudaQwenGemmArgmaxBackend LastCudaQwenGemmArgmaxBackend() {
    return static_cast<CudaQwenGemmArgmaxBackend>(
        g_last_cuda_qwen_gemm_argmax_backend.load(std::memory_order_relaxed));
}

}  // namespace kernel
}  // namespace feather
