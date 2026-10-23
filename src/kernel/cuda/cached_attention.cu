// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#include "src/operator/cached_attention_op.h"

#include <algorithm>
#include <limits>
#include <math_constants.h>

#include "src/kernel/cuda/kernel_io.cuh"

namespace feather {
namespace kernel {
namespace {
constexpr int kAttentionThreads = 128;
constexpr int kScoreTile = 32;
struct AttentionShape {
    int64_t batch, heads, kv_heads, tokens, depth, value_depth, capacity, value_tiles;
};

template <typename T>
__device__ float RoundStorage(float value) {
    T storage;
    cuda_detail::WriteDevice(&storage, 0, value);
    return cuda_detail::ReadDevice(&storage, 0);
}

template <typename T>
__global__ void WriteAttentionCache(const T* k, const T* v, T* cache_k, T* cache_v,
                                    const int64_t* position, AttentionShape s) {
    const int64_t start = position[0];
    // Caller validates the absolute position before launch. Retain guards on
    // device so malformed controls cannot corrupt the cache during replay.
    if (start < 0 || start > s.capacity - s.tokens) return;
    const int64_t k_count = s.batch * s.kv_heads * s.tokens * s.depth;
    const int64_t v_count = s.batch * s.kv_heads * s.tokens * s.value_depth;
    const int64_t count = k_count > v_count ? k_count : v_count;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride) {
        if (i < k_count) {
            const int64_t d = i % s.depth, token = (i / s.depth) % s.tokens;
            const int64_t head = i / (s.tokens * s.depth);
            cache_k[(head * s.capacity + start + token) * s.depth + d] = k[i];
        }
        if (i < v_count) {
            const int64_t d = i % s.value_depth, token = (i / s.value_depth) % s.tokens;
            const int64_t head = i / (s.tokens * s.value_depth);
            cache_v[(head * s.capacity + start + token) * s.value_depth + d] = v[i];
        }
    }
}

// Four warps cooperate on 32 keys. Every warp reduces one QK dot at a time;
// shared storage contains only this small score tile, independent of capacity.
template <typename T, bool rounded>
__device__ void FillScores(const T* q, const T* cache_k, float* scores, int64_t q_base,
                           int64_t kv_head, int64_t first, int64_t prefix,
                           AttentionShape s, float scale) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int j = warp; j < kScoreTile; j += kAttentionThreads / 32) {
        float dot = 0.0f;
        if (first + j < prefix) {
            const int64_t key_base = (kv_head * s.capacity + first + j) * s.depth;
            for (int64_t d = lane; d < s.depth; d += 32)
                dot += cuda_detail::ReadDevice(q, q_base + d) * cuda_detail::ReadDevice(cache_k, key_base + d);
            for (int offset = 16; offset > 0; offset /= 2)
                dot += __shfl_down_sync(0xffffffffu, dot, offset);
        }
        if (lane == 0) {
            if (rounded) dot = RoundStorage<T>(dot);
            scores[j] = first + j < prefix ? dot * scale : -CUDART_INF_F;
        }
    }
}

template <typename T, bool rounded>
__global__ void FusedCachedAttention(const T* q, const T* cache_k, const T* cache_v,
                                     const int64_t* position, T* out, AttentionShape s, float scale) {
    __shared__ float scores[kScoreTile];
    __shared__ float maximum, denominator, alpha;
    const int64_t row = static_cast<int64_t>(blockIdx.x) / s.value_tiles;
    const int64_t value_dimension = (static_cast<int64_t>(blockIdx.x) % s.value_tiles) * blockDim.x + threadIdx.x;
    const int64_t token = row % s.tokens;
    const int64_t head = (row / s.tokens) % s.heads;
    const int64_t batch = row / (s.tokens * s.heads);
    const int64_t start = position[0];
    if (start < 0 || start > s.capacity - s.tokens) {
        // Invalid device controls are observable without a context-poisoning
        // trap: no cache writes, and every output is NaN. Captured/deferred
        // execution requires caller-side position bounds validation.
        if (value_dimension < s.value_depth)
            cuda_detail::WriteDevice(out, row * s.value_depth + value_dimension, CUDART_NAN_F);
        return;
    }
    const int64_t prefix = start + token + 1;
    const int64_t kv_head = batch * s.kv_heads + head / (s.heads / s.kv_heads);
    const int64_t q_base = row * s.depth;
    if (threadIdx.x == 0) { maximum = -CUDART_INF_F; denominator = 0.0f; }
    __syncthreads();
    float value = 0.0f;
    for (int64_t first = 0; first < prefix; first += kScoreTile) {
        FillScores<T, rounded>(q, cache_k, scores, q_base, kv_head, first, prefix, s, scale);
        __syncthreads();
        if (threadIdx.x == 0) {
            float tile_max = -CUDART_INF_F;
            for (int j = 0; j < kScoreTile; ++j) tile_max = fmaxf(tile_max, scores[j]);
            const float next_max = fmaxf(maximum, tile_max);
            alpha = expf(maximum - next_max);
            float tile_sum = 0.0f;
            for (int j = 0; j < kScoreTile; ++j) {
                scores[j] = expf(scores[j] - next_max);
                tile_sum += scores[j];
            }
            denominator = denominator * alpha + tile_sum;
            maximum = next_max;
        }
        __syncthreads();
        if (!rounded && value_dimension < s.value_depth) {
            value *= alpha;
            for (int j = 0; j < kScoreTile && first + j < prefix; ++j) {
                const int64_t offset = (kv_head * s.capacity + first + j) * s.value_depth + value_dimension;
                value += scores[j] * cuda_detail::ReadDevice(cache_v, offset);
            }
        }
        __syncthreads();
    }
    // Compatibility mode must know the final normalization before rounding
    // probabilities. Recompute tiles instead of storing a global score matrix.
    if (rounded) {
        for (int64_t first = 0; first < prefix; first += kScoreTile) {
            FillScores<T, true>(q, cache_k, scores, q_base, kv_head, first, prefix, s, scale);
            __syncthreads();
            if (threadIdx.x == 0) {
                for (int j = 0; j < kScoreTile; ++j)
                    scores[j] = RoundStorage<T>(expf(scores[j] - maximum) / denominator);
            }
            __syncthreads();
            if (value_dimension < s.value_depth) {
                for (int j = 0; j < kScoreTile && first + j < prefix; ++j) {
                    const int64_t offset = (kv_head * s.capacity + first + j) * s.value_depth + value_dimension;
                    value += scores[j] * cuda_detail::ReadDevice(cache_v, offset);
                }
            }
            __syncthreads();
        }
    }
    if (value_dimension < s.value_depth)
        cuda_detail::WriteDevice(out, row * s.value_depth + value_dimension, rounded ? value : value / denominator);
}

template <DataType dtype>
int RunCachedAttention(operators::CachedAttentionParam* p) {
    if (!p || !operators::ValidateCachedAttention(*p, true) || p->q->data_type() != dtype) return -1;
    using T = cuda_detail::StorageT<dtype>;
    const auto& q = p->q->dims();
    AttentionShape s{q[0], q[1], p->k->dims()[1], q[2], q[3], p->v->dims()[3],
                     p->cache_k->dims()[2], cuda_detail::DivUp(p->v->dims()[3], kAttentionThreads)};
    const int64_t rows = s.batch * s.heads * s.tokens;
    if (rows > std::numeric_limits<int>::max() / s.value_tiles) return -1;
    // The nondeferred mode's host position is authoritative. Deferred graph
    // execution may have a newer device value, so never read its host mirror.
    if (!cuda_detail::DeferredHostSyncEnabled()) {
        const int64_t start = p->position->data<int64_t>()[0];
        if (start < 0 || start > s.capacity - s.tokens) return -1;
    }
    cuda_detail::DeviceBuffer<T> dq, dk, dv, cache_k, cache_v, output;
    cuda_detail::DeviceBuffer<int64_t> position;
    if (cuda_detail::CopyTensorToDevice(p->q.get(), &dq) != 0 ||
        cuda_detail::CopyTensorToDevice(p->k.get(), &dk) != 0 ||
        cuda_detail::CopyTensorToDevice(p->v.get(), &dv) != 0 ||
        cuda_detail::CopyTensorToDevice(p->cache_k.get(), &cache_k) != 0 ||
        cuda_detail::CopyTensorToDevice(p->cache_v.get(), &cache_v) != 0 ||
        cuda_detail::CopyTensorToDevice(p->position.get(), &position) != 0 ||
        cuda_detail::AllocateTensorOnDevice(p->out.get(), &output) != 0) return -1;
    // Acquire as outputs only after preserving the existing prefix. This marks
    // their host mirrors stale while retaining the same device allocations.
    if (cuda_detail::AllocateTensorOnDevice(p->cache_k.get(), &cache_k) != 0 ||
        cuda_detail::AllocateTensorOnDevice(p->cache_v.get(), &cache_v) != 0) return -1;
    const int64_t cache_writes = std::max(p->k->numel(), p->v->numel());
    const int write_blocks = static_cast<int>(std::min<int64_t>(65535, cuda_detail::DivUp(cache_writes, kAttentionThreads)));
    WriteAttentionCache<T><<<write_blocks, kAttentionThreads, 0, cuda_detail::InferenceStream()>>>(
        dk.get(), dv.get(), cache_k.get(), cache_v.get(), position.get(), s);
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) return -1;
    const int blocks = static_cast<int>(rows * s.value_tiles);
    const float scale = operators::CachedAttentionScale(*p);
    if (p->round_intermediates) {
        FusedCachedAttention<T, true><<<blocks, kAttentionThreads, 0, cuda_detail::InferenceStream()>>>(
            dq.get(), cache_k.get(), cache_v.get(), position.get(), output.get(), s, scale);
    } else {
        FusedCachedAttention<T, false><<<blocks, kAttentionThreads, 0, cuda_detail::InferenceStream()>>>(
            dq.get(), cache_k.get(), cache_v.get(), position.get(), output.get(), s, scale);
    }
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) return -1;
    if (cuda_detail::DeferredHostSyncEnabled()) return 0;
    if (cuda_detail::CopyDeviceToTensor(&cache_k, p->cache_k.get()) != 0 ||
        cuda_detail::CopyDeviceToTensor(&cache_v, p->cache_v.get()) != 0) return -1;
    return cuda_detail::CopyDeviceToTensor(&output, p->out.get());
}

bool g_cuda_cached_attention_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "CachedAttention",
        [] { return std::make_unique<CachedAttentionKernel<DeviceType::CUDA, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "CachedAttention",
        [] { return std::make_unique<CachedAttentionKernel<DeviceType::CUDA, DataType::FP16>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::BF16, "CachedAttention",
        [] { return std::make_unique<CachedAttentionKernel<DeviceType::CUDA, DataType::BF16>>(); });
    return true;
}();
}  // namespace

template <> int32_t CachedAttentionKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return RunCachedAttention<DataType::FP32>(ParamAs<operators::CachedAttentionParam>());
}
template <> int32_t CachedAttentionKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return RunCachedAttention<DataType::FP16>(ParamAs<operators::CachedAttentionParam>());
}
template <> int32_t CachedAttentionKernel<DeviceType::CUDA, DataType::BF16>::compute() {
    return RunCachedAttention<DataType::BF16>(ParamAs<operators::CachedAttentionParam>());
}
void EnsureCudaCachedAttentionKernelsRegistered() { (void)g_cuda_cached_attention_registered; }
}  // namespace kernel
}  // namespace feather
