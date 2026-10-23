// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#include "src/operator/cached_attention_op.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include "src/kernel/common/kernel_io.h"

namespace feather {
namespace kernel {
namespace {
template <DataType dtype>
using StorageT = std::conditional_t<dtype == DataType::FP32, float,
                 std::conditional_t<dtype == DataType::FP16, uint16_t, BFloat16>>;
inline float Read(float value) { return value; }
inline float Read(uint16_t value) { return HalfToFloat(value); }
inline float Read(BFloat16 value) { return BFloat16ToFloat(value.bits); }
template <DataType dtype> StorageT<dtype> Store(float value) {
    if constexpr (dtype == DataType::FP32) return value;
    else if constexpr (dtype == DataType::FP16) return FloatToHalf(value);
    else return BFloat16{FloatToBFloat16(value)};
}

template <DataType dtype>
int ComputeCachedAttention(operators::CachedAttentionParam* p) {
    if (!p || !operators::ValidateCachedAttention(*p, true) || p->q->data_type() != dtype) return -1;
    const int64_t position = p->position->data<int64_t>()[0];
    const auto& qd = p->q->dims();
    const int64_t batch = qd[0], heads = qd[1], tokens = qd[2], depth = qd[3];
    const int64_t kv_heads = p->k->dims()[1], value_depth = p->v->dims()[3];
    const int64_t capacity = p->cache_k->dims()[2];
    if (position < 0 || position > capacity - tokens) return -1;
    using T = StorageT<dtype>;
    const T* q = p->q->data<T>();
    const T* k = p->k->data<T>();
    const T* v = p->v->data<T>();
    T* cache_k = p->cache_k->mutable_data<T>();
    T* cache_v = p->cache_v->mutable_data<T>();
    T* out = p->out->mutable_data<T>();
    for (int64_t head = 0; head < batch * kv_heads; ++head) {
        std::memcpy(cache_k + (head * capacity + position) * depth,
                    k + head * tokens * depth, static_cast<size_t>(tokens * depth) * sizeof(T));
        std::memcpy(cache_v + (head * capacity + position) * value_depth,
                    v + head * tokens * value_depth, static_cast<size_t>(tokens * value_depth) * sizeof(T));
    }
    const float scale = operators::CachedAttentionScale(*p);
    // Only a single row of reference scores is materialized. CUDA uses tiles
    // in shared memory and never allocates the score matrix.
    std::vector<float> scores(static_cast<size_t>(position + tokens));
    std::vector<float> values(static_cast<size_t>(value_depth));
    const int64_t group = heads / kv_heads;
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h = 0; h < heads; ++h) {
            const int64_t kv_head = b * kv_heads + h / group;
            for (int64_t t = 0; t < tokens; ++t) {
                const int64_t q_base = ((b * heads + h) * tokens + t) * depth;
                const int64_t prefix = position + t + 1;
                float maximum = -std::numeric_limits<float>::infinity();
                for (int64_t j = 0; j < prefix; ++j) {
                    float dot = 0.0f;
                    const int64_t key_base = (kv_head * capacity + j) * depth;
                    for (int64_t d = 0; d < depth; ++d) dot += Read(q[q_base + d]) * Read(cache_k[key_base + d]);
                    if (p->round_intermediates) dot = Read(Store<dtype>(dot));
                    scores[static_cast<size_t>(j)] = dot * scale;
                    maximum = std::max(maximum, scores[static_cast<size_t>(j)]);
                }
                float denominator = 0.0f;
                for (int64_t j = 0; j < prefix; ++j) {
                    scores[static_cast<size_t>(j)] = std::exp(scores[static_cast<size_t>(j)] - maximum);
                    denominator += scores[static_cast<size_t>(j)];
                }
                std::fill(values.begin(), values.end(), 0.0f);
                for (int64_t j = 0; j < prefix; ++j) {
                    float probability = scores[static_cast<size_t>(j)] / denominator;
                    if (p->round_intermediates) probability = Read(Store<dtype>(probability));
                    const int64_t value_base = (kv_head * capacity + j) * value_depth;
                    for (int64_t d = 0; d < value_depth; ++d)
                        values[static_cast<size_t>(d)] += probability * Read(cache_v[value_base + d]);
                }
                const int64_t out_base = ((b * heads + h) * tokens + t) * value_depth;
                for (int64_t d = 0; d < value_depth; ++d) out[out_base + d] = Store<dtype>(values[static_cast<size_t>(d)]);
            }
        }
    }
    return 0;
}

bool g_common_cached_attention_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP32, "CachedAttention",
        [] { return std::make_unique<CachedAttentionKernel<DeviceType::COMMON, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP16, "CachedAttention",
        [] { return std::make_unique<CachedAttentionKernel<DeviceType::COMMON, DataType::FP16>>(); });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::BF16, "CachedAttention",
        [] { return std::make_unique<CachedAttentionKernel<DeviceType::COMMON, DataType::BF16>>(); });
    return true;
}();
}  // namespace

template <> int32_t CachedAttentionKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    return ComputeCachedAttention<DataType::FP32>(ParamAs<operators::CachedAttentionParam>());
}
template <> int32_t CachedAttentionKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    return ComputeCachedAttention<DataType::FP16>(ParamAs<operators::CachedAttentionParam>());
}
template <> int32_t CachedAttentionKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    return ComputeCachedAttention<DataType::BF16>(ParamAs<operators::CachedAttentionParam>());
}
void EnsureCachedAttentionKernelsRegistered() { (void)g_common_cached_attention_registered; }
}  // namespace kernel
}  // namespace feather
