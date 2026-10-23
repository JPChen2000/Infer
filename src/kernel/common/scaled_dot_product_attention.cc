// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include "src/kernel/common/kernel_io.h"
#include "src/operator/scaled_dot_product_attention_op.h"

namespace feather {
namespace kernel {
namespace {
template <DataType dtype>
using StorageT =
    std::conditional_t<dtype == DataType::FP32, float, std::conditional_t<dtype == DataType::FP16, uint16_t, BFloat16>>;
inline float Read(float value) { return value; }
inline float Read(uint16_t value) { return HalfToFloat(value); }
inline float Read(BFloat16 value) { return BFloat16ToFloat(value.bits); }
template <DataType dtype>
StorageT<dtype> Store(float value) {
    if constexpr (dtype == DataType::FP32)
        return value;
    else if constexpr (dtype == DataType::FP16)
        return FloatToHalf(value);
    else
        return BFloat16{FloatToBFloat16(value)};
}

template <DataType dtype>
int ComputeScaledDotProductAttention(operators::ScaledDotProductAttentionParam* p) {
    if (!p || !operators::ValidateScaledDotProductAttention(*p, true) || p->q->data_type() != dtype) return -1;
    const auto& shape = p->q->dims();
    const int64_t batch = shape[0], heads = shape[1], queries = shape[2], depth = shape[3];
    const int64_t kv_heads = p->k->dims()[1], keys = p->k->dims()[2], vd = p->v->dims()[3];
    const auto* q = p->q->data<StorageT<dtype>>();
    const auto* k = p->k->data<StorageT<dtype>>();
    const auto* v = p->v->data<StorageT<dtype>>();
    auto* out = p->out->mutable_data<StorageT<dtype>>();
    std::vector<float> scores(static_cast<size_t>(keys));
    const float scale = operators::ScaledDotProductAttentionScale(*p);
    for (int64_t b = 0; b < batch; ++b)
        for (int64_t h = 0; h < heads; ++h)
            for (int64_t i = 0; i < queries; ++i) {
                const int64_t row = (b * heads + h) * queries + i, kh = b * kv_heads + h / (heads / kv_heads);
                float maximum = -std::numeric_limits<float>::infinity();
                for (int64_t j = 0; j < keys; ++j) {
                    float score = -std::numeric_limits<float>::infinity();
                    if (!p->causal || j <= i) {
                        score = 0;
                        for (int64_t d = 0; d < depth; ++d)
                            score += Read(q[row * depth + d]) * Read(k[(kh * keys + j) * depth + d]);
                        score *= scale;
                        if (p->mask) {
                            const auto& m = p->mask->dims();
                            score += p->mask->data<float>()[(((m[0] == 1 ? 0 : b) * m[1] + (m[1] == 1 ? 0 : h)) * m[2] +
                                                             (m[2] == 1 ? 0 : i)) *
                                                                m[3] +
                                                            (m[3] == 1 ? 0 : j)];
                        }
                    }
                    scores[j] = score;
                    maximum = std::max(maximum, score);
                }
                float denominator = 0;
                for (int64_t j = 0; j < keys; ++j) {
                    scores[j] = maximum == -std::numeric_limits<float>::infinity() ? 0 : std::exp(scores[j] - maximum);
                    denominator += scores[j];
                }
                for (int64_t d = 0; d < vd; ++d) {
                    float value = 0;
                    for (int64_t j = 0; j < keys; ++j) value += scores[j] * Read(v[(kh * keys + j) * vd + d]);
                    out[row * vd + d] = Store<dtype>(denominator == 0 ? 0 : value / denominator);
                }
            }
    return 0;
}

bool g_common_scaled_dot_product_attention_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP32, "ScaledDotProductAttention", [] {
        return std::make_unique<ScaledDotProductAttentionKernel<DeviceType::COMMON, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP16, "ScaledDotProductAttention", [] {
        return std::make_unique<ScaledDotProductAttentionKernel<DeviceType::COMMON, DataType::FP16>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::BF16, "ScaledDotProductAttention", [] {
        return std::make_unique<ScaledDotProductAttentionKernel<DeviceType::COMMON, DataType::BF16>>();
    });
    return true;
}();
}  // namespace

template <>
int32_t ScaledDotProductAttentionKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    return ComputeScaledDotProductAttention<DataType::FP32>(ParamAs<operators::ScaledDotProductAttentionParam>());
}
template <>
int32_t ScaledDotProductAttentionKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    return ComputeScaledDotProductAttention<DataType::FP16>(ParamAs<operators::ScaledDotProductAttentionParam>());
}
template <>
int32_t ScaledDotProductAttentionKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    return ComputeScaledDotProductAttention<DataType::BF16>(ParamAs<operators::ScaledDotProductAttentionParam>());
}
void EnsureScaledDotProductAttentionKernelsRegistered() { (void)g_common_scaled_dot_product_attention_registered; }
}  // namespace kernel
}  // namespace feather
