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

#include <math_constants.h>

#include <algorithm>
#include <limits>

#include "src/kernel/cuda/kernel_io.cuh"
#include "src/operator/scaled_dot_product_attention_op.h"

namespace feather {
namespace kernel {
namespace {
constexpr int kAttentionThreads = 128;
constexpr int kScoreTile = 32;
struct AttentionShape {
    int64_t batch, heads, kv_heads, queries, keys, depth, vd, value_tiles;
    int64_t mask[4];
    bool causal;
};
// A block handles one query and 128 value channels. The fixed 32-score tile
// keeps scratch independent of sequence length; online softmax rescales values.
template <typename T>
__global__ void Attention(const T* q, const T* k, const T* v, const float* mask, T* out, AttentionShape s,
                          float scale) {
    __shared__ float scores[kScoreTile], maximum, denominator, alpha;
    const int64_t row = static_cast<int64_t>(blockIdx.x) / s.value_tiles;
    const int64_t d = (static_cast<int64_t>(blockIdx.x) % s.value_tiles) * blockDim.x + threadIdx.x;
    const int64_t i = row % s.queries, h = (row / s.queries) % s.heads, b = row / (s.queries * s.heads);
    const int64_t kh = b * s.kv_heads + h / (s.heads / s.kv_heads);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (threadIdx.x == 0) {
        maximum = -CUDART_INF_F;
        denominator = 0;
    }
    __syncthreads();
    float value = 0;
    for (int64_t first = 0; first < s.keys; first += kScoreTile) {
        for (int j = warp; j < kScoreTile; j += kAttentionThreads / 32) {
            const int64_t key = first + j;
            float dot = 0;
            const bool valid = key < s.keys && (!s.causal || key <= i);
            if (valid)
                for (int64_t c = lane; c < s.depth; c += 32)
                    dot += cuda_detail::ReadDevice(q, row * s.depth + c) *
                           cuda_detail::ReadDevice(k, (kh * s.keys + key) * s.depth + c);
            for (int offset = 16; offset > 0; offset /= 2) dot += __shfl_down_sync(0xffffffffu, dot, offset);
            if (lane == 0) {
                float score = valid ? dot * scale : -CUDART_INF_F;
                if (valid && mask)
                    score += mask[(((s.mask[0] == 1 ? 0 : b) * s.mask[1] + (s.mask[1] == 1 ? 0 : h)) * s.mask[2] +
                                   (s.mask[2] == 1 ? 0 : i)) *
                                      s.mask[3] +
                                  (s.mask[3] == 1 ? 0 : key)];
                scores[j] = score;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float tile_max = -CUDART_INF_F;
            for (int j = 0; j < kScoreTile; ++j) tile_max = fmaxf(tile_max, scores[j]);
            const float next = fmaxf(maximum, tile_max);
            alpha = next == -CUDART_INF_F ? 1 : expf(maximum - next);
            float sum = 0;
            for (int j = 0; j < kScoreTile; ++j) {
                scores[j] = next == -CUDART_INF_F ? 0 : expf(scores[j] - next);
                sum += scores[j];
            }
            denominator = denominator * alpha + sum;
            maximum = next;
        }
        __syncthreads();
        if (d < s.vd) {
            value *= alpha;
            for (int j = 0; j < kScoreTile && first + j < s.keys; ++j)
                value += scores[j] * cuda_detail::ReadDevice(v, (kh * s.keys + first + j) * s.vd + d);
        }
        __syncthreads();
    }
    if (d < s.vd) cuda_detail::WriteDevice(out, row * s.vd + d, denominator == 0 ? 0 : value / denominator);
}

template <DataType dtype>
int RunScaledDotProductAttention(operators::ScaledDotProductAttentionParam* p) {
    if (!p || !operators::ValidateScaledDotProductAttention(*p, true) || p->q->data_type() != dtype) return -1;
    using T = cuda_detail::StorageT<dtype>;
    const auto& q = p->q->dims();
    AttentionShape s{q[0],
                     q[1],
                     p->k->dims()[1],
                     q[2],
                     p->k->dims()[2],
                     q[3],
                     p->v->dims()[3],
                     cuda_detail::DivUp(p->v->dims()[3], kAttentionThreads),
                     {1, 1, 1, 1},
                     p->causal};
    if (p->mask)
        for (int i = 0; i < 4; ++i) s.mask[i] = p->mask->dims()[i];
    const int64_t rows = s.batch * s.heads * s.queries;
    if (rows > std::numeric_limits<int>::max() / s.value_tiles) return -1;
    cuda_detail::DeviceBuffer<T> dq, dk, dv, output;
    cuda_detail::DeviceBuffer<float> mask;
    if (cuda_detail::CopyTensorToDevice(p->q.get(), &dq) != 0 ||
        cuda_detail::CopyTensorToDevice(p->k.get(), &dk) != 0 ||
        cuda_detail::CopyTensorToDevice(p->v.get(), &dv) != 0 ||
        cuda_detail::AllocateTensorOnDevice(p->out.get(), &output) != 0)
        return -1;
    if (p->mask && cuda_detail::CopyTensorToDevice(p->mask.get(), &mask) != 0) return -1;
    Attention<T><<<static_cast<int>(rows * s.value_tiles), kAttentionThreads, 0, cuda_detail::InferenceStream()>>>(
        dq.get(), dk.get(), dv.get(), p->mask ? mask.get() : nullptr, output.get(), s,
        operators::ScaledDotProductAttentionScale(*p));
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) return -1;
    return cuda_detail::CopyDeviceToTensor(&output, p->out.get());
}

bool g_cuda_scaled_dot_product_attention_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "ScaledDotProductAttention", [] {
        return std::make_unique<ScaledDotProductAttentionKernel<DeviceType::CUDA, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "ScaledDotProductAttention", [] {
        return std::make_unique<ScaledDotProductAttentionKernel<DeviceType::CUDA, DataType::FP16>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::BF16, "ScaledDotProductAttention", [] {
        return std::make_unique<ScaledDotProductAttentionKernel<DeviceType::CUDA, DataType::BF16>>();
    });
    return true;
}();
}  // namespace

template <>
int32_t ScaledDotProductAttentionKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return RunScaledDotProductAttention<DataType::FP32>(ParamAs<operators::ScaledDotProductAttentionParam>());
}
template <>
int32_t ScaledDotProductAttentionKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return RunScaledDotProductAttention<DataType::FP16>(ParamAs<operators::ScaledDotProductAttentionParam>());
}
template <>
int32_t ScaledDotProductAttentionKernel<DeviceType::CUDA, DataType::BF16>::compute() {
    return RunScaledDotProductAttention<DataType::BF16>(ParamAs<operators::ScaledDotProductAttentionParam>());
}
void EnsureCudaScaledDotProductAttentionKernelsRegistered() { (void)g_cuda_scaled_dot_product_attention_registered; }
}  // namespace kernel
}  // namespace feather
