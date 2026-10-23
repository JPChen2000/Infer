// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0

#include "src/kernel/qwen_gated_delta_sequence.h"

#include <algorithm>
#include <vector>

#include "util/timer.h"

namespace feather {
namespace kernel {

template <>
int32_t QwenGatedDeltaSequenceKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::QwenGatedDeltaSequence::FP32");
    auto* p = ParamAs<operators::QwenGatedDeltaSequenceParam>();
    if (!operators::qwen_sequence_detail::Validate(p, true)) return -1;
    const int64_t sequence = p->k->dims()[1], heads = p->state->dims()[1];
    const int64_t key = p->state->dims()[2], value = p->state->dims()[3];
    const float* initial_state = p->state->data<float>();
    const float* k = p->k->data<float>();
    const float* q = p->q->data<float>();
    const float* v = p->v->data<float>();
    const float* beta = p->beta->data<float>();
    const float* decay = p->decay->data<float>();
    float* next_state = p->next_state->mutable_data<float>();
    float* out = p->out->mutable_data<float>();
    std::vector<float> kv(static_cast<size_t>(value));
    for (int64_t t = 0; t < sequence; ++t) {
        const float* state = t == 0 ? initial_state : next_state;
        for (int64_t h = 0; h < heads; ++h) {
            const int64_t token_head = t * heads + h;
            std::fill(kv.begin(), kv.end(), 0.0f);
            std::fill(out + token_head * value, out + (token_head + 1) * value, 0.0f);
            // Match the single-token Common reduction and multiplication order.
            for (int64_t i = 0; i < key; ++i) {
                const float scale = decay[token_head] * k[token_head * key + i];
                for (int64_t j = 0; j < value; ++j) {
                    kv[j] += state[(h * key + i) * value + j] * scale;
                }
            }
            for (int64_t i = 0; i < key; ++i) {
                const float ki = k[token_head * key + i], qi = q[token_head * key + i];
                for (int64_t j = 0; j < value; ++j) {
                    const int64_t index = (h * key + i) * value + j;
                    const float updated = state[index] * decay[token_head] +
                        ki * (v[token_head * value + j] - kv[j]) * beta[token_head];
                    next_state[index] = updated;
                    out[token_head * value + j] += updated * qi;
                }
            }
        }
    }
    return 0;
}

void EnsureCommonQwenGatedDeltaSequenceKernelsRegistered() {
    static const bool registered = [] {
        KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32,
            "QwenGatedDeltaSequence", [] {
                return std::make_unique<QwenGatedDeltaSequenceKernel<DeviceType::COMMON, DataType::FP32>>();
            });
        return true;
    }();
    (void)registered;
}

void EnsureQwenGatedDeltaSequenceKernelsRegistered() {
    EnsureCommonQwenGatedDeltaSequenceKernelsRegistered();
#ifdef FEATHER_WITH_CUDA
    EnsureCudaQwenGatedDeltaSequenceKernelsRegistered();
#endif
}
} // namespace kernel
} // namespace feather
