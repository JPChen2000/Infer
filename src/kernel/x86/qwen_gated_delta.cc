#include "src/kernel/qwen_gated_delta.h"

#include <immintrin.h>

#include <algorithm>
#include <vector>

#include "util/timer.h"

namespace feather {
namespace kernel {
template <>
int32_t QwenGatedDeltaStateKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::QwenGatedDeltaState::FP32");
    auto* p = static_cast<operators::QwenGatedDeltaStateParam*>(param_);
    if (p == nullptr || p->state == nullptr || p->k == nullptr || p->v == nullptr || p->beta == nullptr ||
        p->decay == nullptr || p->out == nullptr || p->state->data_type() != DataType::FP32 ||
        p->k->data_type() != DataType::FP32 || p->v->data_type() != DataType::FP32 ||
        p->beta->data_type() != DataType::FP32 || p->decay->data_type() != DataType::FP32 ||
        p->state->dims().size() != 4 || p->state->dims()[0] != 1) return -1;
    const int64_t heads = p->state->dims()[1], key = p->state->dims()[2], value = p->state->dims()[3];
    if (heads <= 0 || key <= 0 || value <= 0 || p->out->numel() != p->state->numel() ||
        p->k->numel() != heads * key || p->v->numel() != heads * value || p->beta->numel() != heads || p->decay->numel() != heads) return -1;
    const float* state = p->state->data<float>(); const float* k = p->k->data<float>(); const float* v = p->v->data<float>();
    const float* beta = p->beta->data<float>(); const float* decay = p->decay->data<float>();
    float* out = static_cast<float*>(p->out->raw_data()); p->out->set_data_type(DataType::FP32);
    std::vector<float> kv(static_cast<size_t>(value));
    for (int64_t h = 0; h < heads; ++h) {
        std::fill(kv.begin(), kv.end(), 0.0f);
        for (int64_t j = 0; j < value; j += 8) {
            const int64_t width = std::min<int64_t>(8, value - j);
            __m256 acc = _mm256_setzero_ps();
            for (int64_t i = 0; i < key; ++i) {
                const float scale = decay[h] * k[h * key + i];
                if (width == 8) {
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(state + (h * key + i) * value + j),
                                          _mm256_set1_ps(scale), acc);
                } else {
                    for (int64_t lane = 0; lane < width; ++lane) kv[static_cast<size_t>(j + lane)] +=
                        state[(h * key + i) * value + j + lane] * scale;
                }
            }
            if (width == 8) _mm256_storeu_ps(kv.data() + j, acc);
        }
        for (int64_t i = 0; i < key; ++i) {
            const float ki = k[h * key + i];
            for (int64_t j = 0; j < value; j += 8) {
                const int64_t width = std::min<int64_t>(8, value - j);
                if (width == 8) {
                    const __m256 old = _mm256_loadu_ps(state + (h * key + i) * value + j);
                    const __m256 mem = _mm256_loadu_ps(kv.data() + j);
                    const __m256 vv = _mm256_loadu_ps(v + h * value + j);
                    const __m256 result = _mm256_add_ps(_mm256_mul_ps(old, _mm256_set1_ps(decay[h])),
                        _mm256_mul_ps(_mm256_set1_ps(ki * beta[h]), _mm256_sub_ps(vv, mem)));
                    _mm256_storeu_ps(out + (h * key + i) * value + j, result);
                } else {
                    for (int64_t lane = 0; lane < width; ++lane) {
                        const size_t index = static_cast<size_t>((h * key + i) * value + j + lane);
                        out[index] = state[index] * decay[h] + ki * (v[h * value + j + lane] - kv[static_cast<size_t>(j + lane)]) * beta[h];
                    }
                }
            }
        }
    }
    return 0;
}

template <>
int32_t QwenGatedDeltaOutputKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::QwenGatedDeltaOutput::FP32");
    auto* p = static_cast<operators::QwenGatedDeltaOutputParam*>(param_);
    if (p == nullptr || p->state == nullptr || p->q == nullptr || p->out == nullptr ||
        p->state->data_type() != DataType::FP32 || p->q->data_type() != DataType::FP32 ||
        p->state->dims().size() != 4 || p->state->dims()[0] != 1) return -1;
    const int64_t heads = p->state->dims()[1], key = p->state->dims()[2], value = p->state->dims()[3];
    if (p->q->numel() != heads * key || p->out->numel() != heads * value) return -1;
    const float* state = p->state->data<float>(); const float* q = p->q->data<float>(); float* out = static_cast<float*>(p->out->raw_data());
    p->out->set_data_type(DataType::FP32);
    for (int64_t h = 0; h < heads; ++h) for (int64_t j = 0; j < value; j += 8) {
        const int64_t width = std::min<int64_t>(8, value - j);
        __m256 acc = _mm256_setzero_ps();
        for (int64_t lane = 0; lane < width; ++lane) out[h * value + j + lane] = 0.0f;
        for (int64_t i = 0; i < key; ++i) {
            if (width == 8) {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(state + (h * key + i) * value + j),
                                      _mm256_set1_ps(q[h * key + i]), acc);
            } else {
                for (int64_t lane = 0; lane < width; ++lane) {
                    out[h * value + j + lane] += state[(h * key + i) * value + j + lane] * q[h * key + i];
                }
            }
        }
        if (width == 8) _mm256_storeu_ps(out + h * value + j, acc);
        else if (width > 0) {
            // The scalar tail accumulated directly into the output buffer.
        }
    }
    return 0;
}

void EnsureX86QwenGatedDeltaKernelsRegistered() {
    static bool registered = []() {
        auto& d = KernelDispatcher::instance();
        d.registerKernel(DeviceType::X86, DataType::FP32, "QwenGatedDeltaState", []() {
            return std::make_unique<QwenGatedDeltaStateKernel<DeviceType::X86, DataType::FP32>>();
        });
        d.registerKernel(DeviceType::X86, DataType::FP32, "QwenGatedDeltaOutput", []() {
            return std::make_unique<QwenGatedDeltaOutputKernel<DeviceType::X86, DataType::FP32>>();
        });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
