#include "src/kernel/qwen_gated_delta.h"

#include <immintrin.h>

#include <algorithm>
#include <cstdlib>
#include <vector>

#include "util/threading.h"
#include "util/timer.h"

#if defined(FEATHER_WITH_OPENMP)
#include <omp.h>
#endif

namespace feather {
namespace kernel {

namespace {

constexpr int64_t kQwenGatedDeltaParallelMinimumElements = 65536;
constexpr size_t kQwenDefaultWorkerLimit = 4;

size_t QwenGatedDeltaWorkerCount(int64_t heads) {
    if (heads <= 1) {
        return 1;
    }

    // Decode uses this FP32 state update alongside BF16 projections. Honor the
    // same process-level cap so one setting controls the entire decode path.
    size_t worker_limit = std::min(DefaultThreadCount(), kQwenDefaultWorkerLimit);
    const char* configured_limit = std::getenv("FEATHER_X86_BF16_THREADS");
    if (configured_limit != nullptr && configured_limit[0] != '\0') {
        char* end = nullptr;
        const unsigned long parsed_limit = std::strtoul(configured_limit, &end, 10);
        if (end != configured_limit && *end == '\0' && parsed_limit > 0) {
            worker_limit = std::min(worker_limit, static_cast<size_t>(parsed_limit));
        }
    }
#if defined(FEATHER_WITH_OPENMP)
    const int openmp_limit = omp_get_max_threads();
    if (openmp_limit > 0) {
        worker_limit = std::min(worker_limit, static_cast<size_t>(openmp_limit));
    }
#endif
    return std::min(worker_limit, static_cast<size_t>(heads));
}

void ComputeGatedDeltaStateHead(const float* state, const float* k, const float* v, const float* beta,
                                const float* decay, int64_t head, int64_t key, int64_t value, float* kv,
                                float* out) {
    std::fill(kv, kv + value, 0.0f);
    for (int64_t j = 0; j < value; j += 8) {
        const int64_t width = std::min<int64_t>(8, value - j);
        __m256 acc = _mm256_setzero_ps();
        for (int64_t i = 0; i < key; ++i) {
            const float scale = decay[head] * k[head * key + i];
            if (width == 8) {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(state + (head * key + i) * value + j),
                                      _mm256_set1_ps(scale), acc);
            } else {
                for (int64_t lane = 0; lane < width; ++lane) {
                    kv[j + lane] += state[(head * key + i) * value + j + lane] * scale;
                }
            }
        }
        if (width == 8) {
            _mm256_storeu_ps(kv + j, acc);
        }
    }

    for (int64_t i = 0; i < key; ++i) {
        const float ki = k[head * key + i];
        for (int64_t j = 0; j < value; j += 8) {
            const int64_t width = std::min<int64_t>(8, value - j);
            if (width == 8) {
                const __m256 old = _mm256_loadu_ps(state + (head * key + i) * value + j);
                const __m256 mem = _mm256_loadu_ps(kv + j);
                const __m256 vv = _mm256_loadu_ps(v + head * value + j);
                const __m256 result = _mm256_add_ps(
                    _mm256_mul_ps(old, _mm256_set1_ps(decay[head])),
                    _mm256_mul_ps(_mm256_set1_ps(ki * beta[head]), _mm256_sub_ps(vv, mem)));
                _mm256_storeu_ps(out + (head * key + i) * value + j, result);
            } else {
                for (int64_t lane = 0; lane < width; ++lane) {
                    const size_t index = static_cast<size_t>((head * key + i) * value + j + lane);
                    out[index] = state[index] * decay[head] +
                                 ki * (v[head * value + j + lane] - kv[j + lane]) * beta[head];
                }
            }
        }
    }
}

void ComputeGatedDeltaStateOutputHead(const float* state, const float* k, const float* v, const float* beta,
                                      const float* decay, const float* q, int64_t head, int64_t key, int64_t value,
                                      float* kv, float* next_state, float* out) {
    std::fill(kv, kv + value, 0.0f);
    for (int64_t j = 0; j < value; j += 8) {
        const int64_t width = std::min<int64_t>(8, value - j);
        __m256 acc = _mm256_setzero_ps();
        for (int64_t i = 0; i < key; ++i) {
            const float scale = decay[head] * k[head * key + i];
            if (width == 8) {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(state + (head * key + i) * value + j),
                                      _mm256_set1_ps(scale), acc);
            } else {
                for (int64_t lane = 0; lane < width; ++lane) {
                    kv[j + lane] += state[(head * key + i) * value + j + lane] * scale;
                }
            }
        }
        if (width == 8) {
            _mm256_storeu_ps(kv + j, acc);
        }
    }

    float* head_out = out + head * value;
    std::fill(head_out, head_out + value, 0.0f);
    for (int64_t i = 0; i < key; ++i) {
        const float ki = k[head * key + i];
        const __m256 q_value = _mm256_set1_ps(q[head * key + i]);
        for (int64_t j = 0; j < value; j += 8) {
            const int64_t width = std::min<int64_t>(8, value - j);
            if (width == 8) {
                const __m256 old = _mm256_loadu_ps(state + (head * key + i) * value + j);
                const __m256 mem = _mm256_loadu_ps(kv + j);
                const __m256 vv = _mm256_loadu_ps(v + head * value + j);
                const __m256 result = _mm256_add_ps(
                    _mm256_mul_ps(old, _mm256_set1_ps(decay[head])),
                    _mm256_mul_ps(_mm256_set1_ps(ki * beta[head]), _mm256_sub_ps(vv, mem)));
                _mm256_storeu_ps(next_state + (head * key + i) * value + j, result);
                _mm256_storeu_ps(head_out + j,
                                 _mm256_fmadd_ps(result, q_value, _mm256_loadu_ps(head_out + j)));
            } else {
                for (int64_t lane = 0; lane < width; ++lane) {
                    const size_t index = static_cast<size_t>((head * key + i) * value + j + lane);
                    const float updated = state[index] * decay[head] +
                                          ki * (v[head * value + j + lane] - kv[j + lane]) * beta[head];
                    next_state[index] = updated;
                    head_out[j + lane] += updated * q[head * key + i];
                }
            }
        }
    }
}

bool ValidateCombined(const operators::QwenGatedDeltaParam* p, int64_t* heads, int64_t* key, int64_t* value) {
    if (p == nullptr || p->state == nullptr || p->k == nullptr || p->v == nullptr || p->beta == nullptr ||
        p->decay == nullptr || p->q == nullptr || p->next_state == nullptr || p->out == nullptr ||
        p->state->data_type() != DataType::FP32 || p->k->data_type() != DataType::FP32 ||
        p->v->data_type() != DataType::FP32 || p->beta->data_type() != DataType::FP32 ||
        p->decay->data_type() != DataType::FP32 || p->q->data_type() != DataType::FP32 ||
        p->state->dims().size() != 4 || p->state->dims()[0] != 1) {
        return false;
    }
    const auto& dims = p->state->dims();
    if (dims[1] <= 0 || dims[2] <= 0 || dims[3] <= 0 || p->next_state->numel() != p->state->numel() ||
        p->out->numel() != dims[1] * dims[3] || p->k->numel() != dims[1] * dims[2] ||
        p->v->numel() != dims[1] * dims[3] || p->beta->numel() != dims[1] ||
        p->decay->numel() != dims[1] || p->q->numel() != dims[1] * dims[2]) {
        return false;
    }
    *heads = dims[1];
    *key = dims[2];
    *value = dims[3];
    return true;
}

}  // namespace

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
    const size_t worker_count = QwenGatedDeltaWorkerCount(heads);
#if !defined(FEATHER_WITH_OPENMP)
    (void)worker_count;
#endif
#if defined(FEATHER_WITH_OPENMP)
    if (worker_count > 1 && p->state->numel() >= kQwenGatedDeltaParallelMinimumElements && !omp_in_parallel()) {
#pragma omp parallel num_threads(static_cast<int>(worker_count))
        {
            std::vector<float> kv(static_cast<size_t>(value));
#pragma omp for schedule(static)
            for (int64_t head = 0; head < heads; ++head) {
                ComputeGatedDeltaStateHead(state, k, v, beta, decay, head, key, value, kv.data(), out);
            }
        }
        return 0;
    }
#endif

    std::vector<float> kv(static_cast<size_t>(value));
    for (int64_t head = 0; head < heads; ++head) {
        ComputeGatedDeltaStateHead(state, k, v, beta, decay, head, key, value, kv.data(), out);
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

template <>
int32_t QwenGatedDeltaKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::QwenGatedDelta::FP32");
    auto* p = static_cast<operators::QwenGatedDeltaParam*>(param_);
    int64_t heads = 0, key = 0, value = 0;
    if (!ValidateCombined(p, &heads, &key, &value)) return -1;

    const float* state = p->state->data<float>();
    const float* k = p->k->data<float>();
    const float* v = p->v->data<float>();
    const float* beta = p->beta->data<float>();
    const float* decay = p->decay->data<float>();
    const float* q = p->q->data<float>();
    float* next_state = static_cast<float*>(p->next_state->raw_data());
    float* out = static_cast<float*>(p->out->raw_data());
    p->next_state->set_data_type(DataType::FP32);
    p->out->set_data_type(DataType::FP32);
    const size_t worker_count = QwenGatedDeltaWorkerCount(heads);
#if !defined(FEATHER_WITH_OPENMP)
    (void)worker_count;
#endif

#if defined(FEATHER_WITH_OPENMP)
    if (worker_count > 1 && p->state->numel() >= kQwenGatedDeltaParallelMinimumElements && !omp_in_parallel()) {
#pragma omp parallel num_threads(static_cast<int>(worker_count))
        {
            std::vector<float> kv(static_cast<size_t>(value));
#pragma omp for schedule(static)
            for (int64_t head = 0; head < heads; ++head) {
                ComputeGatedDeltaStateOutputHead(state, k, v, beta, decay, q, head, key, value, kv.data(),
                                                 next_state, out);
            }
        }
        return 0;
    }
#endif

    std::vector<float> kv(static_cast<size_t>(value));
    for (int64_t head = 0; head < heads; ++head) {
        ComputeGatedDeltaStateOutputHead(state, k, v, beta, decay, q, head, key, value, kv.data(), next_state, out);
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
        d.registerKernel(DeviceType::X86, DataType::FP32, "QwenGatedDelta", []() {
            return std::make_unique<QwenGatedDeltaKernel<DeviceType::X86, DataType::FP32>>();
        });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
