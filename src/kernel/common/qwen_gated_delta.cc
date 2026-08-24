#include "src/kernel/qwen_gated_delta.h"

#include <algorithm>
#include <vector>

#include "util/timer.h"

namespace feather {
namespace kernel {
namespace {

bool g_registered = []() {
    auto& d = KernelDispatcher::instance();
    d.registerKernel(DeviceType::COMMON, DataType::FP32, "QwenGatedDeltaState", []() {
        return std::make_unique<QwenGatedDeltaStateKernel<DeviceType::COMMON, DataType::FP32>>();
    });
    d.registerKernel(DeviceType::COMMON, DataType::FP32, "QwenGatedDeltaOutput", []() {
        return std::make_unique<QwenGatedDeltaOutputKernel<DeviceType::COMMON, DataType::FP32>>();
    });
    d.registerKernel(DeviceType::COMMON, DataType::FP32, "QwenGatedDelta", []() {
        return std::make_unique<QwenGatedDeltaKernel<DeviceType::COMMON, DataType::FP32>>();
    });
    return true;
}();

bool ValidateState(const operators::QwenGatedDeltaStateParam* p, int64_t* heads, int64_t* key, int64_t* value) {
    if (p == nullptr || p->state == nullptr || p->k == nullptr || p->v == nullptr || p->beta == nullptr ||
        p->decay == nullptr || p->out == nullptr || p->state->data_type() != DataType::FP32 ||
        p->k->data_type() != DataType::FP32 || p->v->data_type() != DataType::FP32 ||
        p->beta->data_type() != DataType::FP32 || p->decay->data_type() != DataType::FP32) return false;
    const auto& d = p->state->dims();
    if (d.size() != 4 || d[0] != 1 || d[1] <= 0 || d[2] <= 0 || d[3] <= 0 ||
        p->out->numel() != p->state->numel() || p->k->numel() != d[1] * d[2] || p->v->numel() != d[1] * d[3] ||
        p->beta->numel() != d[1] || p->decay->numel() != d[1]) return false;
    *heads = d[1]; *key = d[2]; *value = d[3];
    return true;
}

bool ValidateCombined(const operators::QwenGatedDeltaParam* p, int64_t* heads, int64_t* key, int64_t* value) {
    if (p == nullptr || p->state == nullptr || p->k == nullptr || p->v == nullptr || p->beta == nullptr ||
        p->decay == nullptr || p->q == nullptr || p->next_state == nullptr || p->out == nullptr ||
        p->state->data_type() != DataType::FP32 || p->k->data_type() != DataType::FP32 ||
        p->v->data_type() != DataType::FP32 || p->beta->data_type() != DataType::FP32 ||
        p->decay->data_type() != DataType::FP32 || p->q->data_type() != DataType::FP32) {
        return false;
    }
    const auto& d = p->state->dims();
    if (d.size() != 4 || d[0] != 1 || d[1] <= 0 || d[2] <= 0 || d[3] <= 0 ||
        p->next_state->numel() != p->state->numel() || p->out->numel() != d[1] * d[3] ||
        p->k->numel() != d[1] * d[2] || p->v->numel() != d[1] * d[3] ||
        p->beta->numel() != d[1] || p->decay->numel() != d[1] || p->q->numel() != d[1] * d[2]) {
        return false;
    }
    *heads = d[1];
    *key = d[2];
    *value = d[3];
    return true;
}

}  // namespace

template <>
int32_t QwenGatedDeltaStateKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::QwenGatedDeltaState::FP32");
    auto* p = static_cast<operators::QwenGatedDeltaStateParam*>(param_);
    int64_t heads = 0, key = 0, value = 0;
    if (!ValidateState(p, &heads, &key, &value)) return -1;
    const float* state = p->state->data<float>();
    const float* k = p->k->data<float>();
    const float* v = p->v->data<float>();
    const float* beta = p->beta->data<float>();
    const float* decay = p->decay->data<float>();
    float* out = static_cast<float*>(p->out->raw_data());
    p->out->set_data_type(DataType::FP32);
    std::vector<float> kv(static_cast<size_t>(value));
    for (int64_t h = 0; h < heads; ++h) {
        std::fill(kv.begin(), kv.end(), 0.0f);
        for (int64_t i = 0; i < key; ++i) {
            const float scale = decay[h] * k[h * key + i];
            for (int64_t j = 0; j < value; ++j) kv[j] += state[(h * key + i) * value + j] * scale;
        }
        for (int64_t i = 0; i < key; ++i) {
            const float k_value = k[h * key + i];
            for (int64_t j = 0; j < value; ++j) {
                const size_t index = static_cast<size_t>((h * key + i) * value + j);
                out[index] = state[index] * decay[h] + k_value * (v[h * value + j] - kv[j]) * beta[h];
            }
        }
    }
    return 0;
}

template <>
int32_t QwenGatedDeltaOutputKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::QwenGatedDeltaOutput::FP32");
    auto* p = static_cast<operators::QwenGatedDeltaOutputParam*>(param_);
    if (p == nullptr || p->state == nullptr || p->q == nullptr || p->out == nullptr ||
        p->state->data_type() != DataType::FP32 || p->q->data_type() != DataType::FP32 ||
        p->out->data_type() != DataType::FP32 || p->state->dims().size() != 4 || p->q->numel() != p->state->dims()[1] * p->state->dims()[2]) return -1;
    const int64_t heads = p->state->dims()[1], key = p->state->dims()[2], value = p->state->dims()[3];
    if (p->out->numel() != heads * value) return -1;
    const float* state = p->state->data<float>(); const float* q = p->q->data<float>(); float* out = static_cast<float*>(p->out->raw_data());
    p->out->set_data_type(DataType::FP32);
    for (int64_t h = 0; h < heads; ++h) for (int64_t j = 0; j < value; ++j) {
        float sum = 0.0f;
        for (int64_t i = 0; i < key; ++i) sum += state[(h * key + i) * value + j] * q[h * key + i];
        out[h * value + j] = sum;
    }
    return 0;
}

template <>
int32_t QwenGatedDeltaKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::QwenGatedDelta::FP32");
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

    std::vector<float> kv(static_cast<size_t>(value));
    for (int64_t h = 0; h < heads; ++h) {
        std::fill(kv.begin(), kv.end(), 0.0f);
        std::fill(out + h * value, out + (h + 1) * value, 0.0f);
        for (int64_t i = 0; i < key; ++i) {
            const float scale = decay[h] * k[h * key + i];
            for (int64_t j = 0; j < value; ++j) {
                kv[j] += state[(h * key + i) * value + j] * scale;
            }
        }
        for (int64_t i = 0; i < key; ++i) {
            const float k_value = k[h * key + i];
            const float q_value = q[h * key + i];
            for (int64_t j = 0; j < value; ++j) {
                const size_t index = static_cast<size_t>((h * key + i) * value + j);
                const float updated = state[index] * decay[h] +
                                      k_value * (v[h * value + j] - kv[j]) * beta[h];
                next_state[index] = updated;
                out[h * value + j] += updated * q_value;
            }
        }
    }
    return 0;
}

void EnsureCommonQwenGatedDeltaKernelsRegistered() { (void)g_registered; }
void EnsureQwenGatedDeltaKernelsRegistered() {
    EnsureCommonQwenGatedDeltaKernelsRegistered();
    EnsureX86QwenGatedDeltaKernelsRegistered();
#ifdef FEATHER_WITH_CUDA
    EnsureCudaQwenGatedDeltaKernelsRegistered();
#endif
}

}  // namespace kernel
}  // namespace feather
