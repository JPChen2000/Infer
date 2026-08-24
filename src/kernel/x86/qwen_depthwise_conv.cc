#include "src/kernel/qwen_depthwise_conv.h"

#include <algorithm>
#include <cstdlib>

#include "util/bf16.h"
#include "util/threading.h"
#include "util/timer.h"

#if defined(FEATHER_WITH_OPENMP)
#include <omp.h>
#endif

namespace feather {
namespace kernel {
namespace {

constexpr int64_t kQwenDepthwiseConvParallelMinimumChannels = 2048;
constexpr size_t kQwenDefaultWorkerLimit = 4;

bool HasShape(const std::shared_ptr<Tensor>& tensor, const std::vector<int64_t>& shape) {
    return tensor != nullptr && tensor->dims().data() == shape;
}

bool Validate(const operators::QwenDepthwiseConvStateParam* param, int64_t* channels) {
    if (param == nullptr || channels == nullptr || param->state == nullptr || param->mixed == nullptr ||
        param->weight == nullptr || param->conv_out == nullptr || param->discarded_prefix == nullptr ||
        param->next_state == nullptr || !param->state->IsInitialized() || !param->mixed->IsInitialized() ||
        !param->weight->IsInitialized() || !param->conv_out->IsInitialized() ||
        !param->discarded_prefix->IsInitialized() || !param->next_state->IsInitialized() ||
        param->state->data_type() != DataType::BF16 || param->mixed->data_type() != DataType::BF16 ||
        param->weight->data_type() != DataType::BF16 || param->conv_out->data_type() != DataType::BF16 ||
        param->discarded_prefix->data_type() != DataType::BF16 || param->next_state->data_type() != DataType::BF16 ||
        param->state->dims().size() != 3) {
        return false;
    }

    const int64_t c = param->state->dims()[1];
    if (c <= 0 || !HasShape(param->state, {1, c, 3}) || !HasShape(param->mixed, {1, c, 1}) ||
        !HasShape(param->weight, {c, 1, 1, 4}) || !HasShape(param->conv_out, {1, c, 1, 1}) ||
        !HasShape(param->discarded_prefix, {1, c, 1}) || !HasShape(param->next_state, {1, c, 3})) {
        return false;
    }

    const size_t bf16_bytes = sizeof(uint16_t);
    return param->state->memory_size() >= static_cast<size_t>(param->state->numel()) * bf16_bytes &&
           param->mixed->memory_size() >= static_cast<size_t>(param->mixed->numel()) * bf16_bytes &&
           param->weight->memory_size() >= static_cast<size_t>(param->weight->numel()) * bf16_bytes &&
           param->conv_out->memory_size() >= static_cast<size_t>(param->conv_out->numel()) * bf16_bytes &&
           param->discarded_prefix->memory_size() >=
               static_cast<size_t>(param->discarded_prefix->numel()) * bf16_bytes &&
           param->next_state->memory_size() >= static_cast<size_t>(param->next_state->numel()) * bf16_bytes &&
           ((*channels = c), true);
}

size_t WorkerCount(int64_t channels) {
    if (channels < kQwenDepthwiseConvParallelMinimumChannels) {
        return 1;
    }
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
    return std::min(worker_limit, static_cast<size_t>(channels));
}

inline void ComputeChannel(const uint16_t* state, const uint16_t* mixed, const uint16_t* weight, int64_t channel,
                           uint16_t* conv_out, uint16_t* discarded_prefix, uint16_t* next_state) {
    const int64_t state_offset = channel * 3;
    const int64_t weight_offset = channel * 4;
    const uint16_t state0 = state[state_offset];
    const uint16_t state1 = state[state_offset + 1];
    const uint16_t state2 = state[state_offset + 2];
    const uint16_t mixed_value = mixed[channel];

    float value = BFloat16ToFloat(state0) * BFloat16ToFloat(weight[weight_offset]);
    value += BFloat16ToFloat(state1) * BFloat16ToFloat(weight[weight_offset + 1]);
    value += BFloat16ToFloat(state2) * BFloat16ToFloat(weight[weight_offset + 2]);
    value += BFloat16ToFloat(mixed_value) * BFloat16ToFloat(weight[weight_offset + 3]);
    conv_out[channel] = FloatToBFloat16(value);

    discarded_prefix[channel] = state0;
    next_state[state_offset] = state1;
    next_state[state_offset + 1] = state2;
    next_state[state_offset + 2] = mixed_value;
}

void ComputeRange(const uint16_t* state, const uint16_t* mixed, const uint16_t* weight, int64_t begin,
                  int64_t end, uint16_t* conv_out, uint16_t* discarded_prefix, uint16_t* next_state) {
    for (int64_t channel = begin; channel < end; ++channel) {
        ComputeChannel(state, mixed, weight, channel, conv_out, discarded_prefix, next_state);
    }
}

}  // namespace

template <>
int32_t QwenDepthwiseConvStateKernel<DeviceType::X86, DataType::BF16>::compute() {
    AutoTimer timer("X86::QwenDepthwiseConvState::BF16");
    auto* param = static_cast<operators::QwenDepthwiseConvStateParam*>(param_);
    int64_t channels = 0;
    if (!Validate(param, &channels)) {
        return -1;
    }

    const auto* state = static_cast<const uint16_t*>(param->state->raw_data());
    const auto* mixed = static_cast<const uint16_t*>(param->mixed->raw_data());
    const auto* weight = static_cast<const uint16_t*>(param->weight->raw_data());
    auto* conv_out = static_cast<uint16_t*>(param->conv_out->raw_data());
    auto* discarded_prefix = static_cast<uint16_t*>(param->discarded_prefix->raw_data());
    auto* next_state = static_cast<uint16_t*>(param->next_state->raw_data());

    const size_t worker_count = WorkerCount(channels);
#if !defined(FEATHER_WITH_OPENMP)
    (void)worker_count;
#endif
#if defined(FEATHER_WITH_OPENMP)
    if (worker_count > 1 && !omp_in_parallel()) {
        const int omp_worker_count = static_cast<int>(worker_count);
#pragma omp parallel for schedule(static) num_threads(omp_worker_count)
        for (int64_t channel = 0; channel < channels; ++channel) {
            ComputeChannel(state, mixed, weight, channel, conv_out, discarded_prefix, next_state);
        }
        return 0;
    }
#endif

    ComputeRange(state, mixed, weight, 0, channels, conv_out, discarded_prefix, next_state);
    return 0;
}

void EnsureX86QwenDepthwiseConvStateKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(DeviceType::X86, DataType::BF16, "QwenDepthwiseConvState", []() {
            return std::make_unique<QwenDepthwiseConvStateKernel<DeviceType::X86, DataType::BF16>>();
        });
        return true;
    }();
    (void)registered;
}

void EnsureQwenDepthwiseConvStateKernelsRegistered() {
    EnsureX86QwenDepthwiseConvStateKernelsRegistered();
#ifdef FEATHER_WITH_CUDA
    EnsureCudaQwenDepthwiseConvStateKernelsRegistered();
#endif
}

}  // namespace kernel
}  // namespace feather
