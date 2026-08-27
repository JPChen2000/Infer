#include "src/kernel/qwen_depthwise_conv.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>

#include "util/bf16.h"
#include "util/fp8.h"
#include "util/threading.h"
#include "util/timer.h"
#include "src/kernel/x86/fp8_utils.h"

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

bool IsValidScale(float scale) { return std::isfinite(scale) && scale > 0.0f; }

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

template <DataType dtype>
const std::array<float, 256>& Fp8DecodeTable() {
    static const std::array<float, 256> table = []() {
        std::array<float, 256> values{};
        for (size_t index = 0; index < values.size(); ++index) {
            values[index] = dtype == DataType::FP8E4M3
                                ? Fp8E4M3ToFloat(static_cast<uint8_t>(index))
                                : Fp8E5M2ToFloat(static_cast<uint8_t>(index));
        }
        return values;
    }();
    return table;
}

template <DataType dtype>
inline uint8_t EncodeFp8(float value) {
    return x86::EncodeFp8ForX86<dtype>(value);
}

template <DataType dtype>
bool ValidateFp8(const operators::QwenDepthwiseConvStateParam* param, int64_t* channels) {
    if (param == nullptr || channels == nullptr || param->fp8_dtype != dtype || param->state == nullptr ||
        param->mixed == nullptr || param->weight == nullptr || param->conv_out == nullptr ||
        param->discarded_prefix == nullptr || param->next_state == nullptr || !param->state->IsInitialized() ||
        !param->mixed->IsInitialized() || !param->weight->IsInitialized() || !param->conv_out->IsInitialized() ||
        !param->discarded_prefix->IsInitialized() || !param->next_state->IsInitialized() ||
        param->state->data_type() != DataType::BF16 || param->mixed->data_type() != DataType::BF16 ||
        param->weight->data_type() != dtype || param->conv_out->data_type() != DataType::BF16 ||
        param->discarded_prefix->data_type() != DataType::BF16 || param->next_state->data_type() != DataType::BF16 ||
        !HasCompatiblePerTensorQuantization(param->weight->quantization()) ||
        !IsValidScale(param->weight->quantization_scale()) || !IsValidScale(param->fp8_input_scale) ||
        !IsValidScale(param->fp8_output_scale) || param->state->dims().size() != 3) {
        return false;
    }

    const int64_t c = param->state->dims()[1];
    if (c <= 0 || !HasShape(param->state, {1, c, 3}) || !HasShape(param->mixed, {1, c, 1}) ||
        !HasShape(param->weight, {c, 1, 1, 4}) || !HasShape(param->conv_out, {1, c, 1, 1}) ||
        !HasShape(param->discarded_prefix, {1, c, 1}) || !HasShape(param->next_state, {1, c, 3})) {
        return false;
    }

    const size_t bf16_bytes = sizeof(uint16_t);
    const size_t fp8_bytes = sizeof(uint8_t);
    return param->state->memory_size() >= static_cast<size_t>(param->state->numel()) * bf16_bytes &&
           param->mixed->memory_size() >= static_cast<size_t>(param->mixed->numel()) * bf16_bytes &&
           param->weight->memory_size() >= static_cast<size_t>(param->weight->numel()) * fp8_bytes &&
           param->conv_out->memory_size() >= static_cast<size_t>(param->conv_out->numel()) * bf16_bytes &&
           param->discarded_prefix->memory_size() >=
               static_cast<size_t>(param->discarded_prefix->numel()) * bf16_bytes &&
           param->next_state->memory_size() >= static_cast<size_t>(param->next_state->numel()) * bf16_bytes &&
           ((*channels = c), true);
}

template <DataType dtype>
inline float QuantizeAndDequantize(float value, float scale, const std::array<float, 256>& table) {
    return table[EncodeFp8<dtype>(value / scale)] * scale;
}

template <DataType dtype>
inline void ComputeFp8Channel(const uint16_t* state, const uint16_t* mixed, const uint8_t* weight,
                              int64_t channel, float input_scale, float weight_scale, float output_scale,
                              uint16_t* conv_out, uint16_t* discarded_prefix, uint16_t* next_state,
                              const std::array<float, 256>& table) {
    const int64_t state_offset = channel * 3;
    const int64_t weight_offset = channel * 4;
    const float state0 = BFloat16ToFloat(state[state_offset]);
    const float state1 = BFloat16ToFloat(state[state_offset + 1]);
    const float state2 = BFloat16ToFloat(state[state_offset + 2]);
    const float mixed_value = BFloat16ToFloat(mixed[channel]);
    float value = QuantizeAndDequantize<dtype>(state0, input_scale, table) *
                  table[weight[weight_offset]] * weight_scale;
    value += QuantizeAndDequantize<dtype>(state1, input_scale, table) *
             table[weight[weight_offset + 1]] * weight_scale;
    value += QuantizeAndDequantize<dtype>(state2, input_scale, table) *
             table[weight[weight_offset + 2]] * weight_scale;
    value += QuantizeAndDequantize<dtype>(mixed_value, input_scale, table) *
             table[weight[weight_offset + 3]] * weight_scale;

    const uint8_t output_code = EncodeFp8<dtype>(value / output_scale);
    conv_out[channel] = FloatToBFloat16(table[output_code] * output_scale);
    discarded_prefix[channel] = state[state_offset];
    next_state[state_offset] = state[state_offset + 1];
    next_state[state_offset + 1] = state[state_offset + 2];
    next_state[state_offset + 2] = mixed[channel];
}

template <DataType dtype>
void ComputeFp8Range(const uint16_t* state, const uint16_t* mixed, const uint8_t* weight, int64_t begin,
                     int64_t end, float input_scale, float weight_scale, float output_scale, uint16_t* conv_out,
                     uint16_t* discarded_prefix, uint16_t* next_state, const std::array<float, 256>& table) {
    for (int64_t channel = begin; channel < end; ++channel) {
        ComputeFp8Channel<dtype>(state, mixed, weight, channel, input_scale, weight_scale, output_scale, conv_out,
                                 discarded_prefix, next_state, table);
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

template <DataType dtype>
int32_t ComputeFp8State(operators::QwenDepthwiseConvStateParam* param) {
    int64_t channels = 0;
    if (!ValidateFp8<dtype>(param, &channels)) {
        return -1;
    }

    const auto* state = static_cast<const uint16_t*>(param->state->raw_data());
    const auto* mixed = static_cast<const uint16_t*>(param->mixed->raw_data());
    const auto* weight = static_cast<const uint8_t*>(param->weight->raw_data());
    auto* conv_out = static_cast<uint16_t*>(param->conv_out->raw_data());
    auto* discarded_prefix = static_cast<uint16_t*>(param->discarded_prefix->raw_data());
    auto* next_state = static_cast<uint16_t*>(param->next_state->raw_data());
    const auto& table = Fp8DecodeTable<dtype>();
    const float input_scale = param->fp8_input_scale;
    const float weight_scale = param->weight->quantization_scale();
    const float output_scale = param->fp8_output_scale;

    const size_t worker_count = WorkerCount(channels);
#if !defined(FEATHER_WITH_OPENMP)
    (void)worker_count;
#endif
#if defined(FEATHER_WITH_OPENMP)
    if (worker_count > 1 && !omp_in_parallel()) {
        const int omp_worker_count = static_cast<int>(worker_count);
#pragma omp parallel for schedule(static) num_threads(omp_worker_count)
        for (int64_t channel = 0; channel < channels; ++channel) {
            ComputeFp8Channel<dtype>(state, mixed, weight, channel, input_scale, weight_scale, output_scale,
                                     conv_out, discarded_prefix, next_state, table);
        }
        return 0;
    }
#endif

    ComputeFp8Range<dtype>(state, mixed, weight, 0, channels, input_scale, weight_scale, output_scale, conv_out,
                           discarded_prefix, next_state, table);
    return 0;
}

template <>
int32_t QwenDepthwiseConvStateKernel<DeviceType::X86, DataType::FP8E4M3>::compute() {
    AutoTimer timer("X86::QwenDepthwiseConvState::FP8E4M3");
    return ComputeFp8State<DataType::FP8E4M3>(static_cast<operators::QwenDepthwiseConvStateParam*>(param_));
}

template <>
int32_t QwenDepthwiseConvStateKernel<DeviceType::X86, DataType::FP8E5M2>::compute() {
    AutoTimer timer("X86::QwenDepthwiseConvState::FP8E5M2");
    return ComputeFp8State<DataType::FP8E5M2>(static_cast<operators::QwenDepthwiseConvStateParam*>(param_));
}

void EnsureX86QwenDepthwiseConvStateKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(DeviceType::X86, DataType::BF16, "QwenDepthwiseConvState", []() {
            return std::make_unique<QwenDepthwiseConvStateKernel<DeviceType::X86, DataType::BF16>>();
        });
        KernelDispatcher::instance().registerKernel(DeviceType::X86, DataType::FP8E4M3,
                                                    "QwenDepthwiseConvState", []() {
                                                        return std::make_unique<
                                                            QwenDepthwiseConvStateKernel<DeviceType::X86,
                                                                                         DataType::FP8E4M3>>();
                                                    });
        KernelDispatcher::instance().registerKernel(DeviceType::X86, DataType::FP8E5M2,
                                                    "QwenDepthwiseConvState", []() {
                                                        return std::make_unique<
                                                            QwenDepthwiseConvStateKernel<DeviceType::X86,
                                                                                         DataType::FP8E5M2>>();
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
