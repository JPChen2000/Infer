#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <immintrin.h>
#include <limits>
#include <numeric>
#include <vector>

#if defined(FEATHER_WITH_OPENMP)
#include <omp.h>
#endif

#include "src/kernel/common/int8_kernel_utils.h"
#include "src/kernel/conv2d.h"
#include "src/kernel/fc.h"
#include "src/kernel/gemm.h"
#include "src/kernel/matmul.h"
#include "src/operator/params.h"
#include "src/kernel/x86/int8_conv.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace x86 {

void PackInt8ConvWeightsOc8(const int8_t* weight, int64_t output_channels, int64_t patch_size,
                            std::vector<int8_t>* packed) {
    if (packed == nullptr) {
        return;
    }
    packed->clear();
    if (weight == nullptr || output_channels <= 0 || patch_size <= 0) {
        return;
    }

    const int64_t oc8_blocks = output_channels / 8;
    packed->resize(static_cast<size_t>(oc8_blocks * patch_size * 8));
    for (int64_t block = 0; block < oc8_blocks; ++block) {
        for (int64_t patch_index = 0; patch_index < patch_size; ++patch_index) {
            for (int64_t lane = 0; lane < 8; ++lane) {
                const int64_t output_channel = block * 8 + lane;
                (*packed)[static_cast<size_t>((block * patch_size + patch_index) * 8 + lane)] =
                    weight[static_cast<size_t>(output_channel * patch_size + patch_index)];
            }
        }
    }
}

#if defined(__AVX2__)
static inline __m256i DotInt8Oc8MaddubsPacked(
    __m256i accumulator, int8_t input_value, __m128i maddubs_weights) {
    // maddubs consumes unsigned input bytes and signed weight bytes. For
    // signed x and w, [x+128, 128] * [w, ~w] + 128 equals x*w.
    const int unsigned_input = static_cast<int>(input_value) + 128;
    const __m128i input_values =
        _mm_set1_epi16((128 << 8) | unsigned_input);
    const __m128i products = _mm_maddubs_epi16(input_values, maddubs_weights);
    const __m256i values = _mm256_add_epi32(
        _mm256_cvtepi16_epi32(products), _mm256_set1_epi32(128));
    return _mm256_add_epi32(accumulator, values);
}

static inline __m256i DotInt8Oc8Maddubs(
    __m256i accumulator, int8_t input_value, const int8_t* packed_weight) {
    const __m128i weight_values =
        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(packed_weight));
    const __m128i inverted_weight_values =
        _mm_xor_si128(weight_values, _mm_set1_epi8(static_cast<char>(-1)));
    const __m128i maddubs_weights =
        _mm_unpacklo_epi8(weight_values, inverted_weight_values);
    return DotInt8Oc8MaddubsPacked(accumulator, input_value, maddubs_weights);
}

static inline __m256i DotInt8Oc8MaddubsPair(
    __m256i accumulator, int8_t input_first, int8_t input_second,
    const int8_t* packed_weight_first) {
    const __m128i first_weights =
        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(packed_weight_first));
    const __m128i second_weights =
        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(packed_weight_first + 8));
    const __m128i inverted_first_weights =
        _mm_xor_si128(first_weights, _mm_set1_epi8(static_cast<char>(-1)));
    const __m128i inverted_second_weights =
        _mm_xor_si128(second_weights, _mm_set1_epi8(static_cast<char>(-1)));
    const __m128i first_maddubs_weights =
        _mm_unpacklo_epi8(first_weights, inverted_first_weights);
    const __m128i second_maddubs_weights =
        _mm_unpacklo_epi8(second_weights, inverted_second_weights);
    const __m128i first_input =
        _mm_set1_epi16((128 << 8) | (static_cast<int>(input_first) + 128));
    const __m128i second_input =
        _mm_set1_epi16((128 << 8) | (static_cast<int>(input_second) + 128));
    const __m256i inputs = _mm256_set_m128i(second_input, first_input);
    const __m256i weights =
        _mm256_set_m128i(second_maddubs_weights, first_maddubs_weights);
    const __m256i products = _mm256_maddubs_epi16(inputs, weights);
    const __m128i summed_products = _mm_add_epi16(
        _mm256_castsi256_si128(products), _mm256_extracti128_si256(products, 1));
    const __m256i values = _mm256_add_epi32(
        _mm256_cvtepi16_epi32(summed_products), _mm256_set1_epi32(256));
    return _mm256_add_epi32(accumulator, values);
}

static inline __m256i DotInt8Oc8MaddubsPairPackedWeights(
    __m256i accumulator, int8_t input_first, int8_t input_second,
    __m256i packed_weight_pair) {
    const __m128i first_input =
        _mm_set1_epi16((128 << 8) | (static_cast<int>(input_first) + 128));
    const __m128i second_input =
        _mm_set1_epi16((128 << 8) | (static_cast<int>(input_second) + 128));
    const __m256i inputs = _mm256_set_m128i(second_input, first_input);
    const __m256i products = _mm256_maddubs_epi16(inputs, packed_weight_pair);
    const __m128i summed_products = _mm_add_epi16(
        _mm256_castsi256_si128(products), _mm256_extracti128_si256(products, 1));
    const __m256i values = _mm256_add_epi32(
        _mm256_cvtepi16_epi32(summed_products), _mm256_set1_epi32(256));
    return _mm256_add_epi32(accumulator, values);
}

static inline __m256i DotInt8Oc8MaddubsPairPacked(
    __m256i accumulator, int8_t input_first, int8_t input_second,
    const int8_t* packed_weight_pair) {
    const __m256i packed_weights =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed_weight_pair));
    return DotInt8Oc8MaddubsPairPackedWeights(
        accumulator, input_first, input_second, packed_weights);
}

static inline __m256i DotInt8Oc8Pair(
    __m256i accumulator, int8_t input_first, int8_t input_second,
    const int8_t* packed_weight_first) {
    return DotInt8Oc8MaddubsPair(accumulator, input_first, input_second,
                                 packed_weight_first);
}
#endif

void AccumulateInt8Oc8Maddubs(
    const int8_t* input, const int8_t* packed_weight, int64_t patch_size,
    int32_t* accumulators) {
    if (input == nullptr || packed_weight == nullptr || accumulators == nullptr ||
        patch_size <= 0) {
        return;
    }
#if defined(__AVX2__)
    __m256i accumulator =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(accumulators));
    int64_t patch_index = 0;
    for (; patch_index + 1 < patch_size; patch_index += 2) {
        accumulator = DotInt8Oc8MaddubsPair(
            accumulator, input[patch_index], input[patch_index + 1],
            packed_weight + patch_index * 8);
    }
    if (patch_index < patch_size) {
        accumulator = DotInt8Oc8Maddubs(
            accumulator, input[patch_index], packed_weight + patch_index * 8);
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(accumulators), accumulator);
#else
    for (int64_t patch_index = 0; patch_index < patch_size; ++patch_index) {
        for (int output_channel = 0; output_channel < 8; ++output_channel) {
            accumulators[output_channel] +=
                static_cast<int32_t>(input[patch_index]) *
                static_cast<int32_t>(packed_weight[patch_index * 8 + output_channel]);
        }
    }
#endif
}

void AccumulateInt8Oc8MaddubsPairPacked(
    const int8_t* input, const int8_t* packed_weight, int64_t pair_count,
    int32_t* accumulators) {
    if (input == nullptr || packed_weight == nullptr || accumulators == nullptr || pair_count <= 0) {
        return;
    }
#if defined(__AVX2__)
    __m256i accumulator =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(accumulators));
    for (int64_t pair = 0; pair < pair_count; ++pair) {
        accumulator = DotInt8Oc8MaddubsPairPacked(
            accumulator, input[pair * 2], input[pair * 2 + 1], packed_weight + pair * 32);
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(accumulators), accumulator);
#else
    for (int64_t pair = 0; pair < pair_count; ++pair) {
        for (int64_t offset = 0; offset < 2; ++offset) {
            const int8_t value = input[pair * 2 + offset];
            for (int output_channel = 0; output_channel < 8; ++output_channel) {
                const int8_t weight = packed_weight[pair * 32 + offset * 16 + output_channel * 2];
                accumulators[output_channel] += static_cast<int32_t>(value) * static_cast<int32_t>(weight);
            }
        }
    }
#endif
}

void AccumulateInt8Oc8Pairwise(
    const int8_t* input, const int8_t* packed_weight, int64_t patch_size,
    int32_t* accumulators) {
    if (input == nullptr || packed_weight == nullptr || accumulators == nullptr ||
        patch_size <= 0) {
        return;
    }
#if defined(__AVX2__)
    __m256i accumulator = _mm256_setzero_si256();
    int64_t patch_index = 0;
    for (; patch_index + 1 < patch_size; patch_index += 2) {
        accumulator = DotInt8Oc8Pair(
            accumulator, input[patch_index], input[patch_index + 1],
            packed_weight + patch_index * 8);
    }
    if (patch_index < patch_size) {
        accumulator = DotInt8Oc8Maddubs(
            accumulator, input[patch_index], packed_weight + patch_index * 8);
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(accumulators), accumulator);
#else
    for (int64_t patch_index = 0; patch_index < patch_size; ++patch_index) {
        for (int output_channel = 0; output_channel < 8; ++output_channel) {
            accumulators[output_channel] +=
                static_cast<int32_t>(input[patch_index]) *
                static_cast<int32_t>(packed_weight[patch_index * 8 + output_channel]);
        }
    }
#endif
}


void PackInt8ConvWeightsMaddubsPair(
    const int8_t* weight, int64_t output_channels, int64_t input_channels,
    int64_t kernel_h, int64_t kernel_w, std::vector<int8_t>* packed) {
    if (weight == nullptr || packed == nullptr || output_channels <= 0 || input_channels <= 0 ||
        kernel_h <= 0 || kernel_w <= 0) {
        if (packed != nullptr) {
            packed->clear();
        }
        return;
    }
    const int64_t oc8_blocks = output_channels / 8;
    const int64_t pair_width = (kernel_w + 1) / 2;
    const int64_t rows_per_block = input_channels * kernel_h * pair_width;
    packed->assign(static_cast<size_t>(oc8_blocks * rows_per_block * 32), 0);
    for (int64_t block = 0; block < oc8_blocks; ++block) {
        for (int64_t input_channel = 0; input_channel < input_channels; ++input_channel) {
            for (int64_t kernel_y = 0; kernel_y < kernel_h; ++kernel_y) {
                for (int64_t pair = 0; pair < pair_width; ++pair) {
                    const int64_t base =
                        (block * rows_per_block +
                         (input_channel * kernel_h + kernel_y) * pair_width + pair) * 32;
                    for (int64_t x_offset = 0; x_offset < 2; ++x_offset) {
                        const int64_t kernel_x = pair * 2 + x_offset;
                        for (int64_t lane = 0; lane < 8; ++lane) {
                            const int64_t output_channel = block * 8 + lane;
                            const int64_t source_index =
                                output_channel * input_channels * kernel_h * kernel_w +
                                input_channel * kernel_h * kernel_w +
                                kernel_y * kernel_w + kernel_x;
                            const int8_t value = kernel_x < kernel_w
                                ? weight[source_index] : static_cast<int8_t>(0);
                            const size_t destination =
                                static_cast<size_t>(base + x_offset * 16 + lane * 2);
                            (*packed)[destination] = value;
                            (*packed)[destination + 1] =
                                static_cast<int8_t>(static_cast<uint8_t>(value) ^ 0xffu);
                        }
                    }
                }
            }
        }
    }
}

void PackInt8PointwiseWeightsMaddubsPair(
    const int8_t* weight, int64_t output_channels, int64_t input_channels,
    std::vector<int8_t>* packed) {
    if (weight == nullptr || packed == nullptr || output_channels <= 0 || input_channels <= 0) {
        if (packed != nullptr) {
            packed->clear();
        }
        return;
    }
    const int64_t oc8_blocks = output_channels / 8;
    const int64_t pair_count = (input_channels + 1) / 2;
    packed->assign(static_cast<size_t>(oc8_blocks * pair_count * 32), 0);
    for (int64_t block = 0; block < oc8_blocks; ++block) {
        for (int64_t pair = 0; pair < pair_count; ++pair) {
            const int64_t base = (block * pair_count + pair) * 32;
            for (int64_t input_offset = 0; input_offset < 2; ++input_offset) {
                const int64_t input_channel = pair * 2 + input_offset;
                for (int64_t lane = 0; lane < 8; ++lane) {
                    const int64_t output_channel = block * 8 + lane;
                    const int8_t value = input_channel < input_channels
                        ? weight[output_channel * input_channels + input_channel]
                        : static_cast<int8_t>(0);
                    const size_t destination = static_cast<size_t>(base + input_offset * 16 + lane * 2);
                    (*packed)[destination] = value;
                    (*packed)[destination + 1] =
                        static_cast<int8_t>(static_cast<uint8_t>(value) ^ 0xffu);
                }
            }
        }
    }
}

void PackInt8ConvWeightsVnni(const int8_t* weight, int64_t output_channels,
                             int64_t patch_size, std::vector<int8_t>* packed,
                             std::vector<int32_t>* weight_sums){
    if (packed == nullptr || weight_sums == nullptr) {
        return;
    }
    packed->clear();
    weight_sums->clear();
    if (weight == nullptr || output_channels <= 0 || patch_size <= 0) {
        return;
    }

    weight_sums->assign(static_cast<size_t>(output_channels), 0);
    for (int64_t output_channel = 0; output_channel < output_channels; ++output_channel) {
        int32_t sum = 0;
        for (int64_t patch_index = 0; patch_index < patch_size; ++patch_index) {
            sum += static_cast<int32_t>(weight[static_cast<size_t>(output_channel * patch_size + patch_index)]);
        }
        (*weight_sums)[static_cast<size_t>(output_channel)] = sum;
    }

    const int64_t oc8_blocks = output_channels / 8;
    const int64_t patch_groups = (patch_size + 3) / 4;
    packed->assign(static_cast<size_t>(oc8_blocks * patch_groups * 32), 0);
    for (int64_t block = 0; block < oc8_blocks; ++block) {
        for (int64_t patch_group = 0; patch_group < patch_groups; ++patch_group) {
            for (int64_t output_channel = 0; output_channel < 8; ++output_channel) {
                const int64_t source_channel = block * 8 + output_channel;
                for (int64_t lane = 0; lane < 4; ++lane) {
                    const int64_t patch_index = patch_group * 4 + lane;
                    if (patch_index >= patch_size) {
                        continue;
                    }
                    const size_t packed_index = static_cast<size_t>(
                        (block * patch_groups + patch_group) * 32 + output_channel * 4 + lane);
                    (*packed)[packed_index] =
                        weight[static_cast<size_t>(source_channel * patch_size + patch_index)];
                }
            }
        }
    }
}

}  // namespace x86

namespace {

using int8_detail::BuildInputQuantizationView;
using int8_detail::BuildOutputQuantizationView;
using int8_detail::BuildWeightQuantizationView;
using int8_detail::FitsInt32;
using int8_detail::QuantizationView;

#if defined(__GNUC__) || defined(__clang__)
#define FEATHER_INT8_AVX_VNNI_TARGET __attribute__((target("avxvnni")))
#else
#define FEATHER_INT8_AVX_VNNI_TARGET
#endif

FEATHER_INT8_AVX_VNNI_TARGET inline __m256i DotInt8Vnni8(__m256i accumulator, uint32_t unsigned_input_word,
                                                        const int8_t* packed_weight) {
    const __m256i input = _mm256_set1_epi32(static_cast<int32_t>(unsigned_input_word));
    const __m256i weight = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed_weight));
    return _mm256_dpbusd_epi32(accumulator, input, weight);
}

FEATHER_INT8_AVX_VNNI_TARGET inline __m256i DotInt8Vnni8Loaded(__m256i accumulator,
                                                               uint32_t unsigned_input_word,
                                                               __m256i packed_weight) {
    const __m256i input = _mm256_set1_epi32(static_cast<int32_t>(unsigned_input_word));
    return _mm256_dpbusd_epi32(accumulator, input, packed_weight);
}

bool HasAvxVnni() {
#if defined(__GNUC__) || defined(__clang__)
    const char* disabled = std::getenv("FEATHER_DISABLE_AVX_VNNI");
    if (disabled != nullptr && disabled[0] == '1') {
        return false;
    }
    __builtin_cpu_init();
    return __builtin_cpu_supports("avxvnni");
#else
    return false;
#endif
}

// Quantize eight INT32 accumulators without entering the scalar double/round
// path for every output. The float conversion is exact for the normal INT8
// convolution range; larger accumulators use the existing scalar path so the
// public rounding and overflow contract remains unchanged.
inline bool FastQuantizeAccumulator8(const int32_t* accumulators, const float* scales,
                                     int32_t output_zero_point, int8_t* output) {
    if (accumulators == nullptr || scales == nullptr || output == nullptr) {
        return false;
    }
    for (int index = 0; index < 8; ++index) {
        const int64_t value = static_cast<int64_t>(accumulators[index]);
        if (value < -(1LL << 23) || value > (1LL << 23) || !std::isfinite(scales[index]) ||
            std::fabs(scales[index]) >= 1.0e20f) {
            return false;
        }
    }

    const __m256 accumulator =
        _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accumulators)));
    const __m256 scale = _mm256_loadu_ps(scales);
    const __m256 transformed = _mm256_add_ps(
        _mm256_mul_ps(accumulator, scale), _mm256_set1_ps(static_cast<float>(output_zero_point)));
    const __m256 zero = _mm256_setzero_ps();
    // std::round is ties-away-from-zero. floor(x + .5) / ceil(x - .5)
    // preserves that rule while remaining vectorizable.
    const __m256 rounded_positive = _mm256_floor_ps(_mm256_add_ps(transformed, _mm256_set1_ps(0.5f)));
    const __m256 rounded_negative = _mm256_ceil_ps(_mm256_sub_ps(transformed, _mm256_set1_ps(0.5f)));
    const __m256 rounded = _mm256_blendv_ps(
        rounded_negative, rounded_positive, _mm256_cmp_ps(transformed, zero, _CMP_GE_OQ));
    const __m256 clamped = _mm256_min_ps(_mm256_set1_ps(127.0f),
                                         _mm256_max_ps(_mm256_set1_ps(-128.0f), rounded));
    alignas(32) int32_t quantized[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(quantized), _mm256_cvtps_epi32(clamped));
    for (int index = 0; index < 8; ++index) {
        output[index] = static_cast<int8_t>(quantized[index]);
    }
    return true;
}

bool HasDims(const Tensor& tensor, const std::vector<int64_t>& expected) {
    if (tensor.dims().size() != expected.size()) {
        return false;
    }
    for (size_t index = 0; index < expected.size(); ++index) {
        if (tensor.dims()[index] != expected[index]) {
            return false;
        }
    }
    return true;
}

int32_t ComputeX86Int8Fc(feather::operators::FcParam* param) {
    if (param == nullptr || param->input == nullptr || param->w == nullptr || param->out == nullptr ||
        param->input->dims().size() != 2 || param->w->dims().size() != 2 ||
        param->input->dims()[1] != param->w->dims()[0]) {
        return -1;
    }
    const int64_t rows = param->input->dims()[0];
    const int64_t k = param->input->dims()[1];
    const int64_t channels = param->w->dims()[1];
    if (rows <= 0 || k <= 0 || channels <= 0 || !HasDims(*param->out, {rows, channels})) {
        return -1;
    }

    QuantizationView input_quantization;
    QuantizationView weight_quantization;
    QuantizationView output_quantization;
    if (!BuildInputQuantizationView(param->input, &input_quantization) ||
        !BuildWeightQuantizationView(param->w, 1, channels, &weight_quantization) ||
        !BuildOutputQuantizationView(param->out, &output_quantization) ||
        !int8_detail::ValidateLinearBias(param->bias, rows, channels)) {
        return -1;
    }

    const int8_t* input = param->input->data<int8_t>();
    const int8_t* weight = param->w->data<int8_t>();
    int8_t* output = param->out->mutable_data<int8_t>();
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t channel = 0; channel < channels; ++channel) {
            int64_t accumulator = 0;
            const int32_t weight_zero_point = weight_quantization.zero_point_for(static_cast<size_t>(channel));
            for (int64_t index = 0; index < k; ++index) {
                accumulator += static_cast<int64_t>(static_cast<int32_t>(input[row * k + index]) -
                                                    input_quantization.zero_point) *
                               (static_cast<int32_t>(weight[index * channels + channel]) - weight_zero_point);
                if (!FitsInt32(accumulator)) {
                    return -1;
                }
            }
            if (!int8_detail::AddInt32Bias(
                    &accumulator, int8_detail::ReadLinearBias(param->bias, row, channel, channels))) {
                return -1;
            }
            const double scale = static_cast<double>(input_quantization.scale) *
                                 weight_quantization.scale_for(static_cast<size_t>(channel));
            if (!int8_detail::QuantizeAccumulator(accumulator, scale, output_quantization,
                                                  &output[row * channels + channel])) {
                return -1;
            }
        }
    }
    return 0;
}

int32_t ComputeX86Int8Gemm(feather::operators::GemmParam* param) {
    if (param == nullptr || param->a == nullptr || param->b == nullptr || param->out == nullptr ||
        param->a->dims().size() < 2 || param->b->dims().size() != 2 || param->trans_a ||
        !std::isfinite(param->alpha) || !std::isfinite(param->beta)) {
        return -1;
    }
    const auto& a_dims = param->a->dims().data();
    const auto& b_dims = param->b->dims().data();
    const int64_t k = a_dims[a_dims.size() - 1];
    const int64_t rows = k > 0 ? param->a->numel() / k : 0;
    const int64_t b_k = param->trans_b ? b_dims[1] : b_dims[0];
    const int64_t channels = param->trans_b ? b_dims[0] : b_dims[1];
    std::vector<int64_t> expected_output = a_dims;
    expected_output[expected_output.size() - 1] = channels;
    if (k <= 0 || rows <= 0 || channels <= 0 || b_k != k || !HasDims(*param->out, expected_output)) {
        return -1;
    }

    QuantizationView input_quantization;
    QuantizationView weight_quantization;
    QuantizationView output_quantization;
    const int64_t weight_axis = param->trans_b ? 0 : 1;
    if (!BuildInputQuantizationView(param->a, &input_quantization) ||
        !BuildWeightQuantizationView(param->b, weight_axis, channels, &weight_quantization) ||
        !BuildOutputQuantizationView(param->out, &output_quantization) ||
        !int8_detail::ValidateLinearBias(param->bias, rows, channels)) {
        return -1;
    }

    const int8_t* lhs = param->a->data<int8_t>();
    const int8_t* rhs = param->b->data<int8_t>();
    int8_t* output = param->out->mutable_data<int8_t>();
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t channel = 0; channel < channels; ++channel) {
            int64_t dot = 0;
            const int32_t weight_zero_point = weight_quantization.zero_point_for(static_cast<size_t>(channel));
            for (int64_t index = 0; index < k; ++index) {
                const int64_t rhs_offset = param->trans_b ? channel * k + index : index * channels + channel;
                dot += static_cast<int64_t>(static_cast<int32_t>(lhs[row * k + index]) -
                                            input_quantization.zero_point) *
                       (static_cast<int32_t>(rhs[rhs_offset]) - weight_zero_point);
                if (!FitsInt32(dot)) {
                    return -1;
                }
            }
            const int32_t bias = int8_detail::ReadLinearBias(param->bias, row, channel, channels);
            const double accumulator = static_cast<double>(param->alpha) * static_cast<double>(dot) +
                                       static_cast<double>(param->beta) * static_cast<double>(bias);
            const double scale = static_cast<double>(input_quantization.scale) *
                                 weight_quantization.scale_for(static_cast<size_t>(channel));
            if (!int8_detail::QuantizeReal(accumulator * scale, output_quantization,
                                           &output[row * channels + channel])) {
                return -1;
            }
        }
    }
    return 0;
}

std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (int64_t index = static_cast<int64_t>(dims.size()) - 2; index >= 0; --index) {
        strides[static_cast<size_t>(index)] =
            strides[static_cast<size_t>(index + 1)] * dims[static_cast<size_t>(index + 1)];
    }
    return strides;
}

int32_t ComputeX86Int8MatMul(feather::operators::MatMulParam* param) {
    if (param == nullptr || param->a == nullptr || param->b == nullptr || param->out == nullptr) {
        return -1;
    }
    const auto& a_dims = param->a->dims().data();
    const auto& b_dims = param->b->dims().data();
    const auto& out_dims = param->out->dims().data();
    const size_t a_rank = a_dims.size();
    const size_t b_rank = b_dims.size();
    const size_t out_rank = out_dims.size();
    if (a_rank < 2 || b_rank < 2 || out_rank != std::max(a_rank, b_rank)) {
        return -1;
    }
    const int64_t m = a_dims[a_rank - 2];
    const int64_t k = a_dims[a_rank - 1];
    const int64_t b_k = b_dims[b_rank - 2];
    const int64_t n = b_dims[b_rank - 1];
    if (m <= 0 || k <= 0 || n <= 0 || b_k != k) {
        return -1;
    }
    const size_t batch_rank = out_rank - 2;
    const size_t a_batch_rank = a_rank - 2;
    const size_t b_batch_rank = b_rank - 2;
    if (a_batch_rank > batch_rank || b_batch_rank > batch_rank || out_dims[out_rank - 2] != m ||
        out_dims[out_rank - 1] != n) {
        return -1;
    }
    for (size_t axis = 0; axis < batch_rank; ++axis) {
        const int64_t a_dim = axis < batch_rank - a_batch_rank ? 1 : a_dims[axis - (batch_rank - a_batch_rank)];
        const int64_t b_dim = axis < batch_rank - b_batch_rank ? 1 : b_dims[axis - (batch_rank - b_batch_rank)];
        if ((a_dim != 1 && b_dim != 1 && a_dim != b_dim) || out_dims[axis] != std::max(a_dim, b_dim)) {
            return -1;
        }
    }

    QuantizationView input_quantization;
    QuantizationView weight_quantization;
    QuantizationView output_quantization;
    if (!BuildInputQuantizationView(param->a, &input_quantization) ||
        !BuildWeightQuantizationView(param->b, static_cast<int64_t>(b_rank - 1), n, &weight_quantization) ||
        !BuildOutputQuantizationView(param->out, &output_quantization)) {
        return -1;
    }

    const auto a_strides = ComputeStrides(a_dims);
    const auto b_strides = ComputeStrides(b_dims);
    const std::vector<int64_t> batch_dims(out_dims.begin(), out_dims.begin() + batch_rank);
    const auto batch_strides = ComputeStrides(batch_dims);
    const int64_t batch_count = batch_rank == 0
                                    ? 1
                                    : std::accumulate(batch_dims.begin(), batch_dims.end(), int64_t{1},
                                                      std::multiplies<int64_t>());
    std::vector<int64_t> batch_coords(batch_rank, 0);
    const int8_t* lhs = param->a->data<int8_t>();
    const int8_t* rhs = param->b->data<int8_t>();
    int8_t* output = param->out->mutable_data<int8_t>();
    for (int64_t batch = 0; batch < batch_count; ++batch) {
        int64_t remaining = batch;
        for (size_t axis = 0; axis < batch_rank; ++axis) {
            batch_coords[axis] = remaining / batch_strides[axis];
            remaining %= batch_strides[axis];
        }
        int64_t a_batch_offset = 0;
        const size_t a_gap = batch_rank - a_batch_rank;
        for (size_t axis = 0; axis < a_batch_rank; ++axis) {
            const int64_t coordinate = a_dims[axis] == 1 ? 0 : batch_coords[axis + a_gap];
            a_batch_offset += coordinate * a_strides[axis];
        }
        int64_t b_batch_offset = 0;
        const size_t b_gap = batch_rank - b_batch_rank;
        for (size_t axis = 0; axis < b_batch_rank; ++axis) {
            const int64_t coordinate = b_dims[axis] == 1 ? 0 : batch_coords[axis + b_gap];
            b_batch_offset += coordinate * b_strides[axis];
        }
        const int64_t output_batch_offset = batch * m * n;
        for (int64_t row = 0; row < m; ++row) {
            for (int64_t channel = 0; channel < n; ++channel) {
                int64_t accumulator = 0;
                const int32_t weight_zero_point = weight_quantization.zero_point_for(static_cast<size_t>(channel));
                for (int64_t index = 0; index < k; ++index) {
                    accumulator += static_cast<int64_t>(
                        static_cast<int32_t>(lhs[a_batch_offset + row * a_strides[a_rank - 2] + index]) -
                        input_quantization.zero_point) *
                        (static_cast<int32_t>(rhs[b_batch_offset + index * b_strides[b_rank - 2] + channel]) -
                         weight_zero_point);
                    if (!FitsInt32(accumulator)) {
                        return -1;
                    }
                }
                const double scale = static_cast<double>(input_quantization.scale) *
                                     weight_quantization.scale_for(static_cast<size_t>(channel));
                if (!int8_detail::QuantizeAccumulator(
                        accumulator, scale, output_quantization,
                        &output[output_batch_offset + row * n + channel])) {
                    return -1;
                }
            }
        }
    }
    return 0;
}

int32_t ComputeX86Int8Conv2D(feather::operators::Conv2dParam* param, const Tensor** cached_weight_tensor,
                             std::vector<int8_t>* cached_packed_weight,
                             const Tensor** cached_maddubs_weight_tensor,
                             std::vector<int8_t>* cached_maddubs_weight,
                             const Tensor** cached_pointwise_maddubs_weight_tensor,
                             std::vector<int8_t>* cached_pointwise_maddubs_weight,
                             const Tensor** cached_vnni_weight_tensor,
                             std::vector<int8_t>* cached_vnni_weight,
                             std::vector<int32_t>* cached_vnni_sums) {
    if (param == nullptr || param->input == nullptr || param->w == nullptr || param->out == nullptr ||
        param->input->dims().size() != 4 || param->w->dims().size() != 4 || param->out->dims().size() != 4 ||
        param->stride_h <= 0 || param->stride_w <= 0 || param->dilation_h <= 0 || param->dilation_w <= 0 ||
        param->group <= 0) {
        return -1;
    }
    ImageShape4D input_shape;
    ImageShape4D output_shape;
    if (!DecodeImageShape4D(param->input->dims().data(), param->input->layout(), &input_shape) ||
        !DecodeImageShape4D(param->out->dims().data(), param->out->layout(), &output_shape)) {
        return -1;
    }
    const int64_t output_channels = param->w->dims()[0];
    const int64_t input_channels_per_group = param->w->dims()[1];
    const int64_t kernel_h = param->w->dims()[2];
    const int64_t kernel_w = param->w->dims()[3];
    if (input_shape.n <= 0 || input_shape.c <= 0 || input_shape.h <= 0 || input_shape.w <= 0 ||
        output_channels <= 0 || input_channels_per_group <= 0 || kernel_h <= 0 || kernel_w <= 0 ||
        input_shape.c % param->group != 0 || output_channels % param->group != 0 ||
        input_channels_per_group != input_shape.c / param->group) {
        return -1;
    }
    const int64_t expected_h = (input_shape.h + 2 * param->pad_h -
                                param->dilation_h * (kernel_h - 1) - 1) /
                                   param->stride_h +
                               1;
    const int64_t expected_w = (input_shape.w + 2 * param->pad_w -
                                param->dilation_w * (kernel_w - 1) - 1) /
                                   param->stride_w +
                               1;
    if (expected_h <= 0 || expected_w <= 0 || output_shape.n != input_shape.n ||
        output_shape.c != output_channels || output_shape.h != expected_h || output_shape.w != expected_w) {
        return -1;
    }

    QuantizationView input_quantization;
    QuantizationView weight_quantization;
    QuantizationView output_quantization;
    if (!BuildInputQuantizationView(param->input, &input_quantization) ||
        !BuildWeightQuantizationView(param->w, 0, output_channels, &weight_quantization) ||
        !BuildOutputQuantizationView(param->out, &output_quantization) ||
        !int8_detail::ValidateConvBias(param->bias, output_channels)) {
        return -1;
    }

    const DataLayout input_layout = NormalizeDataLayout(param->input->layout());
    const DataLayout output_layout = NormalizeDataLayout(param->out->layout());
    const int64_t output_channels_per_group = output_channels / param->group;
    const int8_t* input = param->input->data<int8_t>();
    const int8_t* weight = param->w->data<int8_t>();
    int8_t* output = param->out->mutable_data<int8_t>();
    const bool nchw = input_layout == DataLayout::NCHW && output_layout == DataLayout::NCHW;
    const int64_t work_items = input_shape.n * param->group * output_channels_per_group * output_shape.h * output_shape.w;
    std::atomic<bool> failed{false};

    // The previous implementation walked every output through the generic
    // layout helper and ran the whole graph on one thread.  YOLOv5 is NCHW,
    // so flatten the independent output elements and use direct plane
    // addressing on the hot path.  Keep the generic path for other layouts.
    auto compute_output = [&](int64_t work_index) {
        if (failed.load(std::memory_order_relaxed)) return;
        int64_t remaining = work_index;
        const int64_t output_w = remaining % output_shape.w;
        remaining /= output_shape.w;
        const int64_t output_h = remaining % output_shape.h;
        remaining /= output_shape.h;
        const int64_t channel = remaining % output_channels_per_group;
        remaining /= output_channels_per_group;
        const int64_t group = remaining % param->group;
        const int64_t batch = remaining / param->group;
        const int64_t output_channel = group * output_channels_per_group + channel;
        const int32_t weight_zero_point =
            weight_quantization.zero_point_for(static_cast<size_t>(output_channel));
        const double scale = static_cast<double>(input_quantization.scale) *
                             weight_quantization.scale_for(static_cast<size_t>(output_channel));
        int64_t accumulator = int8_detail::ReadConvBias(param->bias, output_channel);

        for (int64_t input_channel = 0; input_channel < input_channels_per_group; ++input_channel) {
            const int64_t global_input_channel = group * input_channels_per_group + input_channel;
            for (int64_t kernel_y = 0; kernel_y < kernel_h; ++kernel_y) {
                const int64_t input_y = output_h * param->stride_h +
                                        kernel_y * param->dilation_h - param->pad_h;
                if (input_y < 0 || input_y >= input_shape.h) continue;

                if (nchw) {
                    const int64_t input_x_origin = output_w * param->stride_w - param->pad_w;
                    const int64_t kernel_x_begin = std::max<int64_t>(0, -input_x_origin);
                    const int64_t kernel_x_end =
                        std::min<int64_t>(kernel_w, input_shape.w - input_x_origin);
                    if (kernel_x_begin >= kernel_x_end) continue;
                    const int64_t input_plane = input_shape.h * input_shape.w;
                    const int64_t input_row_offset =
                        (batch * input_shape.c + global_input_channel) * input_plane +
                        input_y * input_shape.w + input_x_origin + kernel_x_begin;
                    const int64_t weight_row_offset =
                        ((output_channel * input_channels_per_group + input_channel) * kernel_h + kernel_y) *
                            kernel_w + kernel_x_begin;
                    const int8_t* input_row = input + input_row_offset;
                    const int8_t* weight_row = weight + weight_row_offset;
                    for (int64_t kernel_x = kernel_x_begin; kernel_x < kernel_x_end; ++kernel_x) {
                        accumulator += static_cast<int64_t>(
                            static_cast<int32_t>(*input_row++) - input_quantization.zero_point) *
                            (static_cast<int32_t>(*weight_row++) - weight_zero_point);
                    }
                } else {
                    for (int64_t kernel_x = 0; kernel_x < kernel_w; ++kernel_x) {
                        const int64_t input_x = output_w * param->stride_w +
                                                kernel_x * param->dilation_w - param->pad_w;
                        if (input_x < 0 || input_x >= input_shape.w) continue;
                        const int64_t input_offset = OffsetForImage4D(
                            input_layout, batch, global_input_channel, input_y, input_x,
                            input_shape.c, input_shape.h, input_shape.w);
                        const int64_t weight_offset =
                            ((output_channel * input_channels_per_group + input_channel) * kernel_h + kernel_y) *
                                kernel_w + kernel_x;
                        accumulator += static_cast<int64_t>(
                            static_cast<int32_t>(input[input_offset]) - input_quantization.zero_point) *
                            (static_cast<int32_t>(weight[weight_offset]) - weight_zero_point);
                    }
                }
            }
        }

        // Checking once per output preserves the old overflow contract while
        // removing a branch from every multiply-accumulate iteration.
        if (!FitsInt32(accumulator)) {
            failed.store(true, std::memory_order_relaxed);
            return;
        }
        const int64_t output_offset = nchw
                                          ? ((batch * output_shape.c + output_channel) * output_shape.h + output_h) *
                                                output_shape.w + output_w
                                          : OffsetForImage4D(output_layout, batch, output_channel, output_h, output_w,
                                                             output_shape.c, output_shape.h, output_shape.w);
        if (!int8_detail::QuantizeAccumulator(accumulator, scale, output_quantization,
                                              &output[output_offset])) {
            failed.store(true, std::memory_order_relaxed);
        }
    };

    // AVX2 has no signed INT8 dot-product instruction.  For the symmetric
    // activation/weight case, widen eight packed weights to INT32 lanes and
    // accumulate eight output channels at once.  Asymmetric tensors retain
    // the scalar path below so their zero-point correction remains exact.
    bool can_use_oc8_fast_path = nchw && param->group == 1 && param->dilation_h == 1 &&
                                 param->dilation_w == 1 && output_channels >= 8 &&
                                 input_quantization.zero_point == 0 && cached_weight_tensor != nullptr &&
                                 cached_packed_weight != nullptr;
    if (can_use_oc8_fast_path) {
        for (int64_t output_channel = 0; output_channel < output_channels; ++output_channel) {
            if (weight_quantization.zero_point_for(static_cast<size_t>(output_channel)) != 0) {
                can_use_oc8_fast_path = false;
                break;
            }
        }
    }

    const int64_t patch_size = input_channels_per_group * kernel_h * kernel_w;
    std::vector<float> fast_requant_scales(static_cast<size_t>(output_channels), 0.0f);
    bool can_use_fast_requant = std::isfinite(output_quantization.scale) && output_quantization.scale > 0.0f;
    if (can_use_fast_requant) {
        for (int64_t output_channel = 0; output_channel < output_channels; ++output_channel) {
            const double scale = static_cast<double>(input_quantization.scale) *
                                 weight_quantization.scale_for(static_cast<size_t>(output_channel)) /
                                 static_cast<double>(output_quantization.scale);
            if (!std::isfinite(scale) || std::fabs(scale) >= 1.0e20) {
                can_use_fast_requant = false;
                break;
            }
            fast_requant_scales[static_cast<size_t>(output_channel)] = static_cast<float>(scale);
        }
    }
    bool use_vnni = can_use_oc8_fast_path && HasAvxVnni() && cached_vnni_weight_tensor != nullptr &&
                    cached_vnni_weight != nullptr && cached_vnni_sums != nullptr;
    if (use_vnni) {
        int64_t max_bias_abs = 0;
        if (param->bias != nullptr) {
            const int32_t* bias = param->bias->data<int32_t>();
            for (int64_t output_channel = 0; output_channel < output_channels; ++output_channel) {
                const int64_t value = static_cast<int64_t>(bias[output_channel]);
                const int64_t absolute_value = value < 0 ? -value : value;
                max_bias_abs = std::max(max_bias_abs, absolute_value);
            }
        }
        const int64_t int32_max = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
        constexpr int64_t kMaxVnniIntermediateProduct = 383 * 128;
        use_vnni = max_bias_abs <= int32_max &&
                   patch_size <= (int32_max - max_bias_abs) / kMaxVnniIntermediateProduct;
    }
    if (use_vnni) {
        const int64_t oc8_blocks = output_channels / 8;
        const int64_t patch_groups = (patch_size + 3) / 4;
        const size_t packed_size = static_cast<size_t>(oc8_blocks * patch_groups * 32);
        if (*cached_vnni_weight_tensor != param->w.get() || cached_vnni_weight->size() != packed_size ||
            cached_vnni_sums->size() != static_cast<size_t>(output_channels)) {
            x86::PackInt8ConvWeightsVnni(param->w->data<int8_t>(), output_channels, patch_size,
                                         cached_vnni_weight, cached_vnni_sums);
            *cached_vnni_weight_tensor = param->w.get();
        }
        if (cached_vnni_weight->size() != packed_size ||
            cached_vnni_sums->size() != static_cast<size_t>(output_channels)) {
            return -1;
        }

        const int8_t* packed_weight = cached_vnni_weight->data();
        const int32_t* weight_sums = cached_vnni_sums->data();
        const int64_t output_spatial = output_shape.h * output_shape.w;
        const bool use_vnni_spatial_major =
            param->group == 1 && param->dilation_h == 1 && param->dilation_w == 1;
        const char* vnni_tile_disabled = std::getenv("FEATHER_DISABLE_INT8_VNNI_TILE");
        if (use_vnni_spatial_major &&
            (vnni_tile_disabled == nullptr || vnni_tile_disabled[0] != '1')) {
            constexpr int64_t kTileWidth = 4;
            const int64_t tile_count = (output_shape.w + kTileWidth - 1) / kTileWidth;
            const int64_t tile_work_items = input_shape.n * output_shape.h * tile_count;
            const int64_t unsigned_patch_size = patch_groups * 4;
            auto compute_vnni_tile_range = [&](int64_t begin, int64_t end) {
                std::vector<uint8_t> unsigned_patches(
                    static_cast<size_t>(kTileWidth * unsigned_patch_size), 128U);
                for (int64_t work_index = begin; work_index < end; ++work_index) {
                    if (failed.load(std::memory_order_relaxed)) {
                        return;
                    }
                    const int64_t tile_index = work_index % tile_count;
                    const int64_t batch = work_index / tile_count / output_shape.h;
                    const int64_t output_h = work_index / tile_count % output_shape.h;
                    const int64_t output_w_base = tile_index * kTileWidth;
                    const int64_t actual_width =
                        std::min<int64_t>(kTileWidth, output_shape.w - output_w_base);
                    const int64_t input_plane = input_shape.h * input_shape.w;

                    for (int64_t tile = 0; tile < actual_width; ++tile) {
                        uint8_t* unsigned_patch =
                            unsigned_patches.data() + tile * unsigned_patch_size;
                        std::fill(unsigned_patch, unsigned_patch + unsigned_patch_size, 128U);
                        const int64_t output_w = output_w_base + tile;
                        const int64_t input_y_origin = output_h * param->stride_h - param->pad_h;
                        const int64_t input_x_origin = output_w * param->stride_w - param->pad_w;
                        const bool interior =
                            input_y_origin >= 0 && input_x_origin >= 0 &&
                            input_y_origin + kernel_h - 1 < input_shape.h &&
                            input_x_origin + kernel_w - 1 < input_shape.w;
                        int64_t patch_index = 0;
                        for (int64_t input_channel = 0;
                             input_channel < input_channels_per_group; ++input_channel) {
                            const int64_t input_channel_base =
                                (batch * input_shape.c + input_channel) * input_plane;
                            for (int64_t kernel_y = 0; kernel_y < kernel_h; ++kernel_y) {
                                const int64_t input_y = input_y_origin + kernel_y;
                                if (interior) {
                                    const int8_t* input_row = input + input_channel_base +
                                        input_y * input_shape.w + input_x_origin;
                                    for (int64_t kernel_x = 0; kernel_x < kernel_w; ++kernel_x) {
                                        unsigned_patch[static_cast<size_t>(patch_index++)] =
                                            static_cast<uint8_t>(input_row[kernel_x]) ^ 0x80U;
                                    }
                                } else {
                                    for (int64_t kernel_x = 0; kernel_x < kernel_w; ++kernel_x) {
                                        const int64_t input_x = input_x_origin + kernel_x;
                                        int8_t value = 0;
                                        if (input_y >= 0 && input_y < input_shape.h && input_x >= 0 &&
                                            input_x < input_shape.w) {
                                            value = input[input_channel_base + input_y * input_shape.w + input_x];
                                        }
                                        unsigned_patch[static_cast<size_t>(patch_index++)] =
                                            static_cast<uint8_t>(value) ^ 0x80U;
                                    }
                                }
                            }
                        }
                    }

                    for (int64_t block = 0; block < oc8_blocks; ++block) {
                        const int64_t output_channel_base = block * 8;
                        const __m256i bias =
                            param->bias == nullptr
                                ? _mm256_setzero_si256()
                                : _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                                      param->bias->data<int32_t>() + output_channel_base));
                        const __m256i weight_sums_vector = _mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(weight_sums + output_channel_base));
                        const __m256i correction = _mm256_mullo_epi32(
                            weight_sums_vector, _mm256_set1_epi32(128));
                        __m256i accumulators[kTileWidth];
                        for (int64_t tile = 0; tile < actual_width; ++tile) {
                            accumulators[tile] = _mm256_sub_epi32(bias, correction);
                        }
                        for (int64_t packed_group = 0; packed_group < patch_groups; ++packed_group) {
                            const __m256i packed_weights = _mm256_loadu_si256(
                                reinterpret_cast<const __m256i*>(
                                    packed_weight + (block * patch_groups + packed_group) * 32));
                            for (int64_t tile = 0; tile < actual_width; ++tile) {
                                uint32_t input_word = 0;
                                std::memcpy(&input_word,
                                            unsigned_patches.data() +
                                                (tile * unsigned_patch_size + packed_group * 4),
                                            sizeof(input_word));
                                accumulators[tile] = DotInt8Vnni8Loaded(
                                    accumulators[tile], input_word, packed_weights);
                            }
                        }

                        for (int64_t tile = 0; tile < actual_width; ++tile) {
                            const int64_t output_h_offset = output_h;
                            const int64_t output_w = output_w_base + tile;
                            alignas(32) int32_t accumulator_values[8];
                            _mm256_store_si256(reinterpret_cast<__m256i*>(accumulator_values),
                                               accumulators[tile]);
                            alignas(32) int8_t quantized_values[8];
                            if (can_use_fast_requant &&
                                FastQuantizeAccumulator8(
                                    accumulator_values,
                                    fast_requant_scales.data() + output_channel_base,
                                    output_quantization.zero_point, quantized_values)) {
                                for (int64_t lane = 0; lane < 8; ++lane) {
                                    const int64_t output_channel = output_channel_base + lane;
                                    const int64_t output_offset =
                                        ((batch * output_shape.c + output_channel) * output_shape.h +
                                         output_h_offset) * output_shape.w + output_w;
                                    output[output_offset] = quantized_values[lane];
                                }
                                continue;
                            }
                            for (int64_t lane = 0; lane < 8; ++lane) {
                                const int64_t output_channel = output_channel_base + lane;
                                const int64_t output_offset =
                                    ((batch * output_shape.c + output_channel) * output_shape.h +
                                     output_h_offset) * output_shape.w + output_w;
                                if (!int8_detail::QuantizeAccumulator(
                                        accumulator_values[lane],
                                        static_cast<double>(input_quantization.scale) *
                                            weight_quantization.scale_for(
                                                static_cast<size_t>(output_channel)),
                                        output_quantization, &output[output_offset])) {
                                    failed.store(true, std::memory_order_relaxed);
                                    return;
                                }
                            }
                        }
                    }
                }
            };

#if defined(FEATHER_WITH_OPENMP)
            const int workers = std::max(1, omp_get_max_threads());
            if (tile_work_items >= 64 && workers > 1 && !omp_in_parallel()) {
#pragma omp parallel num_threads(workers)
                {
                    const int64_t chunk_size =
                        (tile_work_items + static_cast<int64_t>(workers) - 1) /
                        static_cast<int64_t>(workers);
                    const int64_t begin = static_cast<int64_t>(omp_get_thread_num()) * chunk_size;
                    const int64_t end = std::min(tile_work_items, begin + chunk_size);
                    if (begin < end) {
                        compute_vnni_tile_range(begin, end);
                    }
                }
            } else
#endif
            {
                compute_vnni_tile_range(0, tile_work_items);
            }

            const int64_t tail_begin = oc8_blocks * 8;
            for (int64_t batch = 0; batch < input_shape.n; ++batch) {
                for (int64_t output_h = 0; output_h < output_shape.h; ++output_h) {
                    for (int64_t output_w = 0; output_w < output_shape.w; ++output_w) {
                        for (int64_t output_channel = tail_begin; output_channel < output_channels; ++output_channel) {
                            const int64_t work_index =
                                ((batch * output_channels + output_channel) * output_shape.h + output_h) *
                                    output_shape.w + output_w;
                            compute_output(work_index);
                        }
                    }
                }
            }
            return failed.load(std::memory_order_relaxed) ? -1 : 0;
        }
        if (use_vnni_spatial_major) {
            const int64_t spatial_work_items = input_shape.n * output_spatial;
            const int64_t unsigned_patch_size = patch_groups * 4;
            auto compute_vnni_spatial_range = [&](int64_t begin, int64_t end) {
                std::vector<uint8_t> unsigned_patch(static_cast<size_t>(unsigned_patch_size), 128U);
                for (int64_t work_index = begin; work_index < end; ++work_index) {
                    if (failed.load(std::memory_order_relaxed)) {
                        return;
                    }
                    const int64_t batch = work_index / output_spatial;
                    const int64_t spatial_index = work_index % output_spatial;
                    const int64_t output_h = spatial_index / output_shape.w;
                    const int64_t output_w = spatial_index % output_shape.w;
                    const int64_t input_y_origin =
                        output_h * param->stride_h - param->pad_h;
                    const int64_t input_x_origin =
                        output_w * param->stride_w - param->pad_w;
                    const bool interior = input_y_origin >= 0 && input_x_origin >= 0 &&
                                          input_y_origin + kernel_h - 1 < input_shape.h &&
                                          input_x_origin + kernel_w - 1 < input_shape.w;
                    const int64_t input_plane = input_shape.h * input_shape.w;
                    int64_t patch_index = 0;
                    for (int64_t input_channel = 0; input_channel < input_channels_per_group; ++input_channel) {
                        const int64_t input_channel_base =
                            (batch * input_shape.c + input_channel) * input_plane;
                        if (interior) {
                            for (int64_t kernel_y = 0; kernel_y < kernel_h; ++kernel_y) {
                                const int8_t* input_row = input + input_channel_base +
                                    (input_y_origin + kernel_y) * input_shape.w + input_x_origin;
                                for (int64_t kernel_x = 0; kernel_x < kernel_w; ++kernel_x) {
                                    unsigned_patch[static_cast<size_t>(patch_index++)] =
                                        static_cast<uint8_t>(input_row[kernel_x]) ^ 0x80U;
                                }
                            }
                        } else {
                            for (int64_t kernel_y = 0; kernel_y < kernel_h; ++kernel_y) {
                                const int64_t input_y = input_y_origin + kernel_y;
                                for (int64_t kernel_x = 0; kernel_x < kernel_w; ++kernel_x) {
                                    const int64_t input_x = input_x_origin + kernel_x;
                                    int8_t value = 0;
                                    if (input_y >= 0 && input_y < input_shape.h && input_x >= 0 &&
                                        input_x < input_shape.w) {
                                        value = input[input_channel_base + input_y * input_shape.w + input_x];
                                    }
                                    unsigned_patch[static_cast<size_t>(patch_index++)] =
                                        static_cast<uint8_t>(value) ^ 0x80U;
                                }
                            }
                        }
                    }
                    for (; patch_index < unsigned_patch_size; ++patch_index) {
                        unsigned_patch[static_cast<size_t>(patch_index)] = 128U;
                    }

                    for (int64_t block = 0; block < oc8_blocks; ++block) {
                        const int64_t output_channel_base = block * 8;
                        const __m256i bias = param->bias == nullptr
                                                 ? _mm256_setzero_si256()
                                                 : _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                                                       param->bias->data<int32_t>() + output_channel_base));
                        const __m256i weight_sums_vector = _mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(weight_sums + output_channel_base));
                        const __m256i correction = _mm256_mullo_epi32(
                            weight_sums_vector, _mm256_set1_epi32(128));
                        __m256i accumulator = _mm256_sub_epi32(bias, correction);
                        for (int64_t packed_group = 0; packed_group < patch_groups; ++packed_group) {
                            uint32_t input_word = 0;
                            std::memcpy(&input_word, unsigned_patch.data() + packed_group * 4, sizeof(input_word));
                            const int8_t* weight_group =
                                packed_weight + (block * patch_groups + packed_group) * 32;
                            accumulator = DotInt8Vnni8(accumulator, input_word, weight_group);
                        }

                        alignas(32) int32_t accumulator_values[8];
                        _mm256_store_si256(reinterpret_cast<__m256i*>(accumulator_values), accumulator);
                        alignas(32) int8_t quantized_values[8];
                        if (can_use_fast_requant &&
                            FastQuantizeAccumulator8(accumulator_values,
                                                     fast_requant_scales.data() + output_channel_base,
                                                     output_quantization.zero_point, quantized_values)) {
                            for (int64_t lane = 0; lane < 8; ++lane) {
                                const int64_t output_channel = output_channel_base + lane;
                                const int64_t output_offset =
                                    ((batch * output_shape.c + output_channel) * output_shape.h + output_h) *
                                        output_shape.w + output_w;
                                output[output_offset] = quantized_values[lane];
                            }
                            continue;
                        }
                        for (int64_t lane = 0; lane < 8; ++lane) {
                            const int64_t output_channel = output_channel_base + lane;
                            const int64_t output_offset =
                                ((batch * output_shape.c + output_channel) * output_shape.h + output_h) *
                                    output_shape.w + output_w;
                            if (!int8_detail::QuantizeAccumulator(
                                    accumulator_values[lane],
                                    static_cast<double>(input_quantization.scale) *
                                        weight_quantization.scale_for(static_cast<size_t>(output_channel)),
                                    output_quantization, &output[output_offset])) {
                                failed.store(true, std::memory_order_relaxed);
                                return;
                            }
                        }
                    }
                }
            };

#if defined(FEATHER_WITH_OPENMP)
            const int workers = std::max(1, omp_get_max_threads());
            if (spatial_work_items >= 64 && workers > 1 && !omp_in_parallel()) {
#pragma omp parallel num_threads(workers)
                {
                    const int64_t chunk_size =
                        (spatial_work_items + static_cast<int64_t>(workers) - 1) /
                        static_cast<int64_t>(workers);
                    const int64_t begin = static_cast<int64_t>(omp_get_thread_num()) * chunk_size;
                    const int64_t end = std::min(spatial_work_items, begin + chunk_size);
                    if (begin < end) {
                        compute_vnni_spatial_range(begin, end);
                    }
                }
            } else
#endif
            {
                compute_vnni_spatial_range(0, spatial_work_items);
            }

            const int64_t tail_begin = oc8_blocks * 8;
            for (int64_t batch = 0; batch < input_shape.n; ++batch) {
                for (int64_t output_h = 0; output_h < output_shape.h; ++output_h) {
                    for (int64_t output_w = 0; output_w < output_shape.w; ++output_w) {
                        for (int64_t output_channel = tail_begin; output_channel < output_channels; ++output_channel) {
                            const int64_t work_index =
                                ((batch * output_channels + output_channel) * output_shape.h + output_h) *
                                    output_shape.w + output_w;
                            compute_output(work_index);
                        }
                    }
                }
            }
            return failed.load(std::memory_order_relaxed) ? -1 : 0;
        }

        const int64_t fast_work_items = input_shape.n * output_spatial * oc8_blocks;
        auto compute_oc8_vnni = [&](int64_t work_index) {
            if (failed.load(std::memory_order_relaxed)) {
                return;
            }
            int64_t remaining = work_index;
            const int64_t block = remaining % oc8_blocks;
            remaining /= oc8_blocks;
            const int64_t spatial_index = remaining % output_spatial;
            const int64_t batch = remaining / output_spatial;
            const int64_t output_h = spatial_index / output_shape.w;
            const int64_t output_w = spatial_index % output_shape.w;
            const int64_t output_channel_base = block * 8;
            const int64_t input_x_origin = output_w * param->stride_w - param->pad_w;
            const int64_t input_y_origin = output_h * param->stride_h - param->pad_h;
            const bool interior = input_x_origin >= 0 && input_y_origin >= 0 &&
                                  input_x_origin + (kernel_w - 1) < input_shape.w &&
                                  input_y_origin + (kernel_h - 1) < input_shape.h;
            if (!interior) {
                for (int64_t lane = 0; lane < 8; ++lane) {
                    const int64_t output_channel = output_channel_base + lane;
                    const int64_t scalar_work_index =
                        ((batch * output_channels + output_channel) * output_shape.h + output_h) * output_shape.w +
                        output_w;
                    compute_output(scalar_work_index);
                }
                return;
            }

            alignas(32) int32_t initial_values[8];
            const int32_t* bias = param->bias != nullptr ? param->bias->data<int32_t>() : nullptr;
            for (int64_t lane = 0; lane < 8; ++lane) {
                const int64_t output_channel = output_channel_base + lane;
                const int64_t adjusted = static_cast<int64_t>(bias != nullptr ? bias[output_channel] : 0) -
                                         128LL * static_cast<int64_t>(weight_sums[output_channel]);
                initial_values[lane] = static_cast<int32_t>(adjusted);
            }
            __m256i accumulator = _mm256_load_si256(reinterpret_cast<const __m256i*>(initial_values));
            int64_t packed_group = 0;
            int64_t input_lane = 0;
            uint32_t input_word = 0;
            const int64_t input_plane = input_shape.h * input_shape.w;
            for (int64_t input_channel = 0; input_channel < input_channels_per_group; ++input_channel) {
                const int64_t input_channel_base = (batch * input_shape.c + input_channel) * input_plane;
                for (int64_t kernel_y = 0; kernel_y < kernel_h; ++kernel_y) {
                    const int64_t input_y = input_y_origin + kernel_y;
                    const int8_t* input_row = input + input_channel_base + input_y * input_shape.w + input_x_origin;
                    for (int64_t kernel_x = 0; kernel_x < kernel_w; ++kernel_x) {
                        const uint32_t unsigned_value = static_cast<uint32_t>(
                            static_cast<int32_t>(input_row[kernel_x]) + 128);
                        input_word |= unsigned_value << (8 * input_lane);
                        ++input_lane;
                        if (input_lane == 4) {
                            const int8_t* weight_group =
                                packed_weight + (block * patch_groups + packed_group) * 32;
                            accumulator = DotInt8Vnni8(accumulator, input_word, weight_group);
                            input_word = 0;
                            input_lane = 0;
                            ++packed_group;
                        }
                    }
                }
            }
            if (input_lane != 0) {
                // The padded lanes represent signed zero. VNNI consumes
                // unsigned activation bytes, so use 128 rather than 0; this
                // keeps the -128 * weight correction valid for non-multiple
                // of four patch sizes such as the first YOLO convolution.
                for (int64_t lane = input_lane; lane < 4; ++lane) {
                    input_word |= static_cast<uint32_t>(128U) << (8 * lane);
                }
                const int8_t* weight_group = packed_weight + (block * patch_groups + packed_group) * 32;
                accumulator = DotInt8Vnni8(accumulator, input_word, weight_group);
            }

            alignas(32) int32_t accumulator_values[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(accumulator_values), accumulator);
            alignas(32) int8_t quantized_values[8];
            if (can_use_fast_requant &&
                FastQuantizeAccumulator8(accumulator_values, fast_requant_scales.data() + output_channel_base,
                                         output_quantization.zero_point, quantized_values)) {
                for (int64_t lane = 0; lane < 8; ++lane) {
                    const int64_t output_channel = output_channel_base + lane;
                    const int64_t output_offset =
                        ((batch * output_shape.c + output_channel) * output_shape.h + output_h) * output_shape.w +
                        output_w;
                    output[output_offset] = quantized_values[lane];
                }
                return;
            }
            for (int64_t lane = 0; lane < 8; ++lane) {
                const int64_t output_channel = output_channel_base + lane;
                const double scale = static_cast<double>(input_quantization.scale) *
                                     weight_quantization.scale_for(static_cast<size_t>(output_channel));
                const int64_t output_offset =
                    ((batch * output_shape.c + output_channel) * output_shape.h + output_h) * output_shape.w +
                    output_w;
                if (!int8_detail::QuantizeAccumulator(accumulator_values[lane], scale, output_quantization,
                                                      &output[output_offset])) {
                    failed.store(true, std::memory_order_relaxed);
                    return;
                }
            }
        };

#if defined(FEATHER_WITH_OPENMP)
        const int workers = std::max(1, omp_get_max_threads());
        if (fast_work_items >= 64 && workers > 1 && !omp_in_parallel()) {
#pragma omp parallel for schedule(static) num_threads(workers)
            for (int64_t work_index = 0; work_index < fast_work_items; ++work_index) {
                compute_oc8_vnni(work_index);
            }
        } else
#endif
        {
            for (int64_t work_index = 0; work_index < fast_work_items; ++work_index) {
                compute_oc8_vnni(work_index);
            }
        }

        const int64_t tail_begin = oc8_blocks * 8;
        for (int64_t batch = 0; batch < input_shape.n; ++batch) {
            for (int64_t output_h = 0; output_h < output_shape.h; ++output_h) {
                for (int64_t output_w = 0; output_w < output_shape.w; ++output_w) {
                    for (int64_t output_channel = tail_begin; output_channel < output_channels; ++output_channel) {
                        const int64_t work_index =
                            ((batch * output_channels + output_channel) * output_shape.h + output_h) * output_shape.w +
                            output_w;
                        compute_output(work_index);
                    }
                }
            }
        }
        return failed.load(std::memory_order_relaxed) ? -1 : 0;
    }

    // Process four neighboring output pixels together for 1x1 convolutions.
    // Pair-pack two input channels as [weight, ~weight] so AVX2 maddubs can
    // accumulate eight output channels without repeated INT8 widening.
    const char* pointwise_pair_disabled = std::getenv("FEATHER_DISABLE_INT8_POINTWISE_TILE");
    if (can_use_oc8_fast_path && pointwise_pair_disabled == nullptr && kernel_h == 1 && kernel_w == 1 &&
        param->stride_h > 0 && param->stride_w > 0 &&
        cached_pointwise_maddubs_weight_tensor != nullptr && cached_pointwise_maddubs_weight != nullptr) {
        const int64_t oc8_blocks = output_channels / 8;
        const int64_t pointwise_pair_count = (input_channels_per_group + 1) / 2;
        const size_t pointwise_packed_size =
            static_cast<size_t>(oc8_blocks * pointwise_pair_count * 32);
        if (*cached_pointwise_maddubs_weight_tensor != param->w.get() ||
            cached_pointwise_maddubs_weight->size() != pointwise_packed_size) {
            x86::PackInt8PointwiseWeightsMaddubsPair(
                param->w->data<int8_t>(), output_channels, input_channels_per_group,
                cached_pointwise_maddubs_weight);
            *cached_pointwise_maddubs_weight_tensor = param->w.get();
        }
        if (cached_pointwise_maddubs_weight->size() != pointwise_packed_size) {
            return -1;
        }

        const int8_t* packed_pointwise_weight = cached_pointwise_maddubs_weight->data();
        constexpr int64_t kTileWidth = 4;
        const int64_t tile_count = (output_shape.w + kTileWidth - 1) / kTileWidth;
        const int64_t tile_work_items = input_shape.n * output_shape.h * tile_count * oc8_blocks;
        auto compute_pointwise_pair_tile = [&](int64_t work_index) {
            if (failed.load(std::memory_order_relaxed)) {
                return;
            }
            int64_t remaining = work_index;
            const int64_t block = remaining % oc8_blocks;
            remaining /= oc8_blocks;
            const int64_t tile_index = remaining % tile_count;
            remaining /= tile_count;
            const int64_t output_h = remaining % output_shape.h;
            const int64_t batch = remaining / output_shape.h;
            const int64_t output_w_base = tile_index * kTileWidth;
            const int64_t actual_width = std::min<int64_t>(kTileWidth, output_shape.w - output_w_base);
            const int64_t input_x_origin = output_w_base * param->stride_w - param->pad_w;
            const int64_t input_y_origin = output_h * param->stride_h - param->pad_h;
            const bool interior = actual_width == kTileWidth && input_x_origin >= 0 && input_y_origin >= 0 &&
                                  input_y_origin < input_shape.h &&
                                  input_x_origin + (actual_width - 1) * param->stride_w < input_shape.w;
            if (!interior) {
                for (int64_t tile = 0; tile < actual_width; ++tile) {
                    for (int64_t lane = 0; lane < 8; ++lane) {
                        const int64_t output_channel = block * 8 + lane;
                        const int64_t scalar_work_index =
                            ((batch * output_channels + output_channel) * output_shape.h + output_h) *
                                output_shape.w + output_w_base + tile;
                        compute_output(scalar_work_index);
                    }
                }
                return;
            }

            const int64_t output_channel_base = block * 8;
            const __m256i bias = param->bias == nullptr
                                     ? _mm256_setzero_si256()
                                     : _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                                           param->bias->data<int32_t>() + output_channel_base));
            __m256i accumulators[kTileWidth] = {bias, bias, bias, bias};
            const int64_t input_plane = input_shape.h * input_shape.w;
            for (int64_t pair = 0; pair < pointwise_pair_count; ++pair) {
                const int64_t input_channel0 = pair * 2;
                const int64_t input_channel1 = input_channel0 + 1;
                const int64_t input_channel0_base =
                    (batch * input_shape.c + input_channel0) * input_plane;
                const int8_t* input_row0 = input + input_channel0_base +
                                            input_y_origin * input_shape.w + input_x_origin;
                const int8_t* input_row1 = nullptr;
                if (input_channel1 < input_channels_per_group) {
                    const int64_t input_channel1_base =
                        (batch * input_shape.c + input_channel1) * input_plane;
                    input_row1 = input + input_channel1_base +
                                 input_y_origin * input_shape.w + input_x_origin;
                }
                const __m256i pair_weights = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                    packed_pointwise_weight + (block * pointwise_pair_count + pair) * 32));
                for (int64_t tile = 0; tile < kTileWidth; ++tile) {
                    const int8_t input_value0 = input_row0[tile * param->stride_w];
                    const int8_t input_value1 = input_row1 == nullptr
                        ? static_cast<int8_t>(0)
                        : input_row1[tile * param->stride_w];
                    accumulators[tile] = x86::DotInt8Oc8MaddubsPairPackedWeights(
                        accumulators[tile], input_value0, input_value1, pair_weights);
                }
            }

            alignas(32) int32_t accumulator_values[kTileWidth][8];
            alignas(32) int8_t quantized_values[kTileWidth][8];
            for (int64_t tile = 0; tile < kTileWidth; ++tile) {
                _mm256_store_si256(reinterpret_cast<__m256i*>(accumulator_values[tile]), accumulators[tile]);
                const int64_t output_w = output_w_base + tile;
                if (can_use_fast_requant &&
                    FastQuantizeAccumulator8(accumulator_values[tile],
                                             fast_requant_scales.data() + output_channel_base,
                                             output_quantization.zero_point, quantized_values[tile])) {
                    for (int64_t lane = 0; lane < 8; ++lane) {
                        const int64_t output_channel = output_channel_base + lane;
                        const int64_t output_offset =
                            ((batch * output_shape.c + output_channel) * output_shape.h + output_h) *
                                output_shape.w + output_w;
                        output[output_offset] = quantized_values[tile][lane];
                    }
                    continue;
                }
                for (int64_t lane = 0; lane < 8; ++lane) {
                    const int64_t output_channel = output_channel_base + lane;
                    const int64_t output_offset =
                        ((batch * output_shape.c + output_channel) * output_shape.h + output_h) *
                            output_shape.w + output_w;
                    if (!int8_detail::QuantizeAccumulator(
                            accumulator_values[tile][lane],
                            static_cast<double>(input_quantization.scale) *
                                weight_quantization.scale_for(static_cast<size_t>(output_channel)),
                            output_quantization, &output[output_offset])) {
                        failed.store(true, std::memory_order_relaxed);
                        return;
                    }
                }
            }
        };

#if defined(FEATHER_WITH_OPENMP)
        const int workers = std::max(1, omp_get_max_threads());
        if (tile_work_items >= 64 && workers > 1 && !omp_in_parallel()) {
#pragma omp parallel for schedule(static) num_threads(workers)
            for (int64_t work_index = 0; work_index < tile_work_items; ++work_index) {
                compute_pointwise_pair_tile(work_index);
            }
        } else
#endif
        {
            for (int64_t work_index = 0; work_index < tile_work_items; ++work_index) {
                compute_pointwise_pair_tile(work_index);
            }
        }

        const int64_t tail_begin = oc8_blocks * 8;
        for (int64_t batch = 0; batch < input_shape.n; ++batch) {
            for (int64_t output_h = 0; output_h < output_shape.h; ++output_h) {
                for (int64_t output_w = 0; output_w < output_shape.w; ++output_w) {
                    for (int64_t output_channel = tail_begin; output_channel < output_channels; ++output_channel) {
                        const int64_t scalar_work_index =
                            ((batch * output_channels + output_channel) * output_shape.h + output_h) *
                                output_shape.w + output_w;
                        compute_output(scalar_work_index);
                    }
                }
            }
        }
        return failed.load(std::memory_order_relaxed) ? -1 : 0;
    }

    // The general OC8 kernel reloads and widens the same eight weights for
    // every output pixel; this tile keeps one widened weight vector live
    // while broadcasting four input values.  It works for both unit and
    // strided pointwise convolutions.  Boundary and short-width tiles use
    // the checked scalar helper below.
    const char* pointwise_tile_disabled = std::getenv("FEATHER_DISABLE_INT8_POINTWISE_TILE");
    if (can_use_oc8_fast_path && pointwise_tile_disabled == nullptr && kernel_h == 1 && kernel_w == 1 &&
        param->stride_h > 0 && param->stride_w > 0) {
        const int64_t oc8_blocks = output_channels / 8;
        const size_t packed_size = static_cast<size_t>(oc8_blocks * patch_size * 8);
        if (*cached_weight_tensor != param->w.get() || cached_packed_weight->size() != packed_size) {
            x86::PackInt8ConvWeightsOc8(param->w->data<int8_t>(), output_channels, patch_size,
                                        cached_packed_weight);
            *cached_weight_tensor = param->w.get();
        }
        if (cached_packed_weight->size() != packed_size) {
            return -1;
        }

        const int8_t* packed_weight = cached_packed_weight->data();
        constexpr int64_t kTileWidth = 4;
        const int64_t tile_count = (output_shape.w + kTileWidth - 1) / kTileWidth;
        const int64_t tile_work_items = input_shape.n * output_shape.h * tile_count * oc8_blocks;
        auto compute_pointwise_tile = [&](int64_t work_index) {
            if (failed.load(std::memory_order_relaxed)) {
                return;
            }
            int64_t remaining = work_index;
            const int64_t block = remaining % oc8_blocks;
            remaining /= oc8_blocks;
            const int64_t tile_index = remaining % tile_count;
            remaining /= tile_count;
            const int64_t output_h = remaining % output_shape.h;
            const int64_t batch = remaining / output_shape.h;
            const int64_t output_w_base = tile_index * kTileWidth;
            const int64_t actual_width = std::min<int64_t>(kTileWidth, output_shape.w - output_w_base);
            const int64_t input_x_origin = output_w_base * param->stride_w - param->pad_w;
            const int64_t input_y_origin = output_h * param->stride_h - param->pad_h;
            const bool interior = actual_width == kTileWidth && input_x_origin >= 0 && input_y_origin >= 0 &&
                                  input_y_origin < input_shape.h &&
                                  input_x_origin + (actual_width - 1) * param->stride_w < input_shape.w;
            if (!interior) {
                for (int64_t tile = 0; tile < actual_width; ++tile) {
                    for (int64_t lane = 0; lane < 8; ++lane) {
                        const int64_t output_channel = block * 8 + lane;
                        const int64_t scalar_work_index =
                            ((batch * output_channels + output_channel) * output_shape.h + output_h) *
                                output_shape.w + output_w_base + tile;
                        compute_output(scalar_work_index);
                    }
                }
                return;
            }

            const int64_t output_channel_base = block * 8;
            const __m256i bias = param->bias == nullptr
                                     ? _mm256_setzero_si256()
                                     : _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                                           param->bias->data<int32_t>() + output_channel_base));
            __m256i accumulators[kTileWidth] = {bias, bias, bias, bias};
            const int64_t input_plane = input_shape.h * input_shape.w;
            for (int64_t input_channel = 0; input_channel < input_channels_per_group; ++input_channel) {
                const int64_t input_channel_base = (batch * input_shape.c + input_channel) * input_plane;
                const int8_t* input_row = input + input_channel_base + input_y_origin * input_shape.w + input_x_origin;
                const int8_t* packed_row = packed_weight + (block * patch_size + input_channel) * 8;
                const __m256i weight_values =
                    _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(packed_row)));
                for (int64_t tile = 0; tile < kTileWidth; ++tile) {
                    const __m256i input_value =
                        _mm256_set1_epi32(static_cast<int32_t>(input_row[tile * param->stride_w]));
                    accumulators[tile] =
                        _mm256_add_epi32(accumulators[tile], _mm256_mullo_epi32(weight_values, input_value));
                }
            }

            alignas(32) int32_t accumulator_values[kTileWidth][8];
            alignas(32) int8_t quantized_values[kTileWidth][8];
            for (int64_t tile = 0; tile < kTileWidth; ++tile) {
                _mm256_store_si256(reinterpret_cast<__m256i*>(accumulator_values[tile]), accumulators[tile]);
                const int64_t output_w = output_w_base + tile;
                if (can_use_fast_requant &&
                    FastQuantizeAccumulator8(accumulator_values[tile], fast_requant_scales.data() + output_channel_base,
                                             output_quantization.zero_point, quantized_values[tile])) {
                    for (int64_t lane = 0; lane < 8; ++lane) {
                        const int64_t output_channel = output_channel_base + lane;
                        const int64_t output_offset =
                            ((batch * output_shape.c + output_channel) * output_shape.h + output_h) * output_shape.w +
                            output_w;
                        output[output_offset] = quantized_values[tile][lane];
                    }
                    continue;
                }
                for (int64_t lane = 0; lane < 8; ++lane) {
                    const int64_t output_channel = output_channel_base + lane;
                    const double scale = static_cast<double>(input_quantization.scale) *
                                         weight_quantization.scale_for(static_cast<size_t>(output_channel));
                    const int64_t output_offset =
                        ((batch * output_shape.c + output_channel) * output_shape.h + output_h) * output_shape.w +
                        output_w;
                    if (!int8_detail::QuantizeAccumulator(accumulator_values[tile][lane], scale, output_quantization,
                                                          &output[output_offset])) {
                        failed.store(true, std::memory_order_relaxed);
                        return;
                    }
                }
            }
        };

#if defined(FEATHER_WITH_OPENMP)
        if (tile_work_items >= 64 && !omp_in_parallel()) {
            const int workers = std::max(1, omp_get_max_threads());
#pragma omp parallel for schedule(static) num_threads(workers)
            for (int64_t work_index = 0; work_index < tile_work_items; ++work_index) {
                compute_pointwise_tile(work_index);
            }
        } else
#endif
        {
            for (int64_t work_index = 0; work_index < tile_work_items; ++work_index) {
                compute_pointwise_tile(work_index);
            }
        }

        const int64_t tail_begin = oc8_blocks * 8;
        for (int64_t batch = 0; batch < input_shape.n; ++batch) {
            for (int64_t output_h = 0; output_h < output_shape.h; ++output_h) {
                for (int64_t output_w = 0; output_w < output_shape.w; ++output_w) {
                    for (int64_t output_channel = tail_begin; output_channel < output_channels; ++output_channel) {
                        const int64_t work_index =
                            ((batch * output_channels + output_channel) * output_shape.h + output_h) * output_shape.w +
                            output_w;
                        compute_output(work_index);
                    }
                }
            }
        }
        return failed.load(std::memory_order_relaxed) ? -1 : 0;
    }

   // Reuse one pair of packed weights across four neighboring output pixels.
    // This is the INT8 equivalent of the FP32 direct OC8 spatial-reuse kernel:
    // the pair microkernel still computes eight output channels, while the
    // tile avoids reloading and rebuilding the same weight vectors four times.
    const char* pair_tile_enabled = std::getenv("FEATHER_ENABLE_INT8_PAIR_TILE");
    if (can_use_oc8_fast_path && pair_tile_enabled != nullptr && pair_tile_enabled[0] == '1' &&
        kernel_h == 3 && kernel_w == 3 &&
        param->stride_h == 1 &&
        param->stride_w == 1 && param->pad_h == 1 && param->pad_w == 1 &&
        cached_maddubs_weight_tensor != nullptr && cached_maddubs_weight != nullptr) {
        int64_t max_bias_abs = 0;
        if (param->bias != nullptr) {
            const int32_t* bias = param->bias->data<int32_t>();
            for (int64_t output_channel = 0; output_channel < output_channels; ++output_channel) {
                const int64_t value = static_cast<int64_t>(bias[output_channel]);
                const int64_t absolute_value = value < 0 ? -value : value;
                max_bias_abs = std::max(max_bias_abs, absolute_value);
            }
        }
        const int64_t int32_max = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
        const int64_t max_product = 128 * 128;
        const bool safe_accumulation = max_bias_abs <= int32_max &&
                                       patch_size <= (int32_max - max_bias_abs) / max_product;
        if (safe_accumulation) {
            const int64_t oc8_blocks = output_channels / 8;
            const int64_t pair_width = (kernel_w + 1) / 2;
            const int64_t pair_rows_per_block = input_channels_per_group * kernel_h * pair_width;
            const size_t pair_packed_size = static_cast<size_t>(oc8_blocks * pair_rows_per_block * 32);
            if (*cached_maddubs_weight_tensor != param->w.get() ||
                cached_maddubs_weight->size() != pair_packed_size) {
                x86::PackInt8ConvWeightsMaddubsPair(
                    param->w->data<int8_t>(), output_channels, input_channels_per_group,
                    kernel_h, kernel_w, cached_maddubs_weight);
                *cached_maddubs_weight_tensor = param->w.get();
            }
            if (cached_maddubs_weight->size() != pair_packed_size) {
                return -1;
            }

            const int8_t* packed_maddubs_weight = cached_maddubs_weight->data();
            constexpr int64_t kTileWidth = 4;
            const int64_t tile_count = (output_shape.w + kTileWidth - 1) / kTileWidth;
            const int64_t tile_work_items = input_shape.n * output_shape.h * tile_count * oc8_blocks;
            auto compute_pair_tile = [&](int64_t work_index) {
                if (failed.load(std::memory_order_relaxed)) {
                    return;
                }
                int64_t remaining = work_index;
                const int64_t block = remaining % oc8_blocks;
                remaining /= oc8_blocks;
                const int64_t tile_index = remaining % tile_count;
                remaining /= tile_count;
                const int64_t output_h = remaining % output_shape.h;
                const int64_t batch = remaining / output_shape.h;
                const int64_t output_w_base = tile_index * kTileWidth;
                const int64_t actual_width = std::min<int64_t>(kTileWidth, output_shape.w - output_w_base);
                const int64_t input_x_origin = output_w_base - param->pad_w;
                const int64_t input_y_origin = output_h - param->pad_h;
                const bool interior = actual_width == kTileWidth && input_x_origin >= 0 && input_y_origin >= 0 &&
                                      input_x_origin + kTileWidth - 1 + kernel_w - 1 < input_shape.w &&
                                      input_y_origin + kernel_h - 1 < input_shape.h;
                if (!interior) {
                    for (int64_t tile = 0; tile < actual_width; ++tile) {
                        for (int64_t lane = 0; lane < 8; ++lane) {
                            const int64_t output_channel = block * 8 + lane;
                            const int64_t scalar_work_index =
                                ((batch * output_channels + output_channel) * output_shape.h + output_h) *
                                    output_shape.w + output_w_base + tile;
                            compute_output(scalar_work_index);
                        }
                    }
                    return;
                }

                const int64_t output_channel_base = block * 8;
                const __m256i bias = param->bias == nullptr
                                         ? _mm256_setzero_si256()
                                         : _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                                               param->bias->data<int32_t>() + output_channel_base));
                __m256i accumulators[kTileWidth] = {bias, bias, bias, bias};
                const int64_t input_plane = input_shape.h * input_shape.w;
                for (int64_t input_channel = 0; input_channel < input_channels_per_group; ++input_channel) {
                    const int64_t input_channel_base = (batch * input_shape.c + input_channel) * input_plane;
                    for (int64_t kernel_y = 0; kernel_y < kernel_h; ++kernel_y) {
                        const int8_t* input_row = input + input_channel_base +
                                                  (input_y_origin + kernel_y) * input_shape.w + input_x_origin;
                        const int64_t pair_row_base =
                            block * pair_rows_per_block + (input_channel * kernel_h + kernel_y) * pair_width;
                        const int8_t* first_pair = packed_maddubs_weight + pair_row_base * 32;
                        const __m256i first_weights =
                            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(first_pair));
                        // The final 3x3 kernel element is the first element of
                        // the second packed pair; it is already [w, ~w].
                        const __m128i last_maddubs_weights =
                            _mm_loadu_si128(reinterpret_cast<const __m128i*>(first_pair + 32));
                        for (int64_t tile = 0; tile < kTileWidth; ++tile) {
                            accumulators[tile] = x86::DotInt8Oc8MaddubsPairPackedWeights(
                                accumulators[tile], input_row[tile], input_row[tile + 1], first_weights);
                            accumulators[tile] = x86::DotInt8Oc8MaddubsPacked(
                                accumulators[tile], input_row[tile + 2], last_maddubs_weights);
                        }
                    }
                }

                alignas(32) int32_t accumulator_values[kTileWidth][8];
                alignas(32) int8_t quantized_values[kTileWidth][8];
                for (int64_t tile = 0; tile < kTileWidth; ++tile) {
                    _mm256_store_si256(reinterpret_cast<__m256i*>(accumulator_values[tile]), accumulators[tile]);
                    const int64_t output_w = output_w_base + tile;
                    if (can_use_fast_requant &&
                        FastQuantizeAccumulator8(accumulator_values[tile],
                                                 fast_requant_scales.data() + output_channel_base,
                                                 output_quantization.zero_point, quantized_values[tile])) {
                        for (int64_t lane = 0; lane < 8; ++lane) {
                            const int64_t output_channel = output_channel_base + lane;
                            const int64_t output_offset =
                                ((batch * output_shape.c + output_channel) * output_shape.h + output_h) *
                                    output_shape.w + output_w;
                            output[output_offset] = quantized_values[tile][lane];
                        }
                        continue;
                    }
                    for (int64_t lane = 0; lane < 8; ++lane) {
                        const int64_t output_channel = output_channel_base + lane;
                        const int64_t output_offset =
                            ((batch * output_shape.c + output_channel) * output_shape.h + output_h) *
                                output_shape.w + output_w;
                        if (!int8_detail::QuantizeAccumulator(
                                accumulator_values[tile][lane],
                                static_cast<double>(input_quantization.scale) *
                                    weight_quantization.scale_for(static_cast<size_t>(output_channel)),
                                output_quantization, &output[output_offset])) {
                            failed.store(true, std::memory_order_relaxed);
                            return;
                        }
                    }
                }
            };

#if defined(FEATHER_WITH_OPENMP)
            const int workers = std::max(1, omp_get_max_threads());
            if (tile_work_items >= 64 && workers > 1 && !omp_in_parallel()) {
#pragma omp parallel for schedule(static) num_threads(workers)
                for (int64_t work_index = 0; work_index < tile_work_items; ++work_index) {
                    compute_pair_tile(work_index);
                }
            } else
#endif
            {
                for (int64_t work_index = 0; work_index < tile_work_items; ++work_index) {
                    compute_pair_tile(work_index);
                }
            }

            const int64_t tail_begin = oc8_blocks * 8;
            for (int64_t batch = 0; batch < input_shape.n; ++batch) {
                for (int64_t output_h = 0; output_h < output_shape.h; ++output_h) {
                    for (int64_t output_w = 0; output_w < output_shape.w; ++output_w) {
                        for (int64_t output_channel = tail_begin; output_channel < output_channels; ++output_channel) {
                            const int64_t scalar_work_index =
                                ((batch * output_channels + output_channel) * output_shape.h + output_h) *
                                    output_shape.w + output_w;
                            compute_output(scalar_work_index);
                        }
                    }
                }
            }
            return failed.load(std::memory_order_relaxed) ? -1 : 0;
        }
    }

    // Keep the older general spatial tile as an opt-in experiment. The default
    // OC8 path below uses the pairwise dot-product microkernel instead.
    const char* spatial_tile_enabled = std::getenv("FEATHER_ENABLE_INT8_SPATIAL_TILE");
    if (can_use_oc8_fast_path && spatial_tile_enabled != nullptr && spatial_tile_enabled[0] == '1' &&
        (kernel_h > 1 || kernel_w > 1) && param->stride_h == 1 && param->stride_w == 1) {
        const int64_t oc8_blocks = output_channels / 8;
        const size_t packed_size = static_cast<size_t>(oc8_blocks * patch_size * 8);
        if (*cached_weight_tensor != param->w.get() || cached_packed_weight->size() != packed_size) {
            x86::PackInt8ConvWeightsOc8(param->w->data<int8_t>(), output_channels, patch_size,
                                        cached_packed_weight);
            *cached_weight_tensor = param->w.get();
        }
        if (cached_packed_weight->size() != packed_size) {
            return -1;
        }

        const int8_t* packed_weight = cached_packed_weight->data();
        constexpr int64_t kTileWidth = 4;
        const int64_t tile_count = (output_shape.w + kTileWidth - 1) / kTileWidth;
        const int64_t tile_work_items = input_shape.n * output_shape.h * tile_count * oc8_blocks;
        auto compute_oc8_tile = [&](int64_t work_index) {
            if (failed.load(std::memory_order_relaxed)) {
                return;
            }
            int64_t remaining = work_index;
            const int64_t block = remaining % oc8_blocks;
            remaining /= oc8_blocks;
            const int64_t tile_index = remaining % tile_count;
            remaining /= tile_count;
            const int64_t output_h = remaining % output_shape.h;
            const int64_t batch = remaining / output_shape.h;
            const int64_t output_w_base = tile_index * kTileWidth;
            const int64_t actual_width = std::min<int64_t>(kTileWidth, output_shape.w - output_w_base);
            const int64_t input_x_origin = output_w_base * param->stride_w - param->pad_w;
            const int64_t input_y_origin = output_h * param->stride_h - param->pad_h;
            const bool interior = actual_width == kTileWidth && input_x_origin >= 0 && input_y_origin >= 0 &&
                                  input_x_origin + (actual_width - 1) * param->stride_w + kernel_w - 1 < input_shape.w &&
                                  input_y_origin + kernel_h - 1 < input_shape.h;
            if (!interior) {
                for (int64_t tile = 0; tile < actual_width; ++tile) {
                    for (int64_t lane = 0; lane < 8; ++lane) {
                        const int64_t output_channel = block * 8 + lane;
                        const int64_t scalar_work_index =
                            ((batch * output_channels + output_channel) * output_shape.h + output_h) *
                                output_shape.w + output_w_base + tile;
                        compute_output(scalar_work_index);
                    }
                }
                return;
            }

            const int64_t output_channel_base = block * 8;
            const __m256i bias = param->bias == nullptr
                                     ? _mm256_setzero_si256()
                                     : _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                                           param->bias->data<int32_t>() + output_channel_base));
            __m256i accumulators[4] = {bias, bias, bias, bias};
            const int64_t input_plane = input_shape.h * input_shape.w;
            for (int64_t input_channel = 0; input_channel < input_channels_per_group; ++input_channel) {
                const int64_t input_channel_base = (batch * input_shape.c + input_channel) * input_plane;
                for (int64_t kernel_y = 0; kernel_y < kernel_h; ++kernel_y) {
                    const int8_t* input_row = input + input_channel_base +
                                              (input_y_origin + kernel_y) * input_shape.w + input_x_origin;
                    const int8_t* packed_row = packed_weight +
                                                (block * patch_size + input_channel * kernel_h * kernel_w +
                                                 kernel_y * kernel_w) * 8;
                    for (int64_t kernel_x = 0; kernel_x < kernel_w; ++kernel_x) {
                        const __m128i weight_values =
                            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(packed_row));
                        const __m128i inverted_weight_values =
                            _mm_xor_si128(weight_values, _mm_set1_epi8(static_cast<char>(-1)));
                        const __m128i maddubs_weights =
                            _mm_unpacklo_epi8(weight_values, inverted_weight_values);
                        for (int64_t tile = 0; tile < kTileWidth; ++tile) {
                            accumulators[tile] = x86::DotInt8Oc8MaddubsPacked(
                                accumulators[tile], input_row[tile * param->stride_w],
                                maddubs_weights);
                        }
                        ++input_row;
                        packed_row += 8;
                    }
                }
            }

            alignas(32) int32_t accumulator_values[4][8];
            alignas(32) int8_t quantized_values[4][8];
            for (int64_t tile = 0; tile < kTileWidth; ++tile) {
                _mm256_store_si256(reinterpret_cast<__m256i*>(accumulator_values[tile]), accumulators[tile]);
                if (can_use_fast_requant &&
                    FastQuantizeAccumulator8(accumulator_values[tile], fast_requant_scales.data() + output_channel_base,
                                             output_quantization.zero_point, quantized_values[tile])) {
                    for (int64_t lane = 0; lane < 8; ++lane) {
                        const int64_t output_channel = output_channel_base + lane;
                        const int64_t output_offset =
                            ((batch * output_shape.c + output_channel) * output_shape.h + output_h) *
                                output_shape.w + output_w_base + tile;
                        output[output_offset] = quantized_values[tile][lane];
                    }
                    continue;
                }
                for (int64_t lane = 0; lane < 8; ++lane) {
                    const int64_t output_channel = output_channel_base + lane;
                    const double scale = static_cast<double>(input_quantization.scale) *
                                         weight_quantization.scale_for(static_cast<size_t>(output_channel));
                    const int64_t output_offset =
                        ((batch * output_shape.c + output_channel) * output_shape.h + output_h) *
                            output_shape.w + output_w_base + tile;
                    if (!int8_detail::QuantizeAccumulator(accumulator_values[tile][lane], scale,
                                                          output_quantization, &output[output_offset])) {
                        failed.store(true, std::memory_order_relaxed);
                        return;
                    }
                }
            }
        };

#if defined(FEATHER_WITH_OPENMP)
        const int workers = std::max(1, omp_get_max_threads());
        if (tile_work_items >= 64 && workers > 1 && !omp_in_parallel()) {
#pragma omp parallel for schedule(static) num_threads(workers)
            for (int64_t work_index = 0; work_index < tile_work_items; ++work_index) {
                compute_oc8_tile(work_index);
            }
        } else
#endif
        {
            for (int64_t work_index = 0; work_index < tile_work_items; ++work_index) {
                compute_oc8_tile(work_index);
            }
        }

        const int64_t tail_begin = oc8_blocks * 8;
        for (int64_t batch = 0; batch < input_shape.n; ++batch) {
            for (int64_t output_h = 0; output_h < output_shape.h; ++output_h) {
                for (int64_t output_w = 0; output_w < output_shape.w; ++output_w) {
                    for (int64_t output_channel = tail_begin; output_channel < output_channels; ++output_channel) {
                        const int64_t scalar_work_index =
                            ((batch * output_channels + output_channel) * output_shape.h + output_h) *
                                output_shape.w + output_w;
                        compute_output(scalar_work_index);
                    }
                }
            }
        }
        return failed.load(std::memory_order_relaxed) ? -1 : 0;
    }

    if (can_use_oc8_fast_path) {
        // The vector accumulator is INT32.  Reject shapes whose conservative
        // worst-case bound could overflow, allowing the checked scalar path to
        // preserve the existing error contract for unusual inputs.
        int64_t max_bias_abs = 0;
        if (param->bias != nullptr) {
            const int32_t* bias = param->bias->data<int32_t>();
            for (int64_t output_channel = 0; output_channel < output_channels; ++output_channel) {
                const int64_t value = static_cast<int64_t>(bias[output_channel]);
                const int64_t absolute_value = value < 0 ? -value : value;
                max_bias_abs = std::max(max_bias_abs, absolute_value);
            }
        }
        const int64_t int32_max = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
        const int64_t max_product = 128 * 128;
        can_use_oc8_fast_path = max_bias_abs <= int32_max &&
                                patch_size <= (int32_max - max_bias_abs) / max_product;
    }

    if (can_use_oc8_fast_path) {
        const int64_t oc8_blocks = output_channels / 8;
        const size_t packed_size = static_cast<size_t>(oc8_blocks * patch_size * 8);
        if (*cached_weight_tensor != param->w.get() || cached_packed_weight->size() != packed_size) {
            x86::PackInt8ConvWeightsOc8(param->w->data<int8_t>(), output_channels, patch_size,
                                        cached_packed_weight);
            *cached_weight_tensor = param->w.get();
        }
    if (cached_packed_weight->size() != packed_size) {
        return -1;
    }

        const bool use_maddubs_pairs = param->stride_w == 1 && param->dilation_w == 1 && kernel_w >= 2 &&
                                       cached_maddubs_weight_tensor != nullptr && cached_maddubs_weight != nullptr;
        const int64_t pair_width = (kernel_w + 1) / 2;
        const int64_t pair_rows_per_block = input_channels_per_group * kernel_h * pair_width;
        const size_t pair_packed_size = static_cast<size_t>(oc8_blocks * pair_rows_per_block * 32);
        if (use_maddubs_pairs) {
            if (*cached_maddubs_weight_tensor != param->w.get() ||
                cached_maddubs_weight->size() != pair_packed_size) {
                x86::PackInt8ConvWeightsMaddubsPair(
                    param->w->data<int8_t>(), output_channels, input_channels_per_group,
                    kernel_h, kernel_w, cached_maddubs_weight);
                *cached_maddubs_weight_tensor = param->w.get();
            }
            if (cached_maddubs_weight->size() != pair_packed_size) {
                return -1;
            }
        }

        const int8_t* packed_weight = cached_packed_weight->data();
        const int8_t* packed_maddubs_weight = use_maddubs_pairs ? cached_maddubs_weight->data() : nullptr;
        const int64_t output_spatial = output_shape.h * output_shape.w;

        // The OC8 work item below walks one output position for every OC8
        // block.  For the common 3x3/s1/p1 case that makes the same input
        // patch get decoded once per output block.  Build the patch once and
        // then consume it across all blocks instead.  This mirrors the
        // spatial-major organization used by the optimized FP32 kernel while
        // keeping the existing signed-INT8 maddubs transform and requantization
        // contract unchanged.
        const char* spatial_major_enabled = std::getenv("FEATHER_ENABLE_INT8_SPATIAL_MAJOR");
        const bool use_spatial_major_3x3 =
            use_maddubs_pairs && kernel_h == 3 && kernel_w == 3 &&
            param->stride_h == 1 && param->stride_w == 1 &&
            param->dilation_h == 1 && param->dilation_w == 1 &&
            param->pad_h == 1 && param->pad_w == 1 && param->group == 1 &&
            spatial_major_enabled != nullptr && spatial_major_enabled[0] == '1';
        if (use_spatial_major_3x3) {
            const int64_t spatial_work_items = input_shape.n * output_spatial;
            auto compute_spatial_major_range = [&](int64_t begin, int64_t end) {
                std::vector<int8_t> input_patch(static_cast<size_t>(patch_size));
                for (int64_t work_index = begin; work_index < end; ++work_index) {
                    if (failed.load(std::memory_order_relaxed)) {
                        return;
                    }
                    const int64_t batch = work_index / output_spatial;
                    const int64_t spatial_index = work_index % output_spatial;
                    const int64_t output_h = spatial_index / output_shape.w;
                    const int64_t output_w = spatial_index % output_shape.w;
                    const int64_t input_y_origin = output_h - param->pad_h;
                    const int64_t input_x_origin = output_w - param->pad_w;
                    const bool interior = input_y_origin >= 0 && input_x_origin >= 0 &&
                                          input_y_origin + 2 < input_shape.h &&
                                          input_x_origin + 2 < input_shape.w;
                    const int64_t input_plane = input_shape.h * input_shape.w;

                    for (int64_t input_channel = 0; input_channel < input_channels_per_group; ++input_channel) {
                        const int64_t input_channel_base =
                            (batch * input_shape.c + input_channel) * input_plane;
                        int8_t* patch_channel = input_patch.data() + input_channel * 9;
                        if (interior) {
                            const int8_t* input_row = input + input_channel_base + input_y_origin * input_shape.w +
                                                      input_x_origin;
                            std::copy_n(input_row, 3, patch_channel);
                            std::copy_n(input_row + input_shape.w, 3, patch_channel + 3);
                            std::copy_n(input_row + 2 * input_shape.w, 3, patch_channel + 6);
                        } else {
                            for (int64_t kernel_y = 0; kernel_y < 3; ++kernel_y) {
                                const int64_t input_y = input_y_origin + kernel_y;
                                for (int64_t kernel_x = 0; kernel_x < 3; ++kernel_x) {
                                    const int64_t input_x = input_x_origin + kernel_x;
                                    int8_t value = 0;
                                    if (input_y >= 0 && input_y < input_shape.h && input_x >= 0 &&
                                        input_x < input_shape.w) {
                                        value = input[input_channel_base + input_y * input_shape.w + input_x];
                                    }
                                    input_patch[static_cast<size_t>(input_channel * 9 + kernel_y * 3 + kernel_x)] =
                                        value;
                                }
                            }
                        }
                    }

                    for (int64_t block = 0; block < oc8_blocks; ++block) {
                        const int64_t output_channel_base = block * 8;
                        __m256i accumulator =
                            param->bias == nullptr
                                ? _mm256_setzero_si256()
                                : _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                                      param->bias->data<int32_t>() + output_channel_base));
                        for (int64_t input_channel = 0; input_channel < input_channels_per_group; ++input_channel) {
                            const int8_t* patch_channel = input_patch.data() + input_channel * 9;
                            for (int64_t kernel_y = 0; kernel_y < 3; ++kernel_y) {
                                const int64_t pair_row = (input_channel * 3 + kernel_y) * 2;
                                const int8_t* pair_weight = packed_maddubs_weight +
                                    (block * pair_rows_per_block + pair_row) * 32;
                                const __m128i third_weights = _mm_loadu_si128(
                                    reinterpret_cast<const __m128i*>(pair_weight + 32));
                                accumulator = x86::DotInt8Oc8MaddubsPairPacked(
                                    accumulator, patch_channel[kernel_y * 3], patch_channel[kernel_y * 3 + 1],
                                    pair_weight);
                                accumulator = x86::DotInt8Oc8MaddubsPacked(
                                    accumulator, patch_channel[kernel_y * 3 + 2], third_weights);
                            }
                        }

                        alignas(32) int32_t accumulator_values[8];
                        _mm256_store_si256(reinterpret_cast<__m256i*>(accumulator_values), accumulator);
                        alignas(32) int8_t quantized_values[8];
                        if (can_use_fast_requant &&
                            FastQuantizeAccumulator8(accumulator_values,
                                                     fast_requant_scales.data() + output_channel_base,
                                                     output_quantization.zero_point, quantized_values)) {
                            for (int64_t lane = 0; lane < 8; ++lane) {
                                const int64_t output_channel = output_channel_base + lane;
                                const int64_t output_offset =
                                    ((batch * output_shape.c + output_channel) * output_shape.h + output_h) *
                                        output_shape.w + output_w;
                                output[output_offset] = quantized_values[lane];
                            }
                            continue;
                        }
                        for (int64_t lane = 0; lane < 8; ++lane) {
                            const int64_t output_channel = output_channel_base + lane;
                            const int64_t output_offset =
                                ((batch * output_shape.c + output_channel) * output_shape.h + output_h) *
                                    output_shape.w + output_w;
                            if (!int8_detail::QuantizeAccumulator(
                                    accumulator_values[lane],
                                    static_cast<double>(input_quantization.scale) *
                                        weight_quantization.scale_for(static_cast<size_t>(output_channel)),
                                    output_quantization, &output[output_offset])) {
                                failed.store(true, std::memory_order_relaxed);
                                return;
                            }
                        }
                    }
                }
            };

#if defined(FEATHER_WITH_OPENMP)
            const int workers = std::max(1, omp_get_max_threads());
            if (spatial_work_items >= 64 && workers > 1 && !omp_in_parallel()) {
#pragma omp parallel num_threads(workers)
                {
                    const int64_t chunk_size =
                        (spatial_work_items + static_cast<int64_t>(workers) - 1) / static_cast<int64_t>(workers);
                    const int64_t begin = static_cast<int64_t>(omp_get_thread_num()) * chunk_size;
                    const int64_t end = std::min(spatial_work_items, begin + chunk_size);
                    if (begin < end) {
                        compute_spatial_major_range(begin, end);
                    }
                }
            } else
#endif
            {
                compute_spatial_major_range(0, spatial_work_items);
            }

            const int64_t tail_begin = oc8_blocks * 8;
            for (int64_t batch = 0; batch < input_shape.n; ++batch) {
                for (int64_t output_h = 0; output_h < output_shape.h; ++output_h) {
                    for (int64_t output_w = 0; output_w < output_shape.w; ++output_w) {
                        for (int64_t output_channel = tail_begin; output_channel < output_channels; ++output_channel) {
                            const int64_t scalar_work_index =
                                ((batch * output_channels + output_channel) * output_shape.h + output_h) *
                                    output_shape.w + output_w;
                            compute_output(scalar_work_index);
                        }
                    }
                }
            }
            return failed.load(std::memory_order_relaxed) ? -1 : 0;
        }

        const int64_t fast_work_items = input_shape.n * output_spatial * oc8_blocks;
        auto compute_oc8 = [&](int64_t work_index) {
            if (failed.load(std::memory_order_relaxed)) {
                return;
            }
            int64_t remaining = work_index;
            const int64_t block = remaining % oc8_blocks;
            remaining /= oc8_blocks;
            const int64_t spatial_index = remaining % output_spatial;
            const int64_t batch = remaining / output_spatial;
            const int64_t output_h = spatial_index / output_shape.w;
            const int64_t output_w = spatial_index % output_shape.w;
            const int64_t output_channel_base = block * 8;

            __m256i accumulator = param->bias == nullptr
                                      ? _mm256_setzero_si256()
                                      : _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                                            param->bias->data<int32_t>() + output_channel_base));
            const int64_t input_plane = input_shape.h * input_shape.w;
            const int64_t input_x_origin = output_w * param->stride_w - param->pad_w;
            const int64_t kernel_x_begin = std::max<int64_t>(0, -input_x_origin);
            const int64_t kernel_x_end = std::min<int64_t>(kernel_w, input_shape.w - input_x_origin);

            const bool use_specialized_3x3 = use_maddubs_pairs && kernel_h == 3 && kernel_w == 3 &&
                                             param->stride_h == 1 && param->stride_w == 1 &&
                                             param->pad_h == 1 && param->pad_w == 1 &&
                                             output_h > 0 && output_h + 1 < input_shape.h &&
                                             input_x_origin >= 0 && input_x_origin + 2 < input_shape.w;
            if (use_specialized_3x3) {
                for (int64_t input_channel = 0; input_channel < input_channels_per_group; ++input_channel) {
                    const int64_t input_channel_base =
                        (batch * input_shape.c + input_channel) * input_plane;
                    for (int64_t kernel_y = 0; kernel_y < 3; ++kernel_y) {
                        const int8_t* input_row = input + input_channel_base +
                                                  (output_h - 1 + kernel_y) * input_shape.w + input_x_origin;
                        const int64_t pair_row = (input_channel * 3 + kernel_y) * 2;
                        const int8_t* pair_weight = packed_maddubs_weight +
                            (block * pair_rows_per_block + pair_row) * 32;
                        const __m128i third_weights = _mm_loadu_si128(
                            reinterpret_cast<const __m128i*>(pair_weight + 32));
                        accumulator = x86::DotInt8Oc8MaddubsPairPacked(
                            accumulator, input_row[0], input_row[1], pair_weight);
                        accumulator = x86::DotInt8Oc8MaddubsPacked(
                            accumulator, input_row[2], third_weights);
                    }
                }
            } else {
                for (int64_t input_channel = 0; input_channel < input_channels_per_group; ++input_channel) {
                    const int64_t input_channel_base =
                        (batch * input_shape.c + input_channel) * input_plane;
                    const int64_t packed_channel_base = block * patch_size + input_channel * kernel_h * kernel_w;
                    for (int64_t kernel_y = 0; kernel_y < kernel_h; ++kernel_y) {
                        const int64_t input_y =
                            output_h * param->stride_h + kernel_y * param->dilation_h - param->pad_h;
                        if (input_y < 0 || input_y >= input_shape.h || kernel_x_begin >= kernel_x_end) {
                            continue;
                        }
                        const int8_t* input_row = input + input_channel_base + input_y * input_shape.w +
                                                  input_x_origin + kernel_x_begin;
                        const int8_t* packed_row = packed_weight +
                                                    (packed_channel_base + kernel_y * kernel_w + kernel_x_begin) * 8;
                        // input_row and packed_row already point at kernel_x_begin;
                        // keep the local pair index zero-based to avoid applying
                        // the left-padding offset twice.
                        int64_t local_kernel_x = 0;
                        const int64_t valid_kernel_width = kernel_x_end - kernel_x_begin;
                        if (use_maddubs_pairs) {
                            if ((kernel_x_begin & 1) != 0 && local_kernel_x < valid_kernel_width) {
                                accumulator = x86::DotInt8Oc8Maddubs(
                                    accumulator, input_row[local_kernel_x], packed_row + local_kernel_x * 8);
                                ++local_kernel_x;
                            }
                            for (; local_kernel_x + 1 < valid_kernel_width; local_kernel_x += 2) {
                                const int64_t kernel_x = kernel_x_begin + local_kernel_x;
                                const int64_t pair_index = kernel_x / 2;
                                const int64_t pair_row =
                                    (input_channel * kernel_h + kernel_y) * pair_width + pair_index;
                                const int8_t* pair_weight = packed_maddubs_weight +
                                    (block * pair_rows_per_block + pair_row) * 32;
                                accumulator = x86::DotInt8Oc8MaddubsPairPacked(
                                    accumulator, input_row[local_kernel_x], input_row[local_kernel_x + 1], pair_weight);
                            }
                            if (local_kernel_x < valid_kernel_width) {
                                accumulator = x86::DotInt8Oc8Maddubs(
                                    accumulator, input_row[local_kernel_x], packed_row + local_kernel_x * 8);
                            }
                        } else {
                            for (; local_kernel_x + 1 < valid_kernel_width; local_kernel_x += 2) {
                                accumulator = x86::DotInt8Oc8Pair(
                                    accumulator, input_row[local_kernel_x], input_row[local_kernel_x + 1],
                                    packed_row + local_kernel_x * 8);
                            }
                            if (local_kernel_x < valid_kernel_width) {
                                accumulator = x86::DotInt8Oc8Maddubs(
                                    accumulator, input_row[local_kernel_x], packed_row + local_kernel_x * 8);
                            }
                        }
                    }
                }
            }

            alignas(32) int32_t accumulator_values[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(accumulator_values), accumulator);
            alignas(32) int8_t quantized_values[8];
            if (can_use_fast_requant &&
                FastQuantizeAccumulator8(accumulator_values, fast_requant_scales.data() + output_channel_base,
                                         output_quantization.zero_point, quantized_values)) {
                for (int64_t lane = 0; lane < 8; ++lane) {
                    const int64_t output_channel = output_channel_base + lane;
                    const int64_t output_offset =
                        ((batch * output_shape.c + output_channel) * output_shape.h + output_h) * output_shape.w +
                        output_w;
                    output[output_offset] = quantized_values[lane];
                }
                return;
            }
            for (int64_t lane = 0; lane < 8; ++lane) {
                const int64_t output_channel = output_channel_base + lane;
                const double scale = static_cast<double>(input_quantization.scale) *
                                     weight_quantization.scale_for(static_cast<size_t>(output_channel));
                const int64_t output_offset =
                    ((batch * output_shape.c + output_channel) * output_shape.h + output_h) * output_shape.w +
                    output_w;
                if (!int8_detail::QuantizeAccumulator(accumulator_values[lane], scale, output_quantization,
                                                      &output[output_offset])) {
                    failed.store(true, std::memory_order_relaxed);
                    return;
                }
            }
        };

#if defined(FEATHER_WITH_OPENMP)
        const int workers = std::max(1, omp_get_max_threads());
        if (fast_work_items >= 64 && workers > 1 && !omp_in_parallel()) {
#pragma omp parallel for schedule(static) num_threads(workers)
            for (int64_t work_index = 0; work_index < fast_work_items; ++work_index) {
                compute_oc8(work_index);
            }
        } else
#endif
        {
            for (int64_t work_index = 0; work_index < fast_work_items; ++work_index) {
                compute_oc8(work_index);
            }
        }

        // Complete output-channel tails with the checked scalar implementation.
        const int64_t tail_begin = oc8_blocks * 8;
        for (int64_t batch = 0; batch < input_shape.n; ++batch) {
            for (int64_t output_h = 0; output_h < output_shape.h; ++output_h) {
                for (int64_t output_w = 0; output_w < output_shape.w; ++output_w) {
                    for (int64_t output_channel = tail_begin; output_channel < output_channels; ++output_channel) {
                        const int64_t work_index =
                            ((batch * output_channels + output_channel) * output_shape.h + output_h) * output_shape.w +
                            output_w;
                        compute_output(work_index);
                    }
                }
            }
        }
        return failed.load(std::memory_order_relaxed) ? -1 : 0;
    }

#if defined(FEATHER_WITH_OPENMP)
    const int workers = std::max(1, omp_get_max_threads());
    if (work_items >= 4096 && workers > 1 && !omp_in_parallel()) {
#pragma omp parallel for schedule(static) num_threads(workers)
        for (int64_t work_index = 0; work_index < work_items; ++work_index) {
            compute_output(work_index);
        }
    } else
#endif
    {
        for (int64_t work_index = 0; work_index < work_items; ++work_index) {
            compute_output(work_index);
        }
    }
    return failed.load(std::memory_order_relaxed) ? -1 : 0;
}

}  // namespace

void EnsureX86Int8KernelsRegistered() {
    static const bool registered = []() {
        auto& dispatcher = KernelDispatcher::instance();
        dispatcher.registerKernel(DeviceType::X86, DataType::INT8, "FC", []() {
            return std::make_unique<FcKernel<DeviceType::X86, DataType::INT8>>();
        });
        dispatcher.registerKernel(DeviceType::X86, DataType::INT8, "Gemm", []() {
            return std::make_unique<GemmKernel<DeviceType::X86, DataType::INT8>>();
        });
        dispatcher.registerKernel(DeviceType::X86, DataType::INT8, "MatMul", []() {
            return std::make_unique<MatMulKernel<DeviceType::X86, DataType::INT8>>();
        });
        dispatcher.registerKernel(DeviceType::X86, DataType::INT8, "Conv2D", []() {
            return std::make_unique<Conv2DKernel<DeviceType::X86, DataType::INT8>>();
        });
        return true;
    }();
    (void)registered;
}

template <>
int32_t FcKernel<DeviceType::X86, DataType::INT8>::compute() {
    AutoTimer timer("X86::FC::INT8");
    return ComputeX86Int8Fc(static_cast<feather::operators::FcParam*>(param_));
}

template <>
int32_t GemmKernel<DeviceType::X86, DataType::INT8>::compute() {
    AutoTimer timer("X86::Gemm::INT8");
    return ComputeX86Int8Gemm(static_cast<feather::operators::GemmParam*>(param_));
}

template <>
int32_t MatMulKernel<DeviceType::X86, DataType::INT8>::compute() {
    AutoTimer timer("X86::MatMul::INT8");
    return ComputeX86Int8MatMul(static_cast<feather::operators::MatMulParam*>(param_));
}

template <>
int32_t Conv2DKernel<DeviceType::X86, DataType::INT8>::compute() {
    AutoTimer timer("X86::Conv2D::INT8");
    return ComputeX86Int8Conv2D(static_cast<feather::operators::Conv2dParam*>(param_),
                                &cached_int8_weight_tensor_, &cached_int8_weight_oc8_buffer_,
                                &cached_int8_maddubs_weight_tensor_, &cached_int8_maddubs_weight_buffer_,
                                &cached_int8_pointwise_maddubs_weight_tensor_,
                                &cached_int8_pointwise_maddubs_weight_buffer_,
                                &cached_int8_vnni_weight_tensor_, &cached_int8_weight_vnni_buffer_,
                                &cached_int8_weight_vnni_sums_);
}

}  // namespace kernel
}  // namespace feather
