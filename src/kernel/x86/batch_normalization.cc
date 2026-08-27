#include "src/kernel/batch_normalization.h"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

#include "src/kernel/common/kernel_io.h"
#include "util/fp16.h"
#include "util/threading.h"
#include "util/timer.h"
#include "util/types.h"

#if defined(FEATHER_WITH_OPENMP)
#include <omp.h>
#endif

namespace feather {
namespace kernel {

namespace {

constexpr int64_t kBatchNormParallelWorkItems = 1 << 15;

bool ValidateBatchNorm(const operators::BatchNormParam* param, DataType dtype, ImageShape4D* shape) {
    if (param == nullptr || param->input == nullptr || param->scale == nullptr || param->bias == nullptr ||
        param->mean == nullptr || param->var == nullptr || param->out == nullptr || shape == nullptr ||
        param->input->data_type() != dtype ||
        (param->out->data_type() != DataType::UNKNOWN && param->out->data_type() != dtype) ||
        param->input->dims().size() != 4 || param->out->dims().data() != param->input->dims().data()) {
        return false;
    }
    if (!DecodeImageShape4D(param->input->dims().data(), NormalizeDataLayout(param->input->layout()), shape)) {
        return false;
    }
    return param->scale->numel() == shape->c && param->bias->numel() == shape->c &&
           param->mean->numel() == shape->c && param->var->numel() == shape->c &&
           param->scale->data_type() == dtype && param->bias->data_type() == dtype &&
           param->mean->data_type() == dtype && param->var->data_type() == dtype;
}

size_t BatchNormWorkerCount(int64_t work_items, int64_t values_per_item) {
    if (work_items < 2 || values_per_item <= 0 ||
        work_items > std::numeric_limits<int64_t>::max() / values_per_item ||
        work_items * values_per_item < kBatchNormParallelWorkItems) {
        return 1;
    }
#if defined(FEATHER_WITH_OPENMP)
    if (omp_in_parallel()) {
        return 1;
    }
    size_t workers = std::min(DefaultThreadCount(), static_cast<size_t>(work_items));
    const int openmp_limit = omp_get_max_threads();
    if (openmp_limit > 0) {
        workers = std::min(workers, static_cast<size_t>(openmp_limit));
    }
    return std::max<size_t>(1, workers);
#else
    return 1;
#endif
}

template <typename Fn>
void ParallelBatchNorm(int64_t work_items, int64_t values_per_item, Fn&& fn) {
    const size_t workers = BatchNormWorkerCount(work_items, values_per_item);
#if defined(FEATHER_WITH_OPENMP)
    if (workers > 1) {
#pragma omp parallel for schedule(static) num_threads(workers)
        for (int64_t index = 0; index < work_items; ++index) {
            fn(index);
        }
        return;
    }
#endif
    for (int64_t index = 0; index < work_items; ++index) {
        fn(index);
    }
}

inline __m256 ApplyBatchNorm(__m256 value, __m256 inv_std, __m256 scale, __m256 bias, __m256 mean) {
    return _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_sub_ps(value, mean), inv_std), scale), bias);
}

int32_t ComputeBatchNormFp32X86(operators::BatchNormParam* param) {
    ImageShape4D shape{};
    if (!ValidateBatchNorm(param, DataType::FP32, &shape)) {
        return -1;
    }
    param->out->set_data_type(DataType::FP32);
    param->out->set_layout(param->input->layout());

    const float* input = param->input->data<float>();
    float* output = param->out->mutable_data<float>();
    const float* scale = param->scale->data<float>();
    const float* bias = param->bias->data<float>();
    const float* mean = param->mean->data<float>();
    const float* variance = param->var->data<float>();
    const int64_t spatial = shape.h * shape.w;

    if (NormalizeDataLayout(param->input->layout()) == DataLayout::NCHW) {
        const int64_t work_items = shape.n * shape.c;
        ParallelBatchNorm(work_items, spatial, [&](int64_t work_index) {
            const int64_t n = work_index / shape.c;
            const int64_t c = work_index % shape.c;
            const float inv_std = 1.0f / std::sqrt(variance[c] + param->epsilon);
            const __m256 mean_vec = _mm256_set1_ps(mean[c]);
            const __m256 inv_vec = _mm256_set1_ps(inv_std);
            const __m256 scale_vec = _mm256_set1_ps(scale[c]);
            const __m256 bias_vec = _mm256_set1_ps(bias[c]);
            const int64_t base = (n * shape.c + c) * spatial;
            int64_t index = 0;
            for (; index + 8 <= spatial; index += 8) {
                _mm256_storeu_ps(output + base + index,
                                 ApplyBatchNorm(_mm256_loadu_ps(input + base + index), inv_vec, scale_vec, bias_vec,
                                                mean_vec));
            }
            for (; index < spatial; ++index) {
                output[base + index] = (input[base + index] - mean[c]) * inv_std * scale[c] + bias[c];
            }
        });
        return 0;
    }

    const int64_t rows = shape.n * spatial;
    ParallelBatchNorm(rows, shape.c, [&](int64_t row) {
        const int64_t base = row * shape.c;
        int64_t channel = 0;
        for (; channel + 8 <= shape.c; channel += 8) {
            const __m256 values = _mm256_loadu_ps(input + base + channel);
            const __m256 means = _mm256_loadu_ps(mean + channel);
            const __m256 variances = _mm256_loadu_ps(variance + channel);
            const __m256 scales = _mm256_loadu_ps(scale + channel);
            const __m256 biases = _mm256_loadu_ps(bias + channel);
            _mm256_storeu_ps(output + base + channel,
                             ApplyBatchNorm(values,
                                            _mm256_div_ps(_mm256_set1_ps(1.0f),
                                                          _mm256_sqrt_ps(_mm256_add_ps(
                                                              variances, _mm256_set1_ps(param->epsilon)))),
                                            scales, biases, means));
        }
        for (; channel < shape.c; ++channel) {
            const float inv_std = 1.0f / std::sqrt(variance[channel] + param->epsilon);
            output[base + channel] = (input[base + channel] - mean[channel]) * inv_std * scale[channel] + bias[channel];
        }
    });
    return 0;
}

inline __m256 LoadFp16x8(const uint16_t* input) {
    return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(input)));
}

inline void StoreFp16x8(__m256 value, uint16_t* output) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output),
                     _mm256_cvtps_ph(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

int32_t ComputeBatchNormFp16X86(operators::BatchNormParam* param) {
    ImageShape4D shape{};
    if (!ValidateBatchNorm(param, DataType::FP16, &shape)) {
        return -1;
    }
    param->out->set_data_type(DataType::FP16);
    param->out->set_layout(param->input->layout());

    const uint16_t* input = param->input->data<uint16_t>();
    uint16_t* output = param->out->mutable_data<uint16_t>();
    const uint16_t* scale = param->scale->data<uint16_t>();
    const uint16_t* bias = param->bias->data<uint16_t>();
    const uint16_t* mean = param->mean->data<uint16_t>();
    const uint16_t* variance = param->var->data<uint16_t>();
    const int64_t spatial = shape.h * shape.w;

    if (NormalizeDataLayout(param->input->layout()) == DataLayout::NCHW) {
        const int64_t work_items = shape.n * shape.c;
        ParallelBatchNorm(work_items, spatial, [&](int64_t work_index) {
            const int64_t n = work_index / shape.c;
            const int64_t c = work_index % shape.c;
            const float scale_value = HalfToFloat(scale[c]);
            const float bias_value = HalfToFloat(bias[c]);
            const float mean_value = HalfToFloat(mean[c]);
            const float inv_std = 1.0f / std::sqrt(HalfToFloat(variance[c]) + param->epsilon);
            const __m256 mean_vec = _mm256_set1_ps(mean_value);
            const __m256 inv_vec = _mm256_set1_ps(inv_std);
            const __m256 scale_vec = _mm256_set1_ps(scale_value);
            const __m256 bias_vec = _mm256_set1_ps(bias_value);
            const int64_t base = (n * shape.c + c) * spatial;
            int64_t index = 0;
            for (; index + 8 <= spatial; index += 8) {
                StoreFp16x8(ApplyBatchNorm(LoadFp16x8(input + base + index), inv_vec, scale_vec, bias_vec, mean_vec),
                            output + base + index);
            }
            for (; index < spatial; ++index) {
                output[base + index] = FloatToHalf((HalfToFloat(input[base + index]) - mean_value) * inv_std *
                                                   scale_value + bias_value);
            }
        });
        return 0;
    }

    const int64_t rows = shape.n * spatial;
    ParallelBatchNorm(rows, shape.c, [&](int64_t row) {
        const int64_t base = row * shape.c;
        for (int64_t channel = 0; channel < shape.c; ++channel) {
            const float scale_value = HalfToFloat(scale[channel]);
            const float bias_value = HalfToFloat(bias[channel]);
            const float mean_value = HalfToFloat(mean[channel]);
            const float inv_std = 1.0f / std::sqrt(HalfToFloat(variance[channel]) + param->epsilon);
            output[base + channel] = FloatToHalf((HalfToFloat(input[base + channel]) - mean_value) * inv_std *
                                                 scale_value + bias_value);
        }
    });
    return 0;
}

}  // namespace

template <>
int32_t BatchNormalizationKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::BatchNormalization::FP32");
    return ComputeBatchNormFp32X86(static_cast<operators::BatchNormParam*>(param_));
}

template <>
int32_t BatchNormalizationKernel<DeviceType::X86, DataType::FP16>::compute() {
    AutoTimer timer("X86::BatchNormalization::FP16");
    return ComputeBatchNormFp16X86(static_cast<operators::BatchNormParam*>(param_));
}

void EnsureX86BatchNormalizationKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "BatchNormalization",
            []() { return std::make_unique<BatchNormalizationKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "BatchNormalization",
            []() { return std::make_unique<BatchNormalizationKernel<DeviceType::X86, DataType::FP16>>(); });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
