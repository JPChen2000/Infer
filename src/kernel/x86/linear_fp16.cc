#include "src/kernel/x86/linear_fp16.h"

#include <immintrin.h>

#include <vector>

#include "util/fp16.h"

namespace feather {
namespace kernel {
namespace x86 {

namespace {

inline __m256 LoadBiasVectorFp16(const uint16_t* bias, LinearBiasType bias_type, int64_t row, int64_t n,
                                 int64_t col) {
    if (bias == nullptr || bias_type == LinearBiasType::kNone) {
        return _mm256_setzero_ps();
    }
    const uint16_t* bias_ptr = bias_type == LinearBiasType::kVector ? (bias + col) : (bias + row * n + col);
    const __m128i half = _mm_loadu_si128(reinterpret_cast<const __m128i*>(bias_ptr));
    return _mm256_cvtph_ps(half);
}

inline float LoadBiasScalarFp16(const uint16_t* bias, LinearBiasType bias_type, int64_t row, int64_t n, int64_t col) {
    if (bias == nullptr || bias_type == LinearBiasType::kNone) {
        return 0.0f;
    }
    const uint16_t value = bias_type == LinearBiasType::kVector ? bias[col] : bias[row * n + col];
    return HalfToFloat(value);
}

inline void ConvertHalfArrayToFp32(const uint16_t* input, int64_t count, float* output) {
    int64_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        const __m128i half = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + idx));
        const __m256 value = _mm256_cvtph_ps(half);
        _mm256_storeu_ps(output + idx, value);
    }

    for (; idx < count; ++idx) {
        output[idx] = HalfToFloat(input[idx]);
    }
}

inline void ConvertRhsPanelToFp32(const uint16_t* rhs, int64_t k, int64_t n, int64_t col, int64_t panel_width,
                                  float* rhs_panel) {
    for (int64_t t = 0; t < k; ++t) {
        const uint16_t* rhs_ptr = rhs + t * n + col;
        float* panel_ptr = rhs_panel + t * panel_width;
        const __m128i half0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rhs_ptr));
        const __m256 value0 = _mm256_cvtph_ps(half0);
        _mm256_storeu_ps(panel_ptr, value0);
        if (panel_width == 16) {
            const __m128i half1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rhs_ptr + 8));
            const __m256 value1 = _mm256_cvtph_ps(half1);
            _mm256_storeu_ps(panel_ptr + 8, value1);
        }
    }
}

}  // namespace

int32_t ComputeLinearRowMajorX86Fp16(const uint16_t* lhs, const uint16_t* rhs, const uint16_t* bias, int64_t m,
                                     int64_t k, int64_t n, LinearBiasType bias_type, uint16_t* out) {
    if (lhs == nullptr || rhs == nullptr || out == nullptr || m <= 0 || k <= 0 || n <= 0) {
        return -1;
    }

    std::vector<float> lhs_fp32(static_cast<size_t>(m * k), 0.0f);
    ConvertHalfArrayToFp32(lhs, m * k, lhs_fp32.data());

    std::vector<float> rhs_panel(static_cast<size_t>(k * 16), 0.0f);

    int64_t j = 0;
    for (; j + 16 <= n; j += 16) {
        ConvertRhsPanelToFp32(rhs, k, n, j, 16, rhs_panel.data());
        const bool has_vector_bias = bias != nullptr && bias_type == LinearBiasType::kVector;
        const __m256 vector_bias0 = has_vector_bias ? LoadBiasVectorFp16(bias, bias_type, 0, n, j) : _mm256_setzero_ps();
        const __m256 vector_bias1 =
            has_vector_bias ? LoadBiasVectorFp16(bias, bias_type, 0, n, j + 8) : _mm256_setzero_ps();
        for (int64_t i = 0; i < m; ++i) {
            const float* lhs_row = lhs_fp32.data() + i * k;
            uint16_t* out_row = out + i * n;
            __m256 acc0 = has_vector_bias ? vector_bias0 : LoadBiasVectorFp16(bias, bias_type, i, n, j);
            __m256 acc1 = has_vector_bias ? vector_bias1 : LoadBiasVectorFp16(bias, bias_type, i, n, j + 8);
            for (int64_t t = 0; t < k; ++t) {
                const __m256 lhs_value = _mm256_set1_ps(lhs_row[t]);
                const float* rhs_ptr = rhs_panel.data() + t * 16;
                acc0 = _mm256_fmadd_ps(lhs_value, _mm256_loadu_ps(rhs_ptr), acc0);
                acc1 = _mm256_fmadd_ps(lhs_value, _mm256_loadu_ps(rhs_ptr + 8), acc1);
            }
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out_row + j),
                             _mm256_cvtps_ph(acc0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out_row + j + 8),
                             _mm256_cvtps_ph(acc1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
    }

    for (; j + 8 <= n; j += 8) {
        ConvertRhsPanelToFp32(rhs, k, n, j, 8, rhs_panel.data());
        const bool has_vector_bias = bias != nullptr && bias_type == LinearBiasType::kVector;
        const __m256 vector_bias = has_vector_bias ? LoadBiasVectorFp16(bias, bias_type, 0, n, j) : _mm256_setzero_ps();
        for (int64_t i = 0; i < m; ++i) {
            const float* lhs_row = lhs_fp32.data() + i * k;
            uint16_t* out_row = out + i * n;
            __m256 acc = has_vector_bias ? vector_bias : LoadBiasVectorFp16(bias, bias_type, i, n, j);
            for (int64_t t = 0; t < k; ++t) {
                const __m256 lhs_value = _mm256_set1_ps(lhs_row[t]);
                const __m256 rhs_value = _mm256_loadu_ps(rhs_panel.data() + t * 8);
                acc = _mm256_fmadd_ps(lhs_value, rhs_value, acc);
            }
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out_row + j),
                             _mm256_cvtps_ph(acc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
    }

    for (int64_t i = 0; i < m; ++i) {
        const float* lhs_row = lhs_fp32.data() + i * k;
        uint16_t* out_row = out + i * n;
        for (int64_t tail_col = j; tail_col < n; ++tail_col) {
            float sum = LoadBiasScalarFp16(bias, bias_type, i, n, tail_col);
            for (int64_t t = 0; t < k; ++t) {
                sum += lhs_row[t] * HalfToFloat(rhs[t * n + tail_col]);
            }
            out_row[tail_col] = FloatToHalf(sum);
        }
    }

    return 0;
}

}  // namespace x86
}  // namespace kernel
}  // namespace feather
