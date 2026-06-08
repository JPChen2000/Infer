#include "src/kernel/x86/linear_fp32.h"

#include <immintrin.h>

namespace feather {
namespace kernel {
namespace x86 {

namespace {

inline __m256 LoadBiasVector(const float* bias, LinearBiasType bias_type, int64_t row, int64_t n, int64_t col) {
    if (bias == nullptr || bias_type == LinearBiasType::kNone) {
        return _mm256_setzero_ps();
    }
    if (bias_type == LinearBiasType::kVector) {
        return _mm256_loadu_ps(bias + col);
    }
    return _mm256_loadu_ps(bias + row * n + col);
}

inline float LoadBiasScalar(const float* bias, LinearBiasType bias_type, int64_t row, int64_t n, int64_t col) {
    if (bias == nullptr || bias_type == LinearBiasType::kNone) {
        return 0.0f;
    }
    if (bias_type == LinearBiasType::kVector) {
        return bias[col];
    }
    return bias[row * n + col];
}

}  // namespace

int32_t ComputeLinearRowMajorX86Fp32(const float* lhs, const float* rhs, const float* bias, int64_t m, int64_t k,
                                     int64_t n, LinearBiasType bias_type, float* out) {
    if (lhs == nullptr || rhs == nullptr || out == nullptr || m <= 0 || k <= 0 || n <= 0) {
        return -1;
    }

    for (int64_t i = 0; i < m; ++i) {
        const float* lhs_row = lhs + i * k;
        float* out_row = out + i * n;

        int64_t j = 0;
        for (; j + 8 <= n; j += 8) {
            __m256 acc = LoadBiasVector(bias, bias_type, i, n, j);
            for (int64_t t = 0; t < k; ++t) {
                const __m256 lhs_value = _mm256_set1_ps(lhs_row[t]);
                const __m256 rhs_value = _mm256_loadu_ps(rhs + t * n + j);
                acc = _mm256_fmadd_ps(lhs_value, rhs_value, acc);
            }
            _mm256_storeu_ps(out_row + j, acc);
        }

        for (; j < n; ++j) {
            float sum = LoadBiasScalar(bias, bias_type, i, n, j);
            for (int64_t t = 0; t < k; ++t) {
                sum += lhs_row[t] * rhs[t * n + j];
            }
            out_row[j] = sum;
        }
    }

    return 0;
}

}  // namespace x86
}  // namespace kernel
}  // namespace feather
